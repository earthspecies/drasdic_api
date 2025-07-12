"""
Scene generator for few-shot bioacoustic sound event detection
"""

import os
import warnings
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import yaml


class GaussianMixtureGenerator:
    """
    A generator for sampling from a Gaussian mixture model.

    Attributes:
        rng (np.random.Generator): Random number generator instance.
        n_comps (int): Number of Gaussian components in the mixture.
        p_comps (list of float): Probabilities associated with each component.
        comp_means (list of float): Means of the Gaussian components.
        comp_stds (list of float): Standard deviations of the Gaussian components.
    """

    def __init__(
        self, rng: np.random.Generator, p_comps: List[float], comp_means: List[float], comp_stds: List[float]
    ) -> None:
        """
        Initializes the GaussianMixtureGenerator.

        Args:
            rng (np.random.Generator): Random number generator instance.
            p_comps (list of float): Probabilities associated with each component.
            comp_means (list of float): Means of the Gaussian components.
            comp_stds (list of float): Standard deviations of the Gaussian components.
        """
        self.rng = rng
        self.n_comps = len(p_comps)
        self.p_comps = p_comps
        self.comp_means = comp_means
        self.comp_stds = comp_stds

    def generate(self) -> float:
        """
        Generates a random sample from the Gaussian mixture model.

        Returns:
            float: A randomly generated value from the mixture distribution.
        """
        comp = self.rng.choice(np.arange(self.n_comps), p=self.p_comps)
        x = self.rng.normal(loc=self.comp_means[comp], scale=self.comp_stds[comp])
        return x


class SceneGenerator:
    """
    A class for generating audio scenes based on a given configuration file.

    Attributes:
        cfg (dict): Configuration dictionary loaded from the provided YAML file.
        support_duration_sec (float): Duration of the support audio in seconds.
        support_dur_samples (int): Support duration in samples, based on the sampling rate.
        query_duration_sec (float): Duration of the query audio in seconds.
        query_dur_samples (int): Query duration in samples, based on the sampling rate.
        sr (int): Sampling rate in Hz.
        resamplers (dict): Dictionary to store resampler objects for different sampling rates.
        background_info: Dictionary containing background audio metadata.
        pseudovox_info: Dictionary containing pseudovox metadata and cluster groupings.
        RIRs: Dictionary of preloaded RIR (Room Impulse Response) tensors
    """

    def __init__(
        self, scene_generator_config_fp: str, support_duration_sec: float, query_duration_sec: float, sr: int
    ) -> None:
        """
        Initializes the SceneGenerator by loading a configuration file and setting parameters.

        Args:
            scene_generator_config_fp (str): File path to the scene generator configuration YAML file.
            support_duration_sec (float): Duration of the support segment in seconds.
            query_duration_sec (float): Duration of the query segment in seconds.
            sr (int): Sampling rate in Hz.
        """
        # Load configuration from YAML file
        with open(scene_generator_config_fp, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.support_duration_sec = support_duration_sec
        self.support_dur_samples = int(support_duration_sec * sr)
        self.query_duration_sec = query_duration_sec
        self.query_dur_samples = int(query_duration_sec * sr)
        self.sr = sr

        # Initialize resamplers for dynamic sample rate conversion
        self.resamplers = {}

        print("Computing metadata for scene generator")
        self.compute_metadata()

    def compute_metadata(self) -> None:
        """
        Computes and loads metadata for background audio, pseudovox,
        and room impulse responses (RIRs).

        Updates:
            - self.background_info: Dictionary containing background audio metadata.
            - self.pseudovox_info: Dictionary containing pseudovox metadata and cluster groupings.
            - self.RIRs: Dictionary of preloaded RIR (Room Impulse Response) tensors
        """

        # Gather background clips
        self.background_info = {}

        for info_fp, weight in zip(self.cfg["background_info_fps"], self.cfg["background_weights"], strict=False):
            # Load and filter background audio metadata
            df = pd.read_csv(info_fp)
            df = df[df["duration"] >= self.cfg["background_min_duration"]].reset_index(drop=True)

            # Update file paths to absolute locations
            df["audio_fp"] = df["audio_fp"].map(lambda x: os.path.join(self.cfg["data_parent_dir"], x))

            # Store metadata and weight for this background dataset
            self.background_info[info_fp] = {"weight": weight, "info": df}

        # Gather pseudovox metadata
        self.pseudovox_info = {}

        for info_fp, weight in zip(self.cfg["pseudovox_info_fps"], self.cfg["pseudovox_weights"], strict=False):
            # Load and filter pseudovox metadata
            df = pd.read_csv(info_fp)
            df["duration"] = df["End Time (s)"] - df["Begin Time (s)"]
            df = df[
                (df["duration"] >= self.cfg["pseudovox_min_duration"])
                & (df["duration"] <= self.cfg["pseudovox_max_duration"])
            ].reset_index(drop=True)

            # Exclude low-quality pseudovox
            df = df[~df["qf_exclude"]]

            # Update file paths to absolute locations
            df["pseudovox_audio_fp"] = df["pseudovox_audio_fp"].map(
                lambda x: os.path.join(self.cfg["data_parent_dir"], x)
            )

            # Store metadata and weight for this pseudovox dataset
            self.pseudovox_info[info_fp] = {"weight": weight, "info": df, "allowed_clusters": {}}

            # Compute allowed clusters based on size constraints
            for cc in self.cfg["cluster_columns"]:
                print(f"Computing allowed clusters of size {cc} for {info_fp}")

                # Identify clusters with valid sizes
                counts = df[cc].value_counts()
                counts = counts[
                    (counts >= self.cfg["pseudovox_min_clustersize"])
                    & (counts <= self.cfg["pseudovox_max_clustersize"])
                ]

                self.pseudovox_info[info_fp]["allowed_clusters"][cc] = pd.Series(counts.index)

        # Load RIR data
        RIR_info = pd.read_csv(self.cfg["RIR_info_fp"])

        # Update file paths to absolute locations
        RIR_info["audio_fp"] = RIR_info["audio_fp"].map(lambda x: os.path.join(self.cfg["data_parent_dir"], x))

        self.RIRs = {}

        for i, row in RIR_info.iterrows():
            fp = row["audio_fp"]

            # Load RIR audio into memory and normalize
            rir = self.load_audio(fp, self.sr)
            rir = rir / torch.linalg.vector_norm(rir, ord=2)  # Normalize using L2 norm

            self.RIRs[i] = rir

    def load_audio(self, fp: str, target_sr: int) -> torch.Tensor:
        """
        Loads an audio file and resamples it to the target sample rate if necessary.

        Args:
            fp (str): File path to the audio file.
            target_sr (int): Desired sample rate for the audio.

        Returns:
            torch.Tensor: The loaded and processed audio signal. If loading or resampling fails,
                          returns a tensor of silence (zeros) with a length of 16000 samples.
        """
        try:
            audio, file_sr = torchaudio.load(fp)
        except (OSError, RuntimeError) as e:
            warnings.warn(f"Error loading {fp}: {e}. Using silence instead", RuntimeWarning, stacklevel=2)
            return torch.zeros((16000,))

        # Resample if the file sample rate differs from the target sample rate
        if file_sr != target_sr:
            try:
                if (file_sr, target_sr) in self.resamplers:
                    audio = self.resamplers[(file_sr, target_sr)](audio)
                else:
                    self.resamplers[(file_sr, target_sr)] = torchaudio.transforms.Resample(
                        orig_freq=file_sr, new_freq=target_sr
                    )
                    audio = self.resamplers[(file_sr, target_sr)](audio)
            except (OSError, RuntimeError, ValueError) as e:
                warnings.warn(f"Error resampling {fp}: {e}. Using silence instead", RuntimeWarning, stacklevel=2)
                return torch.zeros((16000,))

        # Correct DC offset by subtracting the mean
        audio = audio - torch.mean(audio, -1, keepdim=True)

        # Convert stereo/multi-channel audio to mono by averaging across channels
        if len(audio.size()) == 2:
            audio = torch.mean(audio, dim=0)

        return audio

    def fill_holes(self, labels: torch.Tensor, max_hole: int) -> torch.Tensor:
        """
        Fills small gaps (holes) in a sequence of labels where value `2` represents a target region.

        Args:
            labels (torch.Tensor): A 1D tensor containing label values, where `2` represents a target region.
            max_hole (int): The maximum hole size to fill.

        Returns:
            torch.Tensor: A modified label tensor where small holes (gaps) in label `2` regions are filled.
        """
        # Create a mask where labels equal 2
        m = labels == 2

        # Identify positions where a `2` region stops
        stops = m[:-1] * ~m[1:]
        stops = torch.nonzero(stops)  # Indices where `2` regions end

        # Iterate over each stop position
        for stop in np.ravel(stops.numpy()):
            # Look ahead within the max_hole range to check if another `2` exists
            look_forward = m[stop + 1 : stop + 1 + max_hole]
            if torch.any(look_forward):
                # Find the next start of a `2` region and fill the gap
                next_start = torch.amin(torch.nonzero(look_forward)) + stop + 1
                m[stop:next_start] = True

        # Convert boolean mask back to label `2`
        m = m.float() * 2

        # Ensure original labels are preserved where `m` is lower
        m = torch.maximum(m, labels)

        return m

    def loop_with_crops(self, audio: torch.Tensor, desired_dur_samples: int, rng: np.random.Generator) -> torch.Tensor:
        """
        Loops an audio clip to a desired length, randomly cropping each segment before appending it.

        Args:
            audio (torch.Tensor): The input audio tensor (1D).
            desired_dur_samples (int): The target duration in samples.
            rng (numpy.random.Generator): Random number generator for reproducibility.

        Returns:
            torch.Tensor: The looped and cropped audio tensor with the specified duration.
                          Returns silence (zeros) if the input audio is too short.
        """
        dur_audio = audio.size(0)

        # Handle unexpectedly short audio
        if dur_audio < 10:
            warnings.warn("Encountered unexpectedly short audio, returning silence.", RuntimeWarning, stacklevel=2)
            return torch.zeros((16000,))

        # Generate a random crop from the input audio
        crop_start = rng.integers(low=0, high=dur_audio // 2)
        crop_end = rng.integers(low=dur_audio // 2 + 1, high=dur_audio)
        final_audio = audio[crop_start:crop_end]

        # Continue looping and cropping until the desired duration is met
        while final_audio.size(0) <= desired_dur_samples:
            crop_start = rng.integers(low=0, high=dur_audio // 2)
            crop_end = rng.integers(low=dur_audio // 2 + 1, high=dur_audio)
            final_audio = torch.cat([final_audio, audio[crop_start:crop_end]])

        # Trim to the exact desired duration
        final_audio = final_audio[:desired_dur_samples]

        return final_audio

    def add_pseudovox_to_clip(
        self,
        pseudovox_df: pd.DataFrame,
        audio: torch.Tensor,
        annotations: torch.Tensor,
        speed_adjust_rate: int,
        snr_db_generator: GaussianMixtureGenerator,
        rms_background: float,
        time_reverse_aug: bool,
        timegap_generator: GaussianMixtureGenerator,
        rir: torch.Tensor,
        snr_min: float,
        rng: np.random.Generator,
        label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inserts pseudovox into an audio clip at random positions.

        Args:
            pseudovox_df (pd.DataFrame): DataFrame containing pseudovox audio file paths.
            audio (torch.Tensor): The background audio tensor.
            annotations (torch.Tensor): Annotation tensor to track inserted pseudovox regions.
            speed_adjust_rate (float): Speed adjustment factor for pseudovox processing.
            snr_db_generator (GaussianMixtureGenerator): Generator for SNR values.
            rms_background (float): Root mean square (RMS) of the background audio.
            time_reverse_aug (bool): Whether to apply time-reversal augmentation.
            timegap_generator (GaussianMixtureGenerator): Generator for time gaps between pseudovox insertions.
            rir (torch.Tensor): Room impulse response (RIR) filter for reverberation.
            snr_min (float): Minimum SNR value allowed.
            rng (np.random.Generator): Random number generator for reproducibility.
            label (int): Label to assign to the annotations where pseudovox is inserted. 0: NEG, 1: UNK, 2: POS

        Returns:
            tuple:
                - torch.Tensor: The modified audio with pseudovox inserted.
                - torch.Tensor: Updated annotations indicating where pseudovox was placed.
        """
        # Initialize insertion pointer at a random position in the audio
        pointer = rng.integers(audio.size(0))

        for _, row in pseudovox_df.iterrows():
            # Load and preprocess pseudovox audio
            pseudovox = self.load_audio(row["pseudovox_audio_fp"], speed_adjust_rate)
            rms_pseudovox = torch.std(pseudovox)

            # Generate and clip SNR value
            snr_db = snr_db_generator.generate()
            snr_db = np.clip(snr_db, a_min=snr_min, a_max=self.cfg["pseudovox_snr_max"])

            # Adjust pseudovox amplitude to match target SNR
            pseudovox = pseudovox * (rms_background / rms_pseudovox) * (10 ** (0.05 * snr_db))

            # Apply time-reversal augmentation if enabled
            if time_reverse_aug:
                pseudovox = torch.flip(pseudovox, (0,))

            # Generate and apply a random time gap between insertions
            timegap_samples = int(max(timegap_generator.generate() * self.sr, 0))

            background_dur_samples = audio.size(0)
            pseudovox_start = pointer

            # Handle cases where pseudovox extends beyond the end of the clip
            if pseudovox_start >= background_dur_samples - pseudovox.size(0):
                off_right_end = rng.binomial(1, 0.5)  # Randomly decide if it should be clipped or wrapped
                if off_right_end:
                    pseudovox = pseudovox[: background_dur_samples - pseudovox_start]
                else:
                    pseudovox = pseudovox[pseudovox_start - background_dur_samples :]
                    pseudovox_start = 0

            # Store duration before applying RIR
            pseudovox_dur_before_rir = pseudovox.size(0)

            # Apply room impulse response (RIR) if enabled
            if self.cfg["use_RIR"]:
                pseudovox = F.fftconvolve(pseudovox, rir)
                # Ensure RIR-processed pseudovox does not exceed clip length
                if pseudovox_start >= background_dur_samples - pseudovox.size(0):
                    pseudovox = pseudovox[: background_dur_samples - pseudovox_start]

            # Mix pseudovox into the background audio
            audio[pseudovox_start : pseudovox_start + pseudovox.size(0)] += pseudovox

            # Update annotations with the inserted pseudovox label
            annotations[pseudovox_start : pseudovox_start + pseudovox_dur_before_rir] = torch.maximum(
                annotations[pseudovox_start : pseudovox_start + pseudovox_dur_before_rir],
                torch.full_like(annotations[pseudovox_start : pseudovox_start + pseudovox_dur_before_rir], label),
            )

            # Move the insertion pointer forward by the duration of the inserted pseudovox and time gap
            pointer = (pointer + pseudovox_dur_before_rir + timegap_samples) % background_dur_samples

            # Debugging breakpoint for label 2 if no annotations were added
            if (torch.amax(annotations) == 0) and label == 2:
                import pdb

                pdb.set_trace()

        return audio, annotations

    def generate_scene(self, seed: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Generates a synthetic acoustic scene with annotations.

        Args:
            seed (int): Random seed

        Returns:
            dict: A dictionary containing:
                - "support_audio" (torch.Tensor): The support audio clip.
                - "query_audio" (torch.Tensor): The query audio clip.
                - "support_labels" (torch.Tensor): Labels associated with the support audio.
                - "query_labels" (torch.Tensor): Labels associated with the query audio.
                - "scene_info" (dict): Additional metadata related to the generated scene.
        """

        rng = np.random.default_rng(seed)
        scene_info = {}

        # Generate list of background audio clips to sample from
        sources = sorted(self.background_info.keys())
        weights = np.array([self.background_info[s]["weight"] for s in sources])
        weights = weights / np.sum(weights)
        background_source = rng.choice(sources, p=weights)
        background_fps = list(
            self.background_info[background_source]["info"]["audio_fp"].sample(n=4, random_state=seed)
        )

        # Sample background audio for support and choose augmentations
        scene_info["support_background_fp"] = background_fps[0]
        scene_info["support_background_resample"] = rng.choice(self.cfg["resample_options"]).tolist()

        scene_info["support_background_fp_2"] = background_fps[2]
        scene_info["support_background_resample_2"] = rng.choice(self.cfg["resample_options"]).tolist()

        scene_info["support_background_mixup"] = rng.binomial(1, self.cfg["background_audio_mixup_p"])

        # Sample background audio for query and choose augmentations
        scene_info["background_audio_query_domain_shift"] = rng.binomial(
            1, self.cfg["background_audio_query_domain_shift_p"]
        )

        if scene_info["background_audio_query_domain_shift"]:
            scene_info["query_background_resample"] = rng.choice(self.cfg["resample_options"]).tolist()
            scene_info["query_background_fp"] = background_fps[1]

            scene_info["query_background_resample_2"] = rng.choice(self.cfg["resample_options"]).tolist()
            scene_info["query_background_fp_2"] = background_fps[3]

            scene_info["query_background_mixup"] = rng.binomial(1, self.cfg["background_audio_mixup_p"])
        else:
            scene_info["query_background_resample"] = scene_info["support_background_resample"]
            scene_info["query_background_fp"] = background_fps[0]

            scene_info["query_background_resample_2"] = scene_info["support_background_resample_2"]
            scene_info["query_background_fp_2"] = background_fps[2]

            scene_info["query_background_mixup"] = scene_info["support_background_mixup"]

        # Generate list of pseudo-labels to sample from when choosing pseudovox
        sources = sorted(self.pseudovox_info.keys())
        weights = np.array([self.pseudovox_info[s]["weight"] for s in sources])
        weights = weights / np.sum(weights)
        scene_info["pseudovox_source"] = rng.choice(sources, p=weights).tolist()

        scene_info["pseudolabel_column"] = rng.choice(self.cfg["cluster_columns"]).tolist()
        pseudolabels_allowed = self.pseudovox_info[scene_info["pseudovox_source"]]["allowed_clusters"][
            scene_info["pseudolabel_column"]
        ]

        if len(pseudolabels_allowed) < 3:
            import pdb

            pdb.set_trace()

        # Sample the pseudo-labels that are associated with the pseudovox that will appear in this scene
        # While loop handles the corner case where pseudo-labels appear in the background clips we chose.
        # if so, re-choose pseudolabels
        attempts = 0
        while attempts < 5:
            pseudolabels_to_possibly_include = list(pseudolabels_allowed.sample(2, random_state=seed))
            pvi = self.pseudovox_info[scene_info["pseudovox_source"]]["info"]
            pseudovox_to_possibly_include = pvi[
                pvi[scene_info["pseudolabel_column"]].isin(pseudolabels_to_possibly_include)
            ]
            original_audio_clips_represented = list(pseudovox_to_possibly_include["raw_audio_fp"].unique())
            isok = True
            for k in [
                "support_background_fp",
                "support_background_fp_2",
                "query_background_fp",
                "query_background_fp_2",
            ]:
                background_fp = scene_info[k]
                background_fn = os.path.basename(background_fp)
                if background_fn in original_audio_clips_represented:
                    isok = False
            if isok:
                break
            else:
                attempts += 1
                print("attempting again")

        # Record minimum SNR for the scene
        snr_min = self.cfg["pseudovox_snr_min"]
        scene_info["snr_min"] = float(snr_min)

        # Choose augmentations, rate, and snr for focal pseudovox
        scene_info["focal_duration"] = rng.choice(self.cfg["resample_options"]).tolist()
        scene_info["focal_time_reverse"] = rng.binomial(1, self.cfg["pseudovox_time_reverse_p"])
        scene_info["focal_rate_support"] = rng.choice(self.cfg["pseudovox_rates_per_sec"]).tolist()
        scene_info["focal_rate_query"] = rng.choice(self.cfg["pseudovox_rates_per_sec"]).tolist()

        snr_p_comps = rng.choice(self.cfg["pseudovox_snr_p_second_component"]).tolist()
        scene_info["focal_snr_p_comps"] = [snr_p_comps, 1 - snr_p_comps]
        scene_info["focal_snr_comp_means"] = rng.uniform(
            low=self.cfg["pseudovox_snr_mean_low"], high=self.cfg["pseudovox_snr_mean_high"], size=2
        ).tolist()
        scene_info["focal_snr_comp_stds"] = rng.uniform(
            low=self.cfg["pseudovox_snr_std_low"], high=self.cfg["pseudovox_snr_std_high"], size=2
        ).tolist()
        focal_snr_generator = GaussianMixtureGenerator(
            rng, scene_info["focal_snr_p_comps"], scene_info["focal_snr_comp_means"], scene_info["focal_snr_comp_stds"]
        )

        timegap_p_comps = rng.choice(self.cfg["pseudovox_timegap_p_second_component"]).tolist()
        scene_info["focal_timegap_p_comps"] = [timegap_p_comps, 1 - timegap_p_comps]

        scene_info["annotation_gap_min"] = rng.uniform(low=0, high=self.cfg["annotation_max_possible_time_gap"])
        scene_info["focal_rir_idx"] = rng.choice(list(self.RIRs.keys())).tolist()

        # Include positive probability for timegap between pseudovox to be == 0
        focal_timegap_is_zero = rng.binomial(1, self.cfg["p_timegap_is_zero"], size=2).tolist()
        scene_info["focal_timegap_comp_means"] = rng.triangular(
            left=self.cfg["pseudovox_timegap_mean_low"],
            mode=self.cfg["pseudovox_timegap_mean_low"],
            right=self.cfg["pseudovox_timegap_mean_high"],
            size=2,
        ).tolist()
        scene_info["focal_timegap_comp_stds"] = rng.uniform(
            low=self.cfg["pseudovox_timegap_std_low"], high=self.cfg["pseudovox_timegap_std_high"], size=2
        ).tolist()
        for i in range(2):
            if focal_timegap_is_zero[i]:
                scene_info["focal_timegap_comp_means"][i] = 0
                scene_info["focal_timegap_comp_stds"][i] = 0
        focal_timegap_generator = GaussianMixtureGenerator(
            rng,
            scene_info["focal_timegap_p_comps"],
            scene_info["focal_timegap_comp_means"],
            scene_info["focal_timegap_comp_stds"],
        )

        # Sample focal pseudovox
        scene_info["focal_c"] = pseudolabels_to_possibly_include[1]
        pvi = self.pseudovox_info[scene_info["pseudovox_source"]]["info"]
        focal_pseudovox_to_possibly_include = pvi[pvi[scene_info["pseudolabel_column"]] == scene_info["focal_c"]]

        # Choose augmentations, rate, and snr for nonfocal pseudovox
        scene_info["nonfocal_duration"] = rng.choice(self.cfg["resample_options"]).tolist()
        scene_info["nonfocal_time_reverse"] = rng.binomial(1, self.cfg["pseudovox_time_reverse_p"])

        scene_info["nonfocal_rate_support"] = rng.choice(self.cfg["pseudovox_rates_per_sec"]).tolist()
        scene_info["nonfocal_rate_query"] = rng.choice(self.cfg["pseudovox_rates_per_sec"]).tolist()

        snr_p_comps = rng.choice(self.cfg["pseudovox_snr_p_second_component"]).tolist()
        scene_info["nonfocal_snr_p_comps"] = [snr_p_comps, 1 - snr_p_comps]
        scene_info["nonfocal_snr_comp_means"] = rng.uniform(
            low=self.cfg["pseudovox_snr_mean_low"], high=self.cfg["pseudovox_snr_mean_high"], size=2
        ).tolist()
        scene_info["nonfocal_snr_comp_stds"] = rng.uniform(
            low=self.cfg["pseudovox_snr_std_low"], high=self.cfg["pseudovox_snr_std_high"], size=2
        ).tolist()
        nonfocal_snr_generator = GaussianMixtureGenerator(
            rng,
            scene_info["nonfocal_snr_p_comps"],
            scene_info["nonfocal_snr_comp_means"],
            scene_info["nonfocal_snr_comp_stds"],
        )

        scene_info["nonfocal_rir_idx"] = rng.choice(list(self.RIRs.keys())).tolist()

        # Include positive probability for timegap between pseudovox to be == 0
        timegap_p_comps = rng.choice(self.cfg["pseudovox_timegap_p_second_component"]).tolist()
        nonfocal_timegap_is_zero = rng.binomial(1, self.cfg["p_timegap_is_zero"], size=2).tolist()
        scene_info["nonfocal_timegap_p_comps"] = [timegap_p_comps, 1 - timegap_p_comps]
        scene_info["nonfocal_timegap_comp_means"] = rng.triangular(
            left=self.cfg["pseudovox_timegap_mean_low"],
            mode=self.cfg["pseudovox_timegap_mean_low"],
            right=self.cfg["pseudovox_timegap_mean_high"],
            size=2,
        ).tolist()
        scene_info["nonfocal_timegap_comp_stds"] = rng.uniform(
            low=self.cfg["pseudovox_timegap_std_low"], high=self.cfg["pseudovox_timegap_std_high"], size=2
        ).tolist()
        for i in range(2):
            if nonfocal_timegap_is_zero[i]:
                scene_info["nonfocal_timegap_comp_means"][i] = 0
                scene_info["nonfocal_timegap_comp_stds"][i] = 0
        nonfocal_timegap_generator = GaussianMixtureGenerator(
            rng,
            scene_info["nonfocal_timegap_p_comps"],
            scene_info["nonfocal_timegap_comp_means"],
            scene_info["nonfocal_timegap_comp_stds"],
        )

        # Sample nonfocal pseudovox
        scene_info["nonfocal_c"] = pseudolabels_to_possibly_include[0]
        pvi = self.pseudovox_info[scene_info["pseudovox_source"]]["info"]
        nonfocal_pseudovox_to_possibly_include = pvi[pvi[scene_info["pseudolabel_column"]] == scene_info["nonfocal_c"]]

        # Load background_audio
        audio_support = self.load_audio(
            scene_info["support_background_fp"], int(self.sr * scene_info["support_background_resample"])
        )
        audio_query = self.load_audio(
            scene_info["query_background_fp"], int(self.sr * scene_info["query_background_resample"])
        )

        audio_support_2 = self.load_audio(
            scene_info["support_background_fp_2"], int(self.sr * scene_info["support_background_resample_2"])
        )
        audio_query_2 = self.load_audio(
            scene_info["query_background_fp_2"], int(self.sr * scene_info["query_background_resample_2"])
        )

        # Loop and trim background audio to desired length
        audio_support = self.loop_with_crops(audio_support, self.support_dur_samples, rng)
        audio_query = self.loop_with_crops(audio_query, self.query_dur_samples, rng)
        audio_support_2 = self.loop_with_crops(audio_support_2, self.support_dur_samples, rng)
        audio_query_2 = self.loop_with_crops(audio_query_2, self.query_dur_samples, rng)

        # Apply mixup augmentation to background audio
        if scene_info["support_background_mixup"]:
            rms_background_audio_support = torch.std(audio_support)
            rms_background_audio_support_2 = torch.std(audio_support_2)
            audio_support_2 = (
                audio_support_2
                * rms_background_audio_support
                / torch.maximum(rms_background_audio_support_2, torch.full_like(rms_background_audio_support_2, 1e-10))
            )
            audio_support = audio_support + audio_support_2

        if scene_info["query_background_mixup"]:
            rms_background_audio_query = torch.std(audio_query)
            rms_background_audio_query_2 = torch.std(audio_query_2)
            audio_query_2 = (
                audio_query_2
                * rms_background_audio_query
                / torch.maximum(rms_background_audio_query_2, torch.full_like(rms_background_audio_query_2, 1e-10))
            )
            audio_query = audio_query + audio_query_2

        # Initialize labels
        support_labels = torch.zeros_like(audio_support)
        query_labels = torch.zeros_like(audio_query)

        # Equalize rms of support and query background audio
        rms_background_audio_support = torch.std(audio_support)
        audio_query = (
            audio_query
            * rms_background_audio_support
            / torch.maximum(torch.std(audio_query), torch.full_like(audio_query, 1e-10))
        )

        # Add pseudovox to scene, beginning with POS pseudovox
        for _, label in enumerate([2, 0]):
            # Label semantics:
            # 2 = Positive
            # 1 = Unknown
            # 0 = Negative; note there are already implicit negatives in background track

            # Get number of pseudovox to use
            pseudovox_rate_support = {2: scene_info["focal_rate_support"], 0: scene_info["nonfocal_rate_support"]}[
                label
            ]  # call rate per second
            pseudovox_rate_query = {2: scene_info["focal_rate_query"], 0: scene_info["nonfocal_rate_query"]}[
                label
            ]  # call rate per second

            n_pseudovox_support = rng.poisson(pseudovox_rate_support * self.support_duration_sec)
            n_pseudovox_query = rng.poisson(pseudovox_rate_query * self.query_duration_sec)

            if rng.binomial(1, 0.5):
                n_pseudovox_query = max(n_pseudovox_query, 1)  # reduce the number of queries with nothing happening

            if label == 2:
                n_pseudovox_support = max(n_pseudovox_support, 1)  # Require minimum 1 focal call in support

            # Make a list of the exact pseudovox that will be inserted
            possible_pseudovox = {2: focal_pseudovox_to_possibly_include, 0: nonfocal_pseudovox_to_possibly_include}[
                label
            ]

            pseudovox_query = possible_pseudovox.sample(n=n_pseudovox_query, replace=True, random_state=seed + 1)

            pseudovox_fps_in_query = list(
                pseudovox_query["pseudovox_audio_fp"].unique()
            )  # don't reuse pseudovox from query in support
            possible_pseudovox_not_in_query = possible_pseudovox[
                ~possible_pseudovox["pseudovox_audio_fp"].isin(pseudovox_fps_in_query)
            ]

            if len(possible_pseudovox_not_in_query) >= 2:
                pseudovox_support = possible_pseudovox_not_in_query.sample(
                    n=n_pseudovox_support, replace=True, random_state=seed
                )
            else:
                pseudovox_support = possible_pseudovox.sample(n=n_pseudovox_support, replace=True, random_state=seed)

            # Get augmentations
            snr_db_generator = {2: focal_snr_generator, 0: nonfocal_snr_generator}[label]
            timegap_generator = {2: focal_timegap_generator, 0: nonfocal_timegap_generator}[label]
            dur_aug = {2: scene_info["focal_duration"], 0: scene_info["nonfocal_duration"]}[label]
            speed_adjust_rate = int(dur_aug * self.sr)
            time_reverse_aug = {2: scene_info["focal_time_reverse"], 0: scene_info["nonfocal_time_reverse"]}[label]
            rir_idx = {2: scene_info["focal_rir_idx"], 0: scene_info["nonfocal_rir_idx"]}[label]
            rir = self.RIRs[rir_idx]

            # Add pseudovox to support audio
            audio_support, support_labels = self.add_pseudovox_to_clip(
                pseudovox_support,
                audio_support,
                support_labels,
                speed_adjust_rate,
                snr_db_generator,
                rms_background_audio_support,
                time_reverse_aug,
                timegap_generator,
                rir,
                scene_info["snr_min"],
                rng,
                label,
            )

            # Add pseudovox to query audio
            audio_query, query_labels = self.add_pseudovox_to_clip(
                pseudovox_query,
                audio_query,
                query_labels,
                speed_adjust_rate,
                snr_db_generator,
                rms_background_audio_support,
                time_reverse_aug,
                timegap_generator,
                rir,
                scene_info["snr_min"],
                rng,
                label,
            )

        # Simulate annotation by filling small holes between POS intervals
        support_labels = self.fill_holes(support_labels, int(scene_info["annotation_gap_min"] * self.sr))
        query_labels = self.fill_holes(query_labels, int(scene_info["annotation_gap_min"] * self.sr))

        # Package outputs
        out = {
            "support_audio": audio_support,
            "query_audio": audio_query,
            "support_labels": support_labels,
            "query_labels": query_labels,
            "scene_info": scene_info,
        }

        return out


if __name__ == "__main__":
    """
    Demo Usage: python scene_generator.py --cfg-path=/path/to/data/cfg.yaml --out-path=/path/to/out/dir
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-fp", type=str, required=True, help="path to configuration file")
    parser.add_argument("--out-dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--sr", type=int, default=16000, help="Output audio sample rate")
    parser.add_argument("--support-duration-sec", type=float, default=30.0, help="Duration of support audio")
    parser.add_argument("--query-duration-sec", type=float, default=10.0, help="Duration of query audio")
    parser.add_argument("--n-scenes", type=int, default=20, help="N scenes to generate")
    args = parser.parse_args()

    # Set output dir
    output_dir = args.out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize scene generator
    generator = SceneGenerator(
        scene_generator_config_fp=args.cfg_fp,
        support_duration_sec=args.support_duration_sec,
        query_duration_sec=args.query_duration_sec,
        sr=args.sr,
    )

    # Generate Scenes
    for i in range(args.n_scenes):
        scene = generator.generate_scene(i)
        support_audio = scene["support_audio"]
        support_labels = scene["support_labels"]
        query_audio = scene["query_audio"]
        query_labels = scene["query_labels"]
        scene_info = scene["scene_info"]

        torchaudio.save(
            os.path.join(output_dir, f"audio_{i}.wav"),
            torch.cat([support_audio, query_audio], dim=0).unsqueeze(0),
            args.sr,
        )
        print(librosa.get_duration(os.path.join(output_dir, f"audio_{i}.wav")))  # to make checks happy; not used)
        with open(os.path.join(output_dir, f"scene_info_{i}.yaml"), "w") as f:
            yaml.dump(scene_info, f)

        # Make selection table that can be read in Raven
        labels = torch.cat([support_labels, query_labels], dim=0).numpy() == 2
        starts = labels[1:] * ~labels[:-1]
        starts = np.where(starts)[0] + 1
        if labels[0]:
            starts = np.insert(starts, 0, 0)

        d = {"Begin Time (s)": [], "End Time (s)": [], "Annotation": []}

        for start in starts:
            look_forward = labels[start:]
            ends = np.where(~look_forward)[0]
            if len(ends) > 0:
                end = start + np.amin(ends)
            else:
                end = len(labels) - 1
            d["Begin Time (s)"].append(start / args.sr)
            d["End Time (s)"].append(end / args.sr)
            d["Annotation"].append("POS")

        d = pd.DataFrame(d)
        d.to_csv(os.path.join(output_dir, f"selection_table_{i}.txt"), sep="\t", index=False)

    print(f"{args.n_scenes} example scenes have been saved to {output_dir}")
