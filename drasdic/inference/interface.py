"""
Interface for working with trained DRASDIC model
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from drasdic.inference.inference_utils import (
    SlidingQueryDataset,
    delete_short,
    fill_holes,
    get_min_duration,
    selection_table_to_frame_labels,
)
from drasdic.models.model import get_model
from drasdic.util.raven_util import frames_to_st_dict


class InferenceInterface:
    """
    Interface for running inference using a trained DRASDIC model.

    Parameters
    ----------
    args_fp : str
        Path to YAML file containing model configuration.
    device : str or None, optional
        Device identifier (e.g., "cuda" or "cpu"). Defaults to "cuda" if available.
    """

    def __init__(self, args_fp: str, device: Optional[str] = None) -> None:
        with open(args_fp, "r") as f:
            args = yaml.safe_load(f)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(args)
        self.model = model.to(self.device)
        self.model.eval()
        self.args = args
        # Cache for support prompts. The 'short' mode is a single prompt;
        # the 'long' mode is a list (ensemble) created by centering a window on each event.
        self.cached_support = []  # For load_support()
        self.pos_label = None
        self.min_support_vox_dur = np.infty

    def load_support(
        self, support_audio: torch.Tensor, support_selection_table: pd.DataFrame, pos_label: str = "POS"
    ) -> torch.Tensor:
        """
        Load and encode a single support audio prompt.

        Parameters
        ----------
        support_audio : torch.Tensor
            1D tensor of raw audio (16kHz expected).
        support_selection_table : pd.DataFrame
            Selection table with columns: "Begin Time (s)", "End Time (s)", "Annotation".
        pos_label : str
            Label to treat as "POS". All others are treated as "NEG".

        Returns
        -------
        torch.Tensor
            Encoded support embedding (1, time, feature_dim).
        """
        self.pos_label = pos_label
        st = support_selection_table.copy()
        st["Annotation"] = st["Annotation"].map(lambda x: "POS" if x == self.pos_label else "NEG")
        sr = self.args["sr"]
        total_duration = self.args["support_duration_sec"]
        support_audio = support_audio[: sr * total_duration]
        # Convert selection table to frame-level labels.
        support_labels = selection_table_to_frame_labels(st, total_duration, sr, max_len=total_duration)
        support_labels = support_labels[: len(support_audio)]  # in case audio is too short

        mindur = get_min_duration(st, max_len=total_duration)
        self.min_support_vox_dur = min(self.min_support_vox_dur, mindur)  # update min support vox dur

        support_audio = support_audio.unsqueeze(0).to(self.device)
        support_labels = support_labels.unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoded = self.model.encode_support(support_audio, support_labels)
        self.cached_support.append(encoded)
        print(f"Support clip loaded, total n support clips = {len(self.cached_support)}")
        return encoded

    def load_support_long(
        self, support_audio: torch.Tensor, support_selection_table: pd.DataFrame, pos_label: str = "POS"
    ) -> List[torch.Tensor]:
        """
        Load and encode multiple support prompts by centering a support window on each event.

        Parameters
        ----------
        support_audio : torch.Tensor
            1D tensor of raw audio (16kHz expected).
        support_selection_table : pd.DataFrame
            Selection table with annotations.
        pos_label : str
            Label to treat as "POS".

        Returns
        -------
        List[torch.Tensor]
            List of encoded support embeddings.
        """
        self.pos_label = pos_label
        st = support_selection_table.copy()
        st["Annotation"] = st["Annotation"].map(lambda x: "POS" if x == self.pos_label else "NEG")
        sr = self.args["sr"]
        window_size = int(self.args["support_duration_sec"] * sr)
        # Use the support window duration for label conversion.
        full_labels = selection_table_to_frame_labels(st, len(support_audio) / sr, sr)

        mindur = get_min_duration(st)
        self.min_support_vox_dur = min(self.min_support_vox_dur, mindur)  # update min support vox dur
        # Create a binary mask for positive regions.
        pos_np = (full_labels == 2).cpu().numpy().astype(bool)
        # Detect start and end indices of positive regions.
        diff = np.diff(np.concatenate(([0], pos_np.astype(int))))
        starts = np.where(diff == 1)[0]
        encoded_prompts = []

        for s in starts:
            left_window_target_dur = window_size // 3
            window_low = max(0, int(s - left_window_target_dur))
            left_window = support_audio[window_low:s]
            left_anno = full_labels[window_low:s]
            while len(left_window) < left_window_target_dur:
                left_window = torch.concatenate([support_audio, left_window])
                left_anno = torch.concatenate([full_labels, left_anno])
            left_window = left_window[-left_window_target_dur:]
            left_anno = left_anno[-left_window_target_dur:]

            right_window_target_dur = window_size - left_window_target_dur
            window_high = int(s + right_window_target_dur)
            right_window = support_audio[s:window_high]
            right_anno = full_labels[s:window_high]
            while len(right_window) < right_window_target_dur:
                right_window = torch.concatenate([right_window, support_audio])
                right_anno = torch.concatenate([right_anno, full_labels])
            right_window = right_window[:right_window_target_dur]
            right_anno = right_anno[:right_window_target_dur]

            supp_audio_win = torch.concatenate([left_window, right_window])
            supp_labels_win = torch.concatenate([left_anno, right_anno])
            supp_audio_win = supp_audio_win.unsqueeze(0).to(self.device)
            supp_labels_win = supp_labels_win.unsqueeze(0).to(self.device)

            with torch.no_grad():
                encoded = self.model.encode_support(supp_audio_win, supp_labels_win)
            encoded_prompts.append(encoded)

        if len(encoded_prompts) == 0:
            print("Warning: no support events")
        self.cached_support.extend(encoded_prompts)
        print(f"Support clip loaded, total n support clips = {len(self.cached_support)}")
        return encoded_prompts

    def subsample_support_clips(self, max_n_support_clips: int, seed: int = 0) -> None:
        """
        Randomly subsample support prompts.

        Parameters
        ----------
        max_n_support_clips : int
            Maximum number of support prompts to keep.
        seed : int
            Random seed for reproducibility.
        """
        if (max_n_support_clips == 0) or (len(self.cached_support) == 0):
            self.cached_support = []
        else:
            rng = np.random.default_rng(seed)
            idxs_to_keep = rng.permutation(np.arange(len(self.cached_support)))[:max_n_support_clips]
            self.cached_support = [self.cached_support[idx] for idx in idxs_to_keep]
        print(f"Restricted number of support clips to {len(self.cached_support)}")

    def _aggregate_logits(self, query_logits_windowed: torch.Tensor) -> torch.Tensor:
        """
        Aggregate overlapping window logits into a continuous prediction.

        Parameters
        ----------
        query_logits_windowed : torch.Tensor
            Tensor of shape (n_windows, time).

        Returns
        -------
        torch.Tensor
            Aggregated 1D logits over time.
        """
        output_window_dur_samples = query_logits_windowed.size(-1)
        first_quarter_end = output_window_dur_samples // 4
        last_quarter_start = output_window_dur_samples - first_quarter_end
        logits_first = query_logits_windowed[0, :first_quarter_end]
        logits_middle = query_logits_windowed[:, first_quarter_end:last_quarter_start].flatten()
        logits_last = query_logits_windowed[-1, last_quarter_start:]
        aggregated = torch.cat([logits_first, logits_middle, logits_last])
        return aggregated

    def _get_ensemble_logits_for_query(
        self, query_dataloader: DataLoader, support_prompts: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute average logits over an ensemble of support prompts.

        Parameters
        ----------
        query_dataloader : DataLoader
            Dataloader yielding query audio windows.
        support_prompts : list of torch.Tensor
            List of encoded support prompts.

        Returns
        -------
        torch.Tensor
            1D tensor of ensembled logits.
        """
        ensemble_logits = []
        for support_encoded in tqdm(support_prompts):
            query_logits_list = []
            for batch in tqdm(query_dataloader):
                query_audio = batch["query_audio"].to(self.device)
                batch_size = query_audio.size(0)
                # Tile support encoding to match batch size.
                support_tiled = torch.tile(support_encoded, (batch_size, 1, 1))
                with torch.no_grad():
                    logits, _ = self.model.forward_with_precomputed_support(support_tiled, query_audio)
                query_logits_list.append(logits)
            query_logits_windowed = torch.cat(query_logits_list, dim=0)
            aggregated = self._aggregate_logits(query_logits_windowed)
            ensemble_logits.append(aggregated)
        # Average the logits across all support prompts.
        ensemble_logits = torch.stack(ensemble_logits, dim=0).mean(dim=0)
        return ensemble_logits

    def _postprocess_logits(self, logits: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sigmoid thresholding, hole filling, and minimum-duration filtering.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits from the model.
        threshold : float
            Threshold for binarization.

        Returns
        -------
        preds : torch.Tensor
            Binarized predictions (bool tensor).
        confs : torch.Tensor
            Sigmoid-transformed logits (probabilities).
        """
        confs = torch.sigmoid(logits)
        preds = confs >= threshold
        max_hole = min(self.min_support_vox_dur * 0.5, 1)
        max_hole_samples = int((max_hole * self.args["sr"]) // self.model.downsample_factor)
        preds = fill_holes(preds, max_hole_samples)
        min_pos = min(self.min_support_vox_dur * 0.5, 0.5)
        min_pos_samples = int((min_pos * self.args["sr"]) // self.model.downsample_factor)
        preds = delete_short(preds, min_pos_samples)
        return preds, confs

    def predict_logits(self, query_audio: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """
        Predict logits for a query audio clip using sliding windows.

        Parameters
        ----------
        query_audio : torch.Tensor
            1D audio tensor at 16kHz.
        batch_size : int
            Batch size for DataLoader.

        Raises
        -------
        ValueError
            If support has not been loaded

        Returns
        -------
        torch.Tensor
            1D logits tensor (before thresholding).
        """

        sr = self.args["sr"]
        window_len_sec = self.args.get("window_len_sec", 10)
        window_size = int(window_len_sec * sr)
        hop_size = window_size // 2

        query_ds = SlidingQueryDataset(query_audio, window_size, hop_size)
        query_dl = DataLoader(query_ds, batch_size=batch_size)
        if not self.cached_support:
            raise ValueError("Support not loaded. Call load_support() or load_support_long() first.")
        support_prompts = self.cached_support
        ensemble_logits = self._get_ensemble_logits_for_query(query_dl, support_prompts)
        return ensemble_logits

    def logits_to_selection_table(
        self, logits: torch.Tensor, query_starttime: float = 0, threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Convert logits to a selection table (Raven-style) using postprocessing.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits.
        query_starttime : float
            Time offset (in seconds) to add to all annotations.
        threshold : float
            Threshold for detection.

        Returns
        -------
        pd.DataFrame
            Selection table with columns: "Begin Time (s)", "End Time (s)", "Annotation", "Probability".
        """
        sr = self.args["sr"]
        preds, confs = self._postprocess_logits(logits, threshold=threshold)
        selection_table = frames_to_st_dict(preds.to(torch.int) * 2, sr=sr // self.model.downsample_factor)

        selection_table = pd.DataFrame(selection_table)
        probs = []
        for _, row in selection_table.iterrows():  # add in confidences post-hoc
            startsample = int((sr // self.model.downsample_factor) * row["Begin Time (s)"])
            endsample = int((sr // self.model.downsample_factor) * row["End Time (s)"])
            avg_conf = float(torch.mean(confs[startsample:endsample]))
            probs.append(avg_conf)
        selection_table["Probability"] = pd.Series(probs)

        # Adjust the time shift by iterating over the keys that hold time values.
        for key in ["Begin Time (s)", "End Time (s)"]:
            selection_table[key] += query_starttime
        selection_table["Annotation"] = selection_table["Annotation"].map(lambda x: self.pos_label if x == "POS" else x)
        return selection_table

    def predict(
        self, query_audio: torch.Tensor, query_starttime: float = 0, batch_size: int = 1, threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Full prediction pipeline: from raw query audio to postprocessed selection table.

        Parameters
        ----------
        query_audio : torch.Tensor
            1D query waveform (16kHz).
        query_starttime : float
            Offset to add to output times.
        batch_size : int
            Batch size for inference.
        threshold : float
            Detection threshold.

        Returns
        -------
        pd.DataFrame
            Selection table of detected segments.
        """
        logits = self.predict_logits(query_audio, batch_size=batch_size)
        st = self.logits_to_selection_table(logits, query_starttime=query_starttime, threshold=threshold)
        return st
