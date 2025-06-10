import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import yaml

from drasdic.util.raven_util import frames_to_st_dict
from drasdic.models.model import get_model
from drasdic.inference.inference_utils import SlidingQueryDataset, fill_holes, delete_short, selection_table_to_frame_labels, get_min_duration

class InferenceInterface:
    def __init__(self, args_fp, device=None):
        """
        args_fp: Path to model args file, which is YAML format
        device: If None, uses "cuda" if available.
        """
        with open(args_fp, "r") as f:
            args = yaml.safe_load(f)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(args)
        self.model = model.to(self.device)
        self.model.eval()
        self.args = args
        # Cache for support prompts. The 'short' mode is a single prompt;
        # the 'long' mode is a list (ensemble) created by centering a window on each event.
        self.cached_support = []         # For load_support()
        self.pos_label = None
        self.min_support_vox_dur = np.infty

    def load_support(self, support_audio, support_selection_table, pos_label="POS"):
        """
        Load a single support prompt. If audio is longer than expected support_duration_sec, truncates audio to that duration.
        
        support_audio: 1D tensor of raw audio. Expects sampling rate of 16kHz.
        support_selection_table: Support annotations as a pandas dataframe. Requires columns `Begin Time (s)`, `End Time (s)`, and `Annotation`.
        pos_label: label in "Annotation" column of selection table to use (all other labels are ignored)
        """
        self.pos_label = pos_label
        st = support_selection_table.copy()
        st["Annotation"] = st["Annotation"].map(lambda x : "POS" if x==self.pos_label else "NEG")
        sr = self.args['sr']
        total_duration = self.args['support_duration_sec']
        support_audio = support_audio[:sr * total_duration]
        # Convert selection table to frame-level labels.
        support_labels = selection_table_to_frame_labels(st, total_duration, sr, max_len=total_duration)
        support_labels = support_labels[:len(support_audio)] # in case audio is too short

        mindur = get_min_duration(st, max_len = total_duration)
        self.min_support_vox_dur = min(self.min_support_vox_dur, mindur) # update min support vox dur
        
        support_audio = support_audio.unsqueeze(0).to(self.device)
        support_labels = support_labels.unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoded = self.model.encode_support(support_audio, support_labels)
        self.cached_support.append(encoded)
        print(f"Support clip loaded, total n support clips = {len(self.cached_support)}")
        return encoded

    def load_support_long(self, support_audio, support_selection_table, pos_label="POS"):
        """
        Generate multiple support prompts by taking each event (from the selection table)
        as the center of a support window.
        
        support_audio: 1D tensor (raw audio). Expects sampling rate of 16kHz.
        support_selection_table: Support annotations as a pandas dataframe. Requires columns `Begin Time (s)`, `End Time (s)`, and `Annotation`.
        pos_label: label in "Annotation" column of selection table to use (all other labels are ignored)
        """ 
        self.pos_label=pos_label
        st = support_selection_table.copy()
        st["Annotation"] = st["Annotation"].map(lambda x : "POS" if x==self.pos_label else "NEG")
        sr = self.args['sr']
        window_size = int(self.args['support_duration_sec'] * sr)
        # Use the support window duration for label conversion.
        full_labels = selection_table_to_frame_labels(st,
                                                    len(support_audio) / sr,
                                                    sr)

        mindur = get_min_duration(st)
        self.min_support_vox_dur = min(self.min_support_vox_dur, mindur) # update min support vox dur
        # Create a binary mask for positive regions.
        pos_np = (full_labels == 2).cpu().numpy().astype(bool)
        # Detect start and end indices of positive regions.
        diff = np.diff(np.concatenate(([0], pos_np.astype(int))))
        starts = np.where(diff == 1)[0]
        encoded_prompts = []
        
        for s in starts:
            left_window_target_dur = window_size//3
            window_low = max(0,int(s-left_window_target_dur))
            left_window = support_audio[window_low:s]
            left_anno = full_labels[window_low:s]
            while len(left_window) < left_window_target_dur:
                left_window = torch.concatenate([support_audio, left_window])
                left_anno = torch.concatenate([full_labels, left_anno])
            left_window = left_window[-left_window_target_dur:]
            left_anno = left_anno[-left_window_target_dur:]

            right_window_target_dur = window_size - left_window_target_dur
            window_high = int(s+right_window_target_dur)
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
    
    def subsample_support_clips(self, max_n_support_clips, seed = 0):
        """
        Reduce the number of support clips used (inference time is linear in number of support windows).
        Randomly selects up to max_n_support_clips from self.cached_support

        max_n_support_clips: Nonnegative integer giving max number of support clips to retain
        seed: Integer random seed
        """
        if (max_n_support_clips == 0) or (len(self.cached_support) == 0):
            self.cached_support = []
        else:
            rng = np.random.default_rng(seed)
            idxs_to_keep = rng.permutation(np.arange(len(self.cached_support)))[:max_n_support_clips]
            self.cached_support = [self.cached_support[idx] for idx in idxs_to_keep]
        print(f"Restricted number of support clips to {len(self.cached_support)}")

    def _aggregate_logits(self, query_logits_windowed):
        """
        Replicate windowing:
          - Use the first quarter of the first window,
          - Flatten all middle windows,
          - And use the last quarter of the last window.
        """
        output_window_dur_samples = query_logits_windowed.size(-1)
        first_quarter_end = output_window_dur_samples // 4
        last_quarter_start = output_window_dur_samples - first_quarter_end
        logits_first = query_logits_windowed[0, :first_quarter_end]
        logits_middle = query_logits_windowed[:, first_quarter_end:last_quarter_start].flatten()
        logits_last = query_logits_windowed[-1, last_quarter_start:]
        aggregated = torch.cat([logits_first, logits_middle, logits_last])
        return aggregated

    def _get_ensemble_logits_for_query(self, query_dataloader, support_prompts):    
        """
        For each support prompt in support_prompts, run prediction over the query dataloader,
        perform the windowing aggregation, and then average the logits over the ensemble.
        """
        ensemble_logits = []
        for support_encoded in tqdm(support_prompts):
            query_logits_list = []
            for batch in tqdm(query_dataloader):
                query_audio = batch['query_audio'].to(self.device)
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

    def _postprocess_logits(self, logits, threshold = 0.5):
        """
        Postprocess logits exactly as in the evaluation code.
        """
        confs = torch.sigmoid(logits)
        preds = confs >= threshold
        max_hole = min(self.min_support_vox_dur * 0.5, 1)
        max_hole_samples = int((max_hole * self.args['sr']) // self.model.downsample_factor)
        preds = fill_holes(preds, max_hole_samples)
        min_pos = min(self.min_support_vox_dur * 0.5, 0.5)
        min_pos_samples = int((min_pos * self.args['sr']) // self.model.downsample_factor)
        preds = delete_short(preds, min_pos_samples)
        return preds, confs
    
    def predict_logits(self, query_audio, batch_size=1):
        sr = self.args['sr']
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

    def logits_to_selection_table(self, logits, query_starttime=0, threshold = 0.5):
        sr = self.args['sr']
        preds, confs = self._postprocess_logits(logits, threshold = threshold)
        selection_table = frames_to_st_dict(preds.to(torch.int) * 2,
                                     sr=sr // self.model.downsample_factor)
        
        selection_table = pd.DataFrame(selection_table)
        probs = []
        for i, row in selection_table.iterrows(): # add in confidences post-hoc
            startsample = int((sr // self.model.downsample_factor) * row["Begin Time (s)"])
            endsample = int((sr // self.model.downsample_factor) * row["End Time (s)"])
            avg_conf = float(torch.mean(confs[startsample:endsample]))
            probs.append(avg_conf)
        selection_table["Probability"] = pd.Series(probs)
        
        # Adjust the time shift by iterating over the keys that hold time values.
        for key in ["Begin Time (s)", "End Time (s)"]:
            selection_table[key] += query_starttime
        selection_table["Annotation"] = selection_table["Annotation"].map(lambda x : self.pos_label if x=="POS" else x)
        return selection_table

    def predict(self, query_audio, query_starttime=0, batch_size=1, threshold=0.5):
        """
        Predict on an arbitrarily long query recording using a sliding window approach.
        The windows use 50% overlap, and the logits from all windows
        are aggregated to form the final prediction.

        query_audio: 1D tensor containing the full query recording.  Expects sampling rate of 16kHz.
        query_starttime: Time offset to add, default = 0.
        batch_size: batch size to use for inference, default = 1.
        threshold: probability threshold over which to count detections
        """
        logits = self.predict_logits(query_audio, batch_size=batch_size)
        st = self.logits_to_selection_table(logits, query_starttime=query_starttime, threshold = threshold)
        return st
        