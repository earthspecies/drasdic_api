# Dataset that yields overlapping windows.
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import torchaudio

class SlidingQueryDataset(Dataset):
    def __init__(self, audio, window_size, hop_size):
        self.audio = audio
        self.window_size = window_size
        self.hop_size = hop_size
        self.windows = []
        for start in range(0, len(audio) - window_size + 1, hop_size):
            self.windows.append(audio[start:start+window_size])
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return {"query_audio": self.windows[idx]}

class QueryDataset(Dataset):
    def __init__(self, audio):
        self.audio = audio
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return {"query_audio": self.audio}

def fill_holes(m, max_hole):
    stops = (m[:-1] & ~m[1:]).nonzero(as_tuple=True)[0]  # indices where holes start

    for stop in stops:
        look_forward = m[stop + 1 : stop + 1 + max_hole]
        if torch.any(look_forward):
            next_start = (look_forward.nonzero(as_tuple=True)[0].min() + stop + 1).item()
            m[stop : next_start] = True

    return m

def delete_short(m, min_pos):
    starts = (m[1:] & ~m[:-1]).nonzero(as_tuple=True)[0] + 1

    clips = []

    for start in starts:
        look_forward = m[start:]
        ends = (~look_forward).nonzero(as_tuple=True)[0]
        if len(ends) > 0:
            clips.append((start.item(), start.item() + ends.min().item()))
            
    if m[0]:
        look_forward = m
        ends = (~look_forward).nonzero(as_tuple=True)[0]
        if len(ends) > 0:
            clips.append((start.item(), start.item() + ends.min().item()))

    # Create a new empty tensor of the same size
    m_new = torch.zeros_like(m, dtype=torch.bool)

    # Add back valid segments
    for clip in clips:
        if clip[1] - clip[0] >= min_pos:
            m_new[clip[0] : clip[1]] = True

    return m_new

def selection_table_to_frame_labels(selection_table, total_duration_sec, sr, max_len=None):
    """
    Converts a selection table (Pandas dataframe) into a 1D tensor of frame-level labels.
    
    Labels are zeros except in the regions marked with "POS", which are set to 2.

    This function accepts the Raven-style input format for each row:

            "Begin Time (s)": float,
            "End Time (s)": float,
            "Annotation": str

    Args:
        selection_table (pandas.DataFrame): Each dict represents an annotation/region.
        total_duration_sec (float): Duration of the entire audio in seconds.
        sr (int): Sample rate.

    Returns:
        torch.Tensor: A 1D tensor of length total_duration_sec*sr containing 0s and 2s.
    """
    total_samples = int(total_duration_sec * sr)
    labels = torch.zeros(total_samples, dtype=torch.int)

    for i, sel in selection_table.iterrows():        
        # Figure out which style of columns this entry has
        start_time = float(sel["Begin Time (s)"])
        end_time = float(sel["End Time (s)"])
        annotation = str(sel["Annotation"]).upper()

        if max_len and end_time > max_len:
            break
        
        start_index = int(start_time * sr)
        end_index = int(end_time * sr)
        
        # Adjust the condition below as needed if you want to label "UNK" or others as well
        if annotation == "POS":
            labels[start_index:end_index] = 2

    return labels.float()

def get_min_duration(support_selection_table, max_len = None):
    event_durations = []
    
    for _, row in support_selection_table.iterrows():
        if max_len and row["End Time (s)"] > max_len:
            continue
        dur = row["End Time (s)"] - row["Begin Time (s)"]
        event_durations.append(dur)

    if len(event_durations) > 0:
        return min(event_durations)
    else:
        # fallback if no events
        return 0.1

def load_audio(fp, target_sr=16000):
    audio, file_sr = torchaudio.load(fp)

    if file_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=target_sr)
        audio = resampler(audio)

    # correct DC offset
    audio = audio-torch.mean(audio, -1, keepdim=True)

    if len(audio.size()) == 2:
        # convert to mono
        audio = torch.mean(audio, dim=0)

    return audio


import numpy as np


def nms(
    st: pd.DataFrame,
    iou_thresh: float = 0.8,
) -> pd.DataFrame:
    """
    Applies non-maximal suppression to selection table

    st: pd dataframe selection table
    iou_thresh: threshold of overlap over which to do nms

    Returns
    -------
        selection table with nms applied
    """
    st = st.sort_values("Probability", ascending = False).reset_index(drop=True).copy()

    st["Remove"] = False
    st["Duration"] = st["End Time (s)"] - st["Begin Time (s)"]

    for i, row in st.iterrows():
        if row["Remove"]:
            continue
        xx1 = np.maximum(row["Begin Time (s)"], st["Begin Time (s)"].values)
        xx2 = np.minimum(row["End Time (s)"], st["End Time (s)"].values)
        inter = np.maximum(0.0, xx2-xx1)
        ovr = np.divide(inter, (row["Duration"] + st["Duration"].values - inter))
        ovr = pd.Series(ovr)
        remove = (ovr>=iou_thresh) & (st.index > i)
        st["Remove"] = remove | st["Remove"]
        st_sorted = st.sort_values("Begin Time (s)", ascending = True).reset_index(drop=True)

    st = st[~st["Remove"]].sort_values("Begin Time (s)", ascending = True).reset_index(drop=True).copy()

    return st