"""
Utils for interface
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class SlidingQueryDataset(Dataset):
    """
    Dataset that yields overlapping windows from an input audio tensor.

    Parameters
    ----------
    audio : torch.Tensor
        1D waveform tensor.
    window_size : int
        Size of each window in samples.
    hop_size : int
        Hop size (stride) between windows in samples.
    """

    def __init__(self, audio: torch.Tensor, window_size: int, hop_size: int) -> None:
        self.audio = audio
        self.window_size = window_size
        self.hop_size = hop_size
        self.windows = []
        for start in range(0, len(audio) - window_size + 1, hop_size):
            self.windows.append(audio[start : start + window_size])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"query_audio": self.windows[idx]}


class QueryDataset(Dataset):
    """
    Dataset that yields the full query audio as a single item.

    Parameters
    ----------
    audio : torch.Tensor
        1D waveform tensor.
    """

    def __init__(self, audio: torch.Tensor) -> None:
        self.audio = audio

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"query_audio": self.audio}


def fill_holes(m: torch.Tensor, max_hole: int) -> torch.Tensor:
    """
    Fill short gaps (False regions) in a binary mask tensor.

    Parameters
    ----------
    m : torch.Tensor
        1D boolean tensor.
    max_hole : int
        Maximum gap size (in samples) to fill.

    Returns
    -------
    torch.Tensor
        Gap-filled mask.
    """
    stops = (m[:-1] & ~m[1:]).nonzero(as_tuple=True)[0]  # indices where holes start

    for stop in stops:
        look_forward = m[stop + 1 : stop + 1 + max_hole]
        if torch.any(look_forward):
            next_start = (look_forward.nonzero(as_tuple=True)[0].min() + stop + 1).item()
            m[stop:next_start] = True

    return m


def delete_short(m: torch.Tensor, min_pos: int) -> torch.Tensor:
    """
    Remove positive segments shorter than a given length.

    Parameters
    ----------
    m : torch.Tensor
        1D boolean tensor.
    min_pos : int
        Minimum number of samples to retain a segment.

    Returns
    -------
    torch.Tensor
        Mask with short segments removed.
    """
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


def selection_table_to_frame_labels(
    selection_table: pd.DataFrame, total_duration_sec: float, sr: int, max_len: Optional[float] = None
) -> torch.Tensor:
    """
    Convert a Raven-style selection table into a frame-level label tensor.

    Parameters
    ----------
    selection_table : pd.DataFrame
        Must contain columns: "Begin Time (s)", "End Time (s)", "Annotation".
    total_duration_sec : float
        Total duration of the audio.
    sr : int
        Sample rate.
    max_len : float, optional
        Maximum time to include annotations.

    Returns
    -------
    torch.Tensor
        1D float tensor with 0 = NEG, 2 = POS.
    """
    total_samples = int(total_duration_sec * sr)
    labels = torch.zeros(total_samples, dtype=torch.int)

    for _, sel in selection_table.iterrows():
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


def get_min_duration(support_selection_table: pd.DataFrame, max_len: Optional[float] = None) -> float:
    """
    Get the minimum duration of annotated events in a selection table.

    Parameters
    ----------
    support_selection_table : pd.DataFrame
        Selection table with "Begin Time (s)" and "End Time (s)" columns.
    max_len : float, optional
        Maximum time cutoff.

    Returns
    -------
    float
        Minimum duration, or 0.1 if no valid events.
    """
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


def load_audio(fp: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load audio file, resample, convert to mono, and remove DC offset.

    Parameters
    ----------
    fp : str
        File path to audio file.
    target_sr : int
        Target sample rate.

    Returns
    -------
    torch.Tensor
        1D audio tensor.
    """
    audio, file_sr = torchaudio.load(fp)

    if file_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=target_sr)
        audio = resampler(audio)

    # correct DC offset
    audio = audio - torch.mean(audio, -1, keepdim=True)

    if len(audio.size()) == 2:
        # convert to mono
        audio = torch.mean(audio, dim=0)

    return audio


def nms(st: pd.DataFrame, iou_thresh: float = 0.8) -> pd.DataFrame:
    """
    Apply non-maximum suppression (NMS) to a selection table.

    Parameters
    ----------
    st : pd.DataFrame
        Selection table with columns "Begin Time (s)", "End Time (s)", and "Probability".
    iou_thresh : float
        IoU threshold for suppression.

    Returns
    -------
    pd.DataFrame
        Pruned selection table.
    """
    st = st.sort_values("Probability", ascending=False).reset_index(drop=True).copy()

    st["Remove"] = False
    st["Duration"] = st["End Time (s)"] - st["Begin Time (s)"]

    for i, row in st.iterrows():
        if row["Remove"]:
            continue
        xx1 = np.maximum(row["Begin Time (s)"], st["Begin Time (s)"].values)
        xx2 = np.minimum(row["End Time (s)"], st["End Time (s)"].values)
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = np.divide(inter, (row["Duration"] + st["Duration"].values - inter))
        ovr = pd.Series(ovr)
        remove = (ovr >= iou_thresh) & (st.index > i)
        st["Remove"] = remove | st["Remove"]

    st = st[~st["Remove"]].sort_values("Begin Time (s)", ascending=True).reset_index(drop=True).copy()

    return st
