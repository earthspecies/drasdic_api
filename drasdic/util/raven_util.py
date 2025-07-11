from typing import Dict, List, Union

import numpy as np
import torch


def frames_to_st_dict(x: torch.Tensor, sr: int) -> Union[Dict[str, List[float]], List[Dict[str, List[float]]]]:
    """
    Convert frame-level annotations into Raven-style selection table dict(s).

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (time,) or (batch, time), with integer entries:
        0 = NEG, 1 = UNK, 2 = POS.
    sr : int
        Sampling rate in Hz.

    Returns
    -------
    dict or list of dict
        If input is 1D: a single dict.
        If input is 2D: a list of dicts.
        Each dict contains:
            - "Begin Time (s)" : list of float
            - "End Time (s)" : list of float
            - "Annotation" : list of str ("POS" or "UNK")
    """

    if len(x.size()) == 2:
        outs = []
        for i in range(x.size(0)):
            x_sub = x[i, :]
            outs.append(_frames_to_st_dict_single(x_sub, sr=sr))
        return outs
    else:
        return _frames_to_st_dict_single(x, sr=sr)


def _frames_to_st_dict_single(x: torch.Tensor, sr: int) -> Dict[str, List[Union[str, float]]]:
    """
    Convert a single sequence of frame-level labels into a Raven-style selection table.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor with integer entries:
        0 = NEG, 1 = UNK, 2 = POS.
    sr : int
        Sampling rate in Hz.

    Returns
    -------
    dict
        Selection table dictionary with:
            - "Begin Time (s)" : list of float
            - "End Time (s)" : list of float
            - "Annotation" : list of str ("POS" or "UNK")
    """
    d = {"Begin Time (s)": [], "End Time (s)": [], "Annotation": []}

    for label_i in [1, 2]:
        labels = x.cpu().numpy() == label_i  # POS : 2, UNK : 1, NEG : 0

        starts = np.where((~labels[:-1]) & (labels[1:]))[0] + 1
        if labels[0]:
            starts = np.insert(starts, 0, 0)

        ends = np.where((labels[:-1]) & (~labels[1:]))[0] + 1
        if labels[-1]:
            ends = np.append(ends, len(labels))

        for start, end in zip(starts, ends, strict=False):
            d["Begin Time (s)"].append(start / sr)
            d["End Time (s)"].append(end / sr)
            d["Annotation"].append("POS" if label_i == 2 else "UNK")

    return d
