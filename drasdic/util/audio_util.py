"""
Util for working with audio
"""

from typing import Union

import torch
import torchaudio
from torch import Tensor


def load_audio(fp: Union[str, bytes], target_sr: int) -> Tensor:
    """
    Load an audio file, resample to target sampling rate if needed,
    remove DC offset, and convert to mono.

    Parameters
    ----------
    fp : str or bytes
        File path to the audio file.
    target_sr : int
        Target sampling rate in Hz.

    Returns
    -------
    audio : torch.Tensor
        1D tensor containing the mono audio waveform, resampled and DC-corrected.
    """
    audio, file_sr = torchaudio.load(fp)

    if file_sr != target_sr:
        print("resampling", fp, file_sr, target_sr)
        audio = torchaudio.functional.resample(audio, file_sr, target_sr)

    # correct DC offset
    audio = audio - torch.mean(audio, -1, keepdim=True)

    if len(audio.size()) == 2:
        # convert to mono
        audio = torch.mean(audio, dim=0)

    return audio
