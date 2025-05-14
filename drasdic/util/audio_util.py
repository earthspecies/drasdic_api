import torchaudio
import torch

def load_audio(fp, target_sr):
    audio, file_sr = torchaudio.load(fp)
    
    if file_sr != target_sr:
        print("resampling", fp, file_sr, target_sr)
        audio = torchaudio.functional.resample(audio, file_sr, target_sr)
    
    # correct DC offset
    audio = audio-torch.mean(audio, -1, keepdim=True)
    
    if len(audio.size()) == 2:
        # convert to mono
        audio = torch.mean(audio, dim=0)
    
    return audio