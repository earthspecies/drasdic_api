import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import yaml
from glob import glob
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import librosa

def get_peakfreq50(spec):
    # spec: [..., nmels, time]
    topmels = torch.argmax(spec, dim=-2)
    peak = float(torch.median(topmels.to(dtype=torch.float32)))
    return peak

def get_peakfreq10(spec):
    # spec: [..., nmels, time]
    topmels = torch.argmax(spec, dim=-2)
    peak = float(torch.quantile(topmels.to(dtype=torch.float32), 0.1))
    return peak

def get_peakfreq90(spec):
    # spec: [..., nmels, time]
    topmels = torch.argmax(spec, dim=-2)
    peak = float(torch.quantile(topmels.to(dtype=torch.float32), 0.9))
    return peak



def subselect_support_random(support_audio, support_anno, args, rng):
    sr = args['sr']
    support_dur_sec = args['support_duration_sec']
    
    chunk_size_samples = int(sr * args['support_subchunk_size_sec'])
    n_chunks_to_keep = int(support_dur_sec // args['support_subchunk_size_sec'])

    chunks_to_keep = []
    chunks_to_maybe_keep = []
        
    for chunk_start in torch.arange(0, len(support_audio), chunk_size_samples):
        chunk_end = min(chunk_start+chunk_size_samples, len(support_audio))
        annot_chunk = support_anno[chunk_start:chunk_end]

        if torch.amax(annot_chunk) >0:
            chunks_to_keep.append(int(chunk_start))
        else:
            chunks_to_maybe_keep.append(int(chunk_start))

    chunks_to_keep = rng.permutation(chunks_to_keep)[:n_chunks_to_keep]
    n_remaining = n_chunks_to_keep - len(chunks_to_keep)
    if n_remaining > 0:
        more_chunks_to_keep = rng.permutation(chunks_to_maybe_keep)[:n_remaining]
        chunks_to_keep = np.concatenate([chunks_to_keep, more_chunks_to_keep]).astype(int)

    chunks_to_keep = sorted(chunks_to_keep)

    support_audio_final = []
    support_anno_final = []
    for c in chunks_to_keep:
        chunk_end = min(c+chunk_size_samples, len(support_audio))
        support_audio_final.append(support_audio[c:chunk_end])
        support_anno_final.append(support_anno[c:chunk_end])
    support_audio_final=torch.concatenate(support_audio_final)
    support_anno_final=torch.concatenate(support_anno_final)

    support_dur_samples = int(support_dur_sec * sr)
    # if len(support_audio_final) < support_dur_samples:
    #     support_audio_final = torch.concatenate([support_audio_final, torch.zeros((support_dur_samples-len(support_audio_final),))])
    #     support_anno_final = torch.concatenate([support_anno_final, torch.zeros((support_dur_samples-len(support_anno_final),))])
    while len(support_audio_final) < support_dur_samples:
        support_audio_final = torch.concatenate([support_audio_final, support_audio_final])
        support_anno_final = torch.concatenate([support_anno_final, support_anno_final])
    support_audio_final = support_audio_final[:support_dur_samples]
    support_anno_final = support_anno_final[:support_dur_samples]

    if not len(support_audio_final) == support_dur_samples:
        import pdb; pdb.set_trace()
        
    return support_audio_final, support_anno_final

def subselect_support_fixed(support_audio, support_anno, args, voxn):
    sr = args['sr']
    support_dur_sec = args['support_duration_sec']
    
    voxn = voxn % args['n_shots']
    
    support_anno_binary = (support_anno > 0).cpu().numpy()
    starts = np.where((~support_anno_binary[:-1]) & (support_anno_binary[1:]))[0] + 1
    if support_anno_binary[0]:
        starts = np.insert(starts, 0, 0)
    starts = sorted(starts)
    
    if voxn >= len(starts):
        print("Too few support vox, re-using")
        voxn = voxn % len(starts)
    
    voxn_start = starts[voxn]
    
    window_dur_samples = int(support_dur_sec * sr)
    
    left_window_target_dur = window_dur_samples//3
    window_low = max(0,int(voxn_start-left_window_target_dur))
    left_window = support_audio[window_low:voxn_start]
    left_anno = support_anno[window_low:voxn_start]
    while len(left_window) < left_window_target_dur:
        left_window = torch.concatenate([support_audio, left_window])
        left_anno = torch.concatenate([support_anno, left_anno])
    left_window = left_window[-left_window_target_dur:]
    left_anno = left_anno[-left_window_target_dur:]
    
    right_window_target_dur = window_dur_samples - left_window_target_dur
    window_high = int(voxn_start+right_window_target_dur)
    right_window = support_audio[voxn_start:window_high]
    right_anno = support_anno[voxn_start:window_high]
    while len(right_window) < right_window_target_dur:
        right_window = torch.concatenate([right_window, support_audio])
        right_anno = torch.concatenate([right_anno, support_anno])
    right_window = right_window[:right_window_target_dur]
    right_anno = right_anno[:right_window_target_dur]
    
    final_audio = torch.concatenate([left_window, right_window])
    final_anno = torch.concatenate([left_anno, right_anno])
    
    return final_audio, final_anno

class TestDataset(Dataset):
    def __init__(self, audio_fp, anno_fp, seed, index, args):
        self.args = args
        self.audio_fp = audio_fp
        self.anno_fp = anno_fp
        self.n_shots = args['n_shots']
        self.eval_after = args['eval_after']
        self.window_len_sec = args["window_len_sec"]
        self.seed = seed
        self.index = index
        
        n_mels=256
        hop_length=160
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=self.args['downsample_factor'])
        self.freq_bins = torch.fft.rfftfreq(2048, d=1/args['sr']) # torch.arange((n + 1) // 2) / (d * n)

        
        self.support_features = args['support_features']
        
        if 'precomputed_features_fp' in args:
            with open(args['precomputed_features_fp'], 'r') as f:
                all_precomputed_features = yaml.safe_load(f)
            anno_fn = os.path.basename(anno_fp)
            if anno_fn in all_precomputed_features:
                self.precomputed_features = all_precomputed_features[anno_fn]
            else:
                print("Could not find associated anno fn for precomputed features")
                self.precomputed_features = {}
                
            if "highfreq" not in self.support_features:
                self.precomputed_features["High Freq (Hz)"] = self.args['sr'] /2
            if "lowfreq" not in self.support_features:
                self.precomputed_features["Low Freq (Hz)"] = 0
                    
        else:
            print("No precomputed features specified")
            self.precomputed_features = {"Low Freq (Hz)" : 0, "High Freq (Hz)" : self.args['sr']/2, "Duration" : 0.5}
        
        self.load_data()
    
    def load_audio(self, fp, target_sr):
        
        try:
            audio, file_sr = torchaudio.load(fp)
        except:
            try:
                audio, file_sr = librosa.load(fp, sr=None, mono=True)
                audio = torch.tensor(audio).unsqueeze(0)
            except:
                print(f"Error loading {fp}")
                return None

        if file_sr != target_sr:
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=target_sr)
                audio = resampler(audio)
            except:
                print(f"Error resampling {fp}")
                return torch.zeros((16000,))

        # correct DC offset
        audio = audio-torch.mean(audio, -1, keepdim=True)

        if len(audio.size()) == 2:
            # convert to mono
            audio = torch.mean(audio, dim=0)

        return audio
    
    def load_data(self):
        audio = self.load_audio(self.audio_fp, self.args['sr'])
        st = pd.read_csv(self.anno_fp)
        self.audiofilename = list(st["Audiofilename"])[0]
        
        st_sub = st[st["Q"] != "UNK"]
        support_endtime = sorted(st_sub["Endtime"])[self.n_shots-1]
        support_endsample = int(support_endtime * self.args['sr'])+1
        
        query_starttime = sorted(st_sub["Endtime"])[self.eval_after-1]
        query_startsample = int(query_starttime * self.args['sr'])+1
        
        labels = torch.zeros_like(audio)
        
        ambient_rms = max(1e-10, float(torch.std(audio)))
        
        st = st.sort_values('Endtime')
        
        for i, row in st.iterrows():
            begin_sample = int(row['Starttime'] * self.args['sr'])
            end_sample = max(begin_sample+1, int(row['Endtime'] * self.args['sr']))
            l = {"POS" : 2, "UNK" : 1, "NEG" : 0}[row["Q"]]
            labels[begin_sample:end_sample] = torch.maximum(labels[begin_sample:end_sample], torch.full_like(labels[begin_sample:end_sample], l))
        
        st_sub_support = st_sub[st_sub["Endtime"] <= support_endtime]
        assert len(st_sub_support) == self.n_shots
        self.min_support_vox_dur = (st_sub_support["Endtime"] - st_sub_support["Starttime"]).min()
        
        self.support_endtime = support_endtime
        self.query_starttime = query_starttime
        
        self.support_audio = audio[:support_endsample]
        self.support_labels = labels[:support_endsample]
        
        self.query_audio = audio[query_startsample:]
        self.query_labels = labels[query_startsample:]
        
        # Get band-limited snr
        event_spec = self.spectrogram(self.support_audio[self.support_labels>0])
        event_spec = rearrange(event_spec, 'c t -> t c').unsqueeze(0)
        background_audio = self.support_audio[self.support_labels==0]
        background_audio_sub = background_audio[:self.args['sr']*10]
        background_spec = self.spectrogram(background_audio_sub)
        background_spec = rearrange(background_spec, 'c t -> t c').unsqueeze(0)
        
        # For frame_number, outputs 1 (detection) or 0 (no detection) based on band-limited energy, as compared to noise floor
        freq_bins = self.freq_bins.unsqueeze(0).unsqueeze(0)
        freq_bins_mask = (freq_bins>=self.precomputed_features['Low Freq (Hz)']) * (freq_bins<=self.precomputed_features['High Freq (Hz)'])
        
        event_freq_bins_mask = torch.tile(freq_bins_mask, (1,event_spec.size(1),1))
        event_spec_bandlim = event_spec * event_freq_bins_mask
        event_energy = event_spec_bandlim.sum(dim=-1)
        event_energy_mean = event_energy.mean(dim=-1)
        
        background_freq_bins_mask = torch.tile(freq_bins_mask, (1,background_spec.size(1),1))
        background_spec_bandlim = background_spec * background_freq_bins_mask
        background_energy = background_spec_bandlim.sum(dim=-1)# [batch,time]

        twenty_percent_noise_estimate = torch.quantile(background_energy, q=0.2, dim = -1)
        
        snr_db = 10.*(torch.log10(event_energy_mean) - torch.log10(twenty_percent_noise_estimate)).squeeze()
        snr_db = torch.nan_to_num(snr_db, posinf=50, neginf=-50)
        
        self.precomputed_features['snr'] = snr_db
        
        # Get peak freq
        peakfreqs = torch.argmax(event_spec, dim = -1) #[1 t c]
        peakfreq_med_idx = int(torch.median(peakfreqs).numpy())
        peakfreq_med = self.freq_bins[peakfreq_med_idx]
        self.precomputed_features['peakfreq'] = peakfreq_med
        
        
    def get_features(self):
        feats = []
        for fname in self.support_features:
            if fname == "duration":
                feats.append(self.precomputed_features["Duration"])
            if fname == "snr":
                feats.append(self.precomputed_features["snr"])
            if fname == "highfreq":
                feats.append(min(self.args['sr']/2, self.precomputed_features["High Freq (Hz)"]))
            if fname == "lowfreq":
                feats.append(self.precomputed_features["Low Freq (Hz)"])
            if fname == "peakfreq":
                feats.append(self.precomputed_features["peakfreq"])
                
        feats = torch.tensor(feats).to(dtype=torch.float32)
        return feats
    
    def get_support(self):
        support_audio_known = self.support_audio[self.support_labels!=1]
        support_labels_known = self.support_labels[self.support_labels!=1]
        rng = np.random.default_rng(self.seed)
        
        features = self.get_features()
        if self.args['support_selection_method'] == "random":
            audio, labels = subselect_support_random(support_audio_known, support_labels_known, self.args, rng)
        
        elif self.args['support_selection_method'] == "fixed":
            audio, labels = subselect_support_fixed(support_audio_known, support_labels_known, self.args, self.index)
            
        return {"audio" : audio, "labels" : labels, "features" : features}
        
    
    def __len__(self):
        window_size_samples = int(self.window_len_sec * self.args['sr'])
        hop_size_samples = window_size_samples//2
        return len(torch.arange(0, self.query_audio.size(0), hop_size_samples))
        
    def __getitem__(self, idx):
        window_size_samples = int(self.window_len_sec * self.args['sr'])
        hop_size_samples = window_size_samples//2
        
        start_idx = idx*hop_size_samples
        audio = self.query_audio[start_idx:start_idx+window_size_samples]
        anno = self.query_labels[start_idx:start_idx+window_size_samples]
        
        if audio.size(0) < window_size_samples:
            audio = torch.nn.functional.pad(audio, (0, window_size_samples-audio.size(0)))
            anno = torch.nn.functional.pad(anno, (0, window_size_samples-anno.size(0)))
        
        scene = {"query_audio" : audio, "query_labels" : anno}
        return scene

def get_test_dataloader(audio_fp, anno_fp, args, seed = None, index=None, shuffle = False):
    if seed == None:
        seed = args['seed']
    
    dataset = TestDataset(audio_fp, anno_fp, seed, index, args)
    test_dataloader = DataLoader(dataset,
                                batch_size=args['batch_size'],
                                shuffle=shuffle,
                                num_workers=args['num_workers'],
                                pin_memory=True,
                                drop_last=False,
                               )
        
    return test_dataloader
