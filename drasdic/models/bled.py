import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
from torchaudio.models import wav2vec2_model
import json
from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder
from einops import rearrange

class BLED(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.downsample_factor = args['downsample_factor']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.db_shift = self.args['bled_db_shift']
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=self.downsample_factor).to(device)
        self.freq_bins = torch.fft.rfftfreq(2048, d=1/args['sr']).to(device) # torch.arange((n + 1) // 2) / (d * n)
        assert args['support_features'] == ["lowfreq", "highfreq", "snr"]
        
    def freeze_audio_encoder(self):
        print("BLED doesn't freeze")

    def unfreeze_audio_encoder(self):
        print("BLED doesn't unfreeze")
    
    def encode_audio(self, audio, sampling_rate=None, is_support=False):
        
        expected_dur_output = audio.size(-1) // self.downsample_factor
        s = self.spectrogram(audio)
        s = rearrange(s, 'b c t -> b t c')
        
        pad = expected_dur_output- s.size(1)
        if pad > 0:
            s = torch.pad(s, (0,0,0,pad))
        s = s[:,:expected_dur_output,:]
        
        return s
        
    def downsample_labels(self, labels):
        expected_dur_output = labels.size(1) // self.downsample_factor
        labels = F.max_pool1d(labels.unsqueeze(1), self.downsample_factor, padding=0).squeeze(1) # (batch, time/scale_factor). 0=NEG 1=UNK 2=POS
        pad = expected_dur_output - labels.size(1)
        if pad>0:
            labels = F.pad(labels, (0,pad), mode='reflect')
            
        labels = labels[:,:expected_dur_output]
        return labels
    
    def encode_support(self, support_audio, support_labels, support_features):
        if self.args['inputs'] == "audio":
            assert False
            
        elif self.args['inputs'] == "features":
            # Semantics
            # [b,0,0] : low freq
            # [b,0,1] : high freq
            # [b,0,2] : snr
            
            support_encoded = support_features
            
            support_encoded = support_encoded.unsqueeze(1)
            
        elif self.args['inputs'] == "both":
            assert False
        
        return support_encoded
    
    def forward_with_precomputed_support(self, support_encoded, query_audio, query_labels = None):
        spectrogram = self.encode_audio(query_audio) # [batch, time, freq] power spectrum

        # For frame_number, outputs 1 (detection) or 0 (no detection) based on band-limited energy, as compared to noise floor
        freq_bins = torch.tile(self.freq_bins.unsqueeze(0).unsqueeze(0), (support_encoded.size(0),1,1))
        freq_bins_mask = (freq_bins>=support_encoded[:,:,0:1]) * (freq_bins<=support_encoded[:,:,1:2])
        freq_bins_mask = torch.tile(freq_bins_mask, (1,spectrogram.size(1),1))
        
        band_limited_spectrum = spectrogram * freq_bins_mask
        band_limited_energy = band_limited_spectrum.sum(dim = -1) # [batch,time]

        twenty_percent_noise_estimate = torch.quantile(band_limited_energy, q=0.2, dim = -1, keepdim=True)
        dB = 10*torch.log10(band_limited_energy/twenty_percent_noise_estimate)
        
        dB_threshold = support_encoded[:,:,2] + torch.full_like(support_encoded[:,:,2], self.db_shift)
        
        dets = dB > dB_threshold
        dets = dets.to(torch.float) # Note that BLED also involves a smoothing step, but this is handled by evaluation code already
        
        logits = 20*(dets - torch.ones_like(dets))  # Make "logits" out of binary detections
        
        if query_labels is not None:
            query_labels = self.downsample_labels(query_labels)
        
        return logits, query_labels
    
    def forward(self, support_audio, support_labels, query_audio, support_features = None, query_labels = None):
        
        support_encoded = self.encode_support(support_audio, support_labels, support_features)
        logits, query_labels = self.forward_with_precomputed_support(support_encoded, query_audio, query_labels = query_labels)
        return logits, query_labels
    
    def fill_holes(self, m, max_hole):
        # Only used in val epochs
        stops = m[:-1] * ~m[1:]
        stops = np.nonzero(stops)[0]

        for stop in np.ravel(stops):
            look_forward = m[stop+1:stop+1+max_hole]
            if np.any(look_forward):
                next_start = np.amin(np.nonzero(look_forward)[0]) + stop + 1
                m[stop : next_start] = True

        return m
        
    def predict_selection_table_dict(self, support_audio, support_labels, query_audio, sr=None, max_hole_sec = 0.1, support_features = None):
        # Only used in val epochs
        # sr is output samplerate
        sr = int(self.args['sr'] // self.downsample_factor)
        
        logits, _ = self.forward(support_audio, support_labels, query_audio, support_features = support_features)
        preds_binary = (logits > 0)
        preds_binary = preds_binary.cpu().numpy()
        
        all_outs = []
        for i in range(np.shape(preds_binary)[0]):
            pb = preds_binary[i,:]
            if max_hole_sec > 0:
                pb = self.fill_holes(pb, int(sr*max_hole_sec))
            d = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}

            starts = np.where((~pb[:-1]) & (pb[1:]))[0] + 1
            if pb[0]:
                starts = np.insert(starts, 0, 0)

            ends = np.where((pb[:-1]) & (~pb[1:]))[0] + 1
            if pb[-1]:
                ends = np.append(ends, len(pb))

            for start, end in zip(starts, ends):
                d["Begin Time (s)"].append(start/sr)
                d["End Time (s)"].append(end/sr)
                d["Annotation"].append("POS")
            all_outs.append(d)
                
        return all_outs