import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
from torchaudio.models import wav2vec2_model
import json
from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder
from einops import rearrange

from torch.amp import autocast

class FeatEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = len(args['support_features'])
        self.hidden_dim = 100
        self.out_dim = 768
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.mlp = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, self.out_dim)).to(device)
        
        self.type_enc = nn.Parameter(nn.init.kaiming_normal_(torch.zeros((1, 768)))).to(device)
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        return (self.mlp(x) + self.type_enc).unsqueeze(1)
    
class ConvEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.downsample_factor = self.args['downsample_factor']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        n_mels = args['n_mels']
        hidden_size = args['hidden_size']
        n_blocks = args['n_blocks']
        
        self.spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = self.args['sr'], n_fft=n_mels*8, hop_length = self.downsample_factor//2, n_mels=n_mels, f_min=10).to(device)
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=7, stride=1, padding="same", bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(hidden_size).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        
        self.pool1 = nn.AdaptiveAvgPool2d((n_mels//2, None)).to(device)

        # Residual block 1
        self.resblock1 = []
        for _ in range(n_blocks):
            self.resblock1.append(nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(hidden_size)
            )
                                 )
            
        self.resblock1 = nn.ModuleList(self.resblock1).to(device)
        
        self.pool2 = nn.AdaptiveAvgPool2d((n_mels//4, None)).to(device)

        # Residual block 2
        self.resblock2 = []
        for _ in range(n_blocks):
            self.resblock2.append(nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(hidden_size)
            )
                                 )
                            
        self.resblock2 = nn.ModuleList(self.resblock2).to(device)
        
        self.pool3 = nn.AdaptiveAvgPool2d((n_mels//8,None)).to(device)
        
        self.pool3b = nn.AvgPool1d(2).to(device) ###
                
        self.head = nn.Linear(hidden_size*(n_mels//8),768).to(device)
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        
    def forward(self, audio, sampling_rate=None, is_support=False):
        expected_output_dur = audio.size(1) // self.downsample_factor
        
        x = self.spectrogram(audio) # b c t
        x = torch.log(x + torch.full_like(x, 1e-10)) 
        
        x = x.unsqueeze(1) # b 1 c t
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        for b in self.resblock1:
            x = b(x)+x
        x = self.pool2(x)
        
        for b in self.resblock2:
            x = b(x)+x
        x = self.pool3(x) # b 64 16 t
        
        x = torch.reshape(x, (x.size(0), -1, x.size(-1)))
        x = F.adaptive_avg_pool1d(x, expected_output_dur)
        
        x = rearrange(x, "b c t -> b t c")
        x = self.head(x)
        
        return x
    
class ConvTransEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.downsample_factor = self.args['downsample_factor']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        n_mels = args['n_mels']
        hidden_size = args['hidden_size']
        n_blocks = args['n_blocks']
        
        self.spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = self.args['sr'], n_fft=n_mels*8, hop_length = self.downsample_factor//2, n_mels=n_mels, f_min=10).to(device)
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=7, stride=1, padding="same", bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(hidden_size).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        
        self.pool1 = nn.AdaptiveAvgPool2d((n_mels//2, None)).to(device)

        # Residual block 1
        self.resblock1 = []
        for _ in range(n_blocks):
            self.resblock1.append(nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(hidden_size)
            )
                                 )
            
        self.resblock1 = nn.ModuleList(self.resblock1).to(device)
        
        self.pool2 = nn.AdaptiveAvgPool2d((n_mels//4, None)).to(device)

        # Residual block 2
        self.resblock2 = []
        for _ in range(n_blocks):
            self.resblock2.append(nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(hidden_size)
            )
                                 )
                            
        self.resblock2 = nn.ModuleList(self.resblock2).to(device)
        
        self.pool3 = nn.AdaptiveAvgPool2d((n_mels//8,None)).to(device)
        
        self.pool3b = nn.AvgPool1d(2).to(device) ###
        
        self.head = ContinuousTransformerWrapper(
            dim_in = hidden_size*(n_mels//8),
            dim_out = 768,
            max_seq_len = 1+int((args['support_duration_sec'] + args['query_duration_sec'])*args['sr']//self.downsample_factor),
            attn_layers = Encoder(
                dim = 768,
                depth = 12,
                heads = 12,
                attn_flash=True,
                ff_swish = True,
                ff_glu = True,
                rotary_pos_emb = True,
            )
        ).to(device)
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        
    def forward(self, audio, sampling_rate=None, is_support=False):
        expected_output_dur = audio.size(1) // self.downsample_factor
        
        x = self.spectrogram(audio) # b c t
        x = torch.log(x + torch.full_like(x, 1e-10)) 
        
        x = x.unsqueeze(1) # b 1 c t
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        for b in self.resblock1:
            x = b(x)+x
        x = self.pool2(x)
        
        for b in self.resblock2:
            x = b(x)+x
        x = self.pool3(x) # b 64 16 t
        
        x = torch.reshape(x, (x.size(0), -1, x.size(-1)))
        x = F.adaptive_avg_pool1d(x, expected_output_dur)
        
        x = rearrange(x, "b c t -> b t c")
        x = self.head(x)
        
        return x

class AvesEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        aves_config_fp = "/home/jupyter/sound_event_detection/weights/birdaves-biox-base.torchaudio.model_config.json"
        # aves_config_fp = "/home/ubuntu/foundation-model-storage/audio-to-text-llm/NatureLM/configs/birdaves_bioxbase.config"
        aves_url = "https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.pt"
        self.downsample_factor = 320
        
        with open(aves_config_fp, 'r') as ff:
            config = json.load(ff)
        
        self.encoder = wav2vec2_model(**config, aux_num_out=None)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        state_dict = torch.hub.load_state_dict_from_url(aves_url, map_location=device)
        self.encoder.load_state_dict(state_dict)
        
    def freeze(self):
        for param in self.encoder.encoder.parameters():
            param.requires_grad = False
        self.encoder.feature_extractor.requires_grad_(False)
        
    def unfreeze(self):
        for param in self.encoder.encoder.parameters():
            param.requires_grad = True
        self.encoder.feature_extractor.requires_grad_(True)
        
    def forward(self, audio, sampling_rate=None, is_support=False):
        expected_dur_output = audio.size(1) // self.downsample_factor

        feats = self.encoder.extract_features(audio)[0][-1]

        pad = expected_dur_output - feats.size(1)
        if pad>0:
            feats = F.pad(feats, (0,0,0,pad), mode='reflect')
        feats = feats[:,:expected_dur_output,:]
        return feats
    
class BEATsEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        from drasdic.models.beats import BEATs, BEATsConfig
        self.args = args
        self.downsample_factor = 320
        
        beats_ckpt = torch.load(args['beats_checkpoint_fp'], map_location='cpu')
        beats_cfg = BEATsConfig(beats_ckpt['cfg'])
        self.encoder = BEATs(beats_cfg)
        self.encoder.load_state_dict(beats_ckpt['model'])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(device)
        
    def freeze(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        self.encoder.eval()
        
    def unfreeze(self):
        for name, param in self.encoder.encoder.parameters():
            param.requires_grad = True
        self.encoder.train(True)
        
    def forward(self, audio, sampling_rate=None, is_support=False):
        expected_dur_output = audio.size(1) // self.downsample_factor

        feats = self.encoder.extract_features(audio, feature_only=True)[0]

        pad = expected_dur_output - feats.size(1)
        if pad>0:
            feats = F.pad(feats, (0,0,0,pad), mode='reflect')
        feats = feats[:,:expected_dur_output,:]
        return feats

class EncoderDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args['encoder'] == "aves":
            self.encoder = AvesEncoder(args)
        elif args['encoder'] == "convnet":
            self.encoder = ConvEncoder(args)
        elif args['encoder'] == "beats":
            self.encoder = BEATsEncoder(args)
            
        self.feat_encoder = FeatEncoder(args)
        
        self.downsample_factor = self.encoder.downsample_factor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if 'encoder_size' in args:
            if args['encoder_size'] == "tiny":
                print("Using encoder size = tiny")
                self.decoder = ContinuousTransformerWrapper(
                    dim_in = 768,
                    dim_out = 512,
                    max_seq_len = 1+int((args['support_duration_sec'] + args['query_duration_sec'])*args['sr']//self.downsample_factor),
                    attn_layers = Encoder(
                        dim = 128,
                        depth = 2,
                        heads = 2,
                        attn_flash=True,
                        ff_swish = True,
                        ff_glu = True,
                        rotary_pos_emb = True,
                    )
                )
            if args['encoder_size'] == "small":
                print("Using encoder size = small")
                self.decoder = ContinuousTransformerWrapper(
                    dim_in = 768,
                    dim_out = 512,
                    max_seq_len = 1+int((args['support_duration_sec'] + args['query_duration_sec'])*args['sr']//self.downsample_factor),
                    attn_layers = Encoder(
                        dim = 512,
                        depth = 4,
                        heads = 8,
                        attn_flash=True,
                        ff_swish = True,
                        ff_glu = True,
                        rotary_pos_emb = True,
                    )
                )
            elif args['encoder_size'] == "base":
                print("Using encoder size = base")
                self.decoder = ContinuousTransformerWrapper(
                    dim_in = 768,
                    dim_out = 512,
                    max_seq_len = 1+int((args['support_duration_sec'] + args['query_duration_sec'])*args['sr']//self.downsample_factor),
                    attn_layers = Encoder(
                        dim = 768,
                        depth = 12,
                        heads = 12,
                        attn_flash=True,
                        ff_swish = True,
                        ff_glu = True,
                        rotary_pos_emb = True,
                    )
                )
                
        else:
            print("Using encoder size = base")
            self.decoder = ContinuousTransformerWrapper(
                dim_in = 768,
                dim_out = 512,
                max_seq_len = 1+int((args['support_duration_sec'] + args['query_duration_sec'])*args['sr']//self.downsample_factor),
                attn_layers = Encoder(
                    dim = 768,
                    depth = 12,
                    heads = 12,
                    attn_flash=True,
                    ff_swish = True,
                    ff_glu = True,
                    rotary_pos_emb = True,
                )
            )
        
        self.head = nn.Linear(512,1)
        self.label_embedding = nn.Embedding(2, 768)
        
    def freeze_audio_encoder(self):
        self.encoder.freeze()

    def unfreeze_audio_encoder(self):
        self.encoder.unfreeze()
    
    def encode_audio(self, audio, sampling_rate=None, is_support=False):
        return self.encoder(audio, sampling_rate=self.args['sr'], is_support=is_support)
        
    def downsample_labels(self, labels):
        expected_dur_output = labels.size(1) // self.downsample_factor
        labels = F.max_pool1d(labels.unsqueeze(1), self.downsample_factor, padding=0).squeeze(1) # (batch, time/scale_factor). 0=NEG 1=UNK 2=POS
        pad = expected_dur_output - labels.size(1)
        if pad>0:
            labels = F.pad(labels, (0,pad), mode='reflect')
            
        labels = labels[:,:expected_dur_output]
        return labels
    
    def encode_support(self, support_audio, support_labels, support_features=None):
        if self.args['inputs'] == "audio":
            support_audio_encoded = self.encode_audio(support_audio, is_support=True) # [batch, time, features]
            support_labels_downsampled = self.downsample_labels(support_labels) # [batch, time]

            support_labels_int = (support_labels_downsampled > 1).to(torch.int)
            support_encoded = support_audio_encoded + self.label_embedding(support_labels_int)
            
        elif self.args['inputs'] == "features":
            support_encoded = self.feat_encoder(support_features) #[batch, 1, features]
            
        elif self.args['inputs'] == "both":
            support_audio_encoded = self.encode_audio(support_audio, is_support=True) # [batch, time, features]
            support_labels_downsampled = self.downsample_labels(support_labels) # [batch, time]

            support_labels_int = (support_labels_downsampled > 1).to(torch.int)
            support_audio_encoded = support_audio_encoded + self.label_embedding(support_labels_int)
            
            support_feats_encoded = self.feat_encoder(support_features) #[batch, 1, features]
            support_encoded = torch.cat([support_audio_encoded, support_feats_encoded], dim = 1)
        
        return support_encoded
    
    def forward_with_precomputed_support(self, support_audio_encoded, query_audio, query_labels = None):
        query_encoded = self.encode_audio(query_audio)
        
        len_support = support_audio_encoded.size(1)
        inp = torch.cat([support_audio_encoded, query_encoded], dim=1)
        decoded = self.decoder(inp)
        decoded = decoded[:,len_support:,:]
        # decoded = self.decoder(query_encoded, context=support_audio_encoded)
        
        logits = self.head(decoded).squeeze(-1)
        
        if query_labels is not None:
            query_labels = self.downsample_labels(query_labels)
                
        return logits, query_labels
    
    def forward(self, support_audio, support_labels, query_audio, support_features = None, query_labels = None):
        
        support_encoded = self.encode_support(support_audio, support_labels, support_features)
        logits, query_labels = self.forward_with_precomputed_support(support_encoded, query_audio, query_labels = query_labels)
        return logits, query_labels
    
    def fill_holes(self, m, max_hole):
        stops = m[:-1] * ~m[1:]
        stops = np.nonzero(stops)[0]

        for stop in np.ravel(stops):
            look_forward = m[stop+1:stop+1+max_hole]
            if np.any(look_forward):
                next_start = np.amin(np.nonzero(look_forward)[0]) + stop + 1
                m[stop : next_start] = True

        return m
        
    def predict_selection_table_dict(self, support_audio, support_labels, query_audio, sr=None, max_hole_sec = 0.1, support_features = None):
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

class FramewiseProtonet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args['encoder'] == "aves":
            self.encoder = AvesEncoder(args)
        elif args['encoder'] == "convnet":
            self.encoder = ConvEncoder(args)
        elif args['encoder'] == "convtrans":
            self.encoder = ConvTransEncoder(args)
        self.downsample_factor = self.args['downsample_factor']
        
        # self.neg_hop_size_frames = int((self.args['neg_proto_hop_dur'] * self.args['sr'])//self.downsample_factor)
        # assert (self.args['query_duration_sec'] *self.args['sr'])/self.downsample_factor % self.neg_hop_size_frames == 0
        # assert (self.args['support_duration_sec'] *self.args['sr'])/self.downsample_factor % self.neg_hop_size_frames == 0
        
    def freeze_audio_encoder(self):
        self.encoder.freeze()

    def unfreeze_audio_encoder(self):
        self.encoder.unfreeze()
    
    def encode_audio(self, audio, sampling_rate=None, is_support=False):
        return self.encoder(audio, sampling_rate=self.args['sr'], is_support=is_support)
        
    def downsample_labels(self, labels):
        expected_dur_output = labels.size(1) // self.downsample_factor
        labels = F.max_pool1d(labels.unsqueeze(1), self.downsample_factor, padding=0).squeeze(1) # (batch, time/scale_factor). 0=NEG 1=UNK 2=POS
        pad = expected_dur_output - labels.size(1)
        if pad>0:
            labels = F.pad(labels, (0,pad), mode='reflect')
            
        labels = labels[:,:expected_dur_output]
        return labels
    
    def encode_support(self, support_audio, support_labels, support_features):
        if self.args['inputs'] != "audio":
            assert False
        
        support_audio_encoded = self.encode_audio(support_audio, is_support=True) # [batch, time, features]
        support_labels_downsampled = self.downsample_labels(support_labels) # [batch, time]
        
        support_labels_pos = (support_labels_downsampled > 1).to(torch.float).unsqueeze(-1)
        support_audio_encoded_pos = support_labels_pos * support_audio_encoded
        support_audio_encoded_pos = torch.sum(support_audio_encoded_pos, dim = 1, keepdim=True) # [batch, 1, features]
        denom = torch.sum(support_labels_pos, dim = 1, keepdim=True)
        support_audio_encoded_pos = support_audio_encoded_pos / (denom + torch.full_like(denom, 1e-10))
        
        support_labels_neg = (support_labels_downsampled < 1).to(torch.float).unsqueeze(-1)
        support_audio_encoded_neg = support_labels_neg * support_audio_encoded
        
        support_audio_encoded_neg = rearrange(support_audio_encoded_neg, 'b t c -> b c t')
        support_audio_encoded_neg = F.avg_pool1d(support_audio_encoded_neg, 25)
        support_audio_encoded_neg = rearrange(support_audio_encoded_neg, 'b c t -> b t c')
        
        support_labels_neg = rearrange(support_labels_neg, 'b t c -> b c t')
        support_labels_neg = F.avg_pool1d(support_labels_neg, 25)
        support_labels_neg = rearrange(support_labels_neg, 'b c t -> b t c')
        
        support_audio_encoded_neg = support_audio_encoded_neg / (support_labels_neg + torch.full_like(support_labels_neg, 1e-6))
        
        support_audio_encoded = torch.cat([support_audio_encoded_pos, support_audio_encoded_neg], dim = 1)
        return support_audio_encoded
    
    def forward_with_precomputed_support(self, support_audio_encoded, query_audio, query_labels = None, support_features=None):
        query_encoded = self.encode_audio(query_audio)
        
        pos_proto = support_audio_encoded[:,:1,:]
        neg_protos = support_audio_encoded[:,1:,:]
        
        query_pos = -torch.linalg.vector_norm(pos_proto - query_encoded, dim=-1)
        query_neg = -torch.linalg.vector_norm(neg_protos.unsqueeze(1) - query_encoded.unsqueeze(2), dim=-1)
        query_neg = torch.amax(query_neg, dim = -1)
        
        logits = query_pos - query_neg
        
        if query_labels is not None:
            query_labels = self.downsample_labels(query_labels)
                
        return logits, query_labels
    
    def forward(self, support_audio, support_labels, query_audio, query_labels = None, support_features=None):
        support_audio_encoded = self.encode_support(support_audio, support_labels, support_features)
        logits, query_labels = self.forward_with_precomputed_support(support_audio_encoded, query_audio, query_labels = query_labels)
        return logits, query_labels
    
    def fill_holes(self, m, max_hole):
        stops = m[:-1] * ~m[1:]
        stops = np.nonzero(stops)[0]

        for stop in np.ravel(stops):
            look_forward = m[stop+1:stop+1+max_hole]
            if np.any(look_forward):
                next_start = np.amin(np.nonzero(look_forward)[0]) + stop + 1
                m[stop : next_start] = True

        return m
        
    def predict_selection_table_dict(self, support_audio, support_labels, query_audio, sr=None, max_hole_sec = 0.1, support_features=None):
        # sr is output samplerate
        sr = int(self.args['sr'] // self.downsample_factor)
        
        logits, _ = self.forward(support_audio, support_labels, query_audio)
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