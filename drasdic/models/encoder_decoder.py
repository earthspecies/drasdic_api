"""
Implementation of DRASDIC model
"""

from typing import Optional

import numpy as np
import torch
import torchaudio
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder


class FeatEncoder(nn.Module):
    """
    Encodes feature vectors (e.g., metadata or auxiliary features) using an MLP.

    Parameters
    ----------
    args : dict
        Configuration dictionary. Must contain "support_features" key.

    Attributes
    ----------
    mlp : nn.Sequential
        Multilayer perceptron used to project input features to output embedding dimension.
    type_enc : nn.Parameter
        A learned embedding added to the MLP output to encode input type information.
    """

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.args = args
        self.in_dim = len(args["support_features"])
        self.hidden_dim = 100
        self.out_dim = 768
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.out_dim),
        ).to(device)

        self.type_enc = nn.Parameter(nn.init.kaiming_normal_(torch.zeros((1, 768)))).to(device)

    def freeze(self) -> None:
        """Freezes all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreezes all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1, out_dim)
        """
        return (self.mlp(x) + self.type_enc).unsqueeze(1)


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for processing raw waveform audio into fixed-dim representations.

    Parameters
    ----------
    args : dict
        Configuration dictionary with keys:
        - "n_mels"
        - "sr"
        - "hidden_size"
        - "n_blocks"
        - "downsample_factor"

    Attributes
    ----------
    spectrogram : torchaudio.transforms.MelSpectrogram
        Converts waveform into mel spectrogram.
    conv1 : nn.Conv2d
        Initial convolution layer.
    resblock1 : nn.ModuleList
        First stack of residual blocks.
    resblock2 : nn.ModuleList
        Second stack of residual blocks.
    head : nn.Linear
        Final linear projection layer.
    """

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.args = args
        self.downsample_factor = self.args["downsample_factor"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        n_mels = args["n_mels"]
        hidden_size = args["hidden_size"]
        n_blocks = args["n_blocks"]

        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.args["sr"],
            n_fft=n_mels * 8,
            hop_length=self.downsample_factor // 2,
            n_mels=n_mels,
            f_min=10,
        ).to(device)

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=7, stride=1, padding="same", bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(hidden_size).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)

        self.pool1 = nn.AdaptiveAvgPool2d((n_mels // 2, None)).to(device)

        # Residual block 1
        self.resblock1 = []
        for _ in range(n_blocks):
            self.resblock1.append(
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
                    nn.BatchNorm2d(hidden_size),
                )
            )

        self.resblock1 = nn.ModuleList(self.resblock1).to(device)

        self.pool2 = nn.AdaptiveAvgPool2d((n_mels // 4, None)).to(device)

        # Residual block 2
        self.resblock2 = []
        for _ in range(n_blocks):
            self.resblock2.append(
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding="same", bias=False),
                    nn.BatchNorm2d(hidden_size),
                )
            )

        self.resblock2 = nn.ModuleList(self.resblock2).to(device)

        self.pool3 = nn.AdaptiveAvgPool2d((n_mels // 8, None)).to(device)

        self.pool3b = nn.AvgPool1d(2).to(device)  #

        self.head = nn.Linear(hidden_size * (n_mels // 8), 768).to(device)

    def freeze(self) -> None:
        """Freezes all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreezes all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, audio: torch.Tensor, sampling_rate: int = None, is_support: bool = False) -> torch.Tensor:
        """
        Forward pass of ConvEncoder.

        Parameters
        ----------
        audio : torch.Tensor
            Input waveform tensor of shape (batch_size, samples)
        sampling_rate : int, optional
            Not used; included for API compatibility.
        is_support : bool, optional
            Not used; included for API compatibility.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, time, feature_dim)
        """
        expected_output_dur = audio.size(1) // self.downsample_factor

        x = self.spectrogram(audio)  # b c t
        x = torch.log(x + torch.full_like(x, 1e-10))

        x = x.unsqueeze(1)  # b 1 c t

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for b in self.resblock1:
            x = b(x) + x
        x = self.pool2(x)

        for b in self.resblock2:
            x = b(x) + x
        x = self.pool3(x)  # b 64 16 t

        x = torch.reshape(x, (x.size(0), -1, x.size(-1)))
        x = F.adaptive_avg_pool1d(x, expected_output_dur)

        x = rearrange(x, "b c t -> b t c")
        x = self.head(x)

        return x


class EncoderDecoder(nn.Module):
    """
    DRASDIC architecture for few-shot sound event detection.
    (Really it's just encoder-only; class name is out-of-date)

    Parameters
    ----------
    args : dict
        Configuration dictionary. Must contain keys for encoder and decoder setup.

    Attributes
    ----------
    encoder : nn.Module
        Audio encoder (e.g., ConvEncoder).
    feat_encoder : FeatEncoder
        Feature encoder for auxiliary input.
    decoder : ContinuousTransformerWrapper
        Transformer decoder for sequence modeling.
    head : nn.Linear
        Final output layer.
    label_embedding : nn.Embedding
        Learned embeddings for downsampled support labels.
    """

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.args = args

        if args["encoder"] == "convnet":
            self.encoder = ConvEncoder(args)
        else:
            raise NotImplementedError

        self.feat_encoder = FeatEncoder(args)

        self.downsample_factor = self.encoder.downsample_factor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if "encoder_size" in args:
            if args["encoder_size"] == "tiny":
                print("Using encoder size = tiny")
                self.decoder = ContinuousTransformerWrapper(
                    dim_in=768,
                    dim_out=512,
                    max_seq_len=1
                    + int(
                        (args["support_duration_sec"] + args["query_duration_sec"])
                        * args["sr"]
                        // self.downsample_factor
                    ),
                    attn_layers=Encoder(
                        dim=128,
                        depth=2,
                        heads=2,
                        attn_flash=True,
                        ff_swish=True,
                        ff_glu=True,
                        rotary_pos_emb=True,
                    ),
                ).to(device)
            if args["encoder_size"] == "small":
                print("Using encoder size = small")
                self.decoder = ContinuousTransformerWrapper(
                    dim_in=768,
                    dim_out=512,
                    max_seq_len=1
                    + int(
                        (args["support_duration_sec"] + args["query_duration_sec"])
                        * args["sr"]
                        // self.downsample_factor
                    ),
                    attn_layers=Encoder(
                        dim=512,
                        depth=4,
                        heads=8,
                        attn_flash=True,
                        ff_swish=True,
                        ff_glu=True,
                        rotary_pos_emb=True,
                    ),
                ).to(device)
            elif args["encoder_size"] == "base":
                print("Using encoder size = base")
                self.decoder = ContinuousTransformerWrapper(
                    dim_in=768,
                    dim_out=512,
                    max_seq_len=1
                    + int(
                        (args["support_duration_sec"] + args["query_duration_sec"])
                        * args["sr"]
                        // self.downsample_factor
                    ),
                    attn_layers=Encoder(
                        dim=768,
                        depth=12,
                        heads=12,
                        attn_flash=True,
                        ff_swish=True,
                        ff_glu=True,
                        rotary_pos_emb=True,
                    ),
                )

        else:
            print("Using encoder size = base")
            self.decoder = ContinuousTransformerWrapper(
                dim_in=768,
                dim_out=512,
                max_seq_len=1
                + int(
                    (args["support_duration_sec"] + args["query_duration_sec"]) * args["sr"] // self.downsample_factor
                ),
                attn_layers=Encoder(
                    dim=768,
                    depth=12,
                    heads=12,
                    attn_flash=True,
                    ff_swish=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                ),
            ).to(device)

        self.head = nn.Linear(512, 1).to(device)
        self.label_embedding = nn.Embedding(2, 768).to(device)

    def freeze_audio_encoder(self) -> None:
        """Freeze parameters of the audio encoder."""
        self.encoder.freeze()

    def unfreeze_audio_encoder(self) -> None:
        """Unfreeze parameters of the audio encoder."""
        self.encoder.unfreeze()

    def encode_audio(self, audio: torch.Tensor, sampling_rate: int = None, is_support: bool = False) -> torch.Tensor:
        """
        Apply encoder to audio waveform.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, time, feature_dim)
        """
        return self.encoder(audio, sampling_rate=self.args["sr"], is_support=is_support)

    def downsample_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Downsample binary label sequence to match encoder resolution.

        Parameters
        ----------
        labels : torch.Tensor
            Input label tensor of shape (batch, time)

        Returns
        -------
        torch.Tensor
            Downsampled labels
        """
        expected_dur_output = labels.size(1) // self.downsample_factor
        labels = F.max_pool1d(labels.unsqueeze(1), self.downsample_factor, padding=0).squeeze(
            1
        )  # (batch, time/scale_factor). 0=NEG 1=UNK 2=POS
        pad = expected_dur_output - labels.size(1)
        if pad > 0:
            labels = F.pad(labels, (0, pad), mode="reflect")

        labels = labels[:, :expected_dur_output]
        return labels

    def encode_support(
        self, support_audio: torch.Tensor, support_labels: torch.Tensor, support_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encodes the support audio and/or features with optional label information.

        Returns
        -------
        torch.Tensor
            Support embeddings
        """
        if self.args["inputs"] == "audio":
            support_audio_encoded = self.encode_audio(support_audio, is_support=True)  # [batch, time, features]
            support_labels_downsampled = self.downsample_labels(support_labels)  # [batch, time]

            support_labels_int = (support_labels_downsampled > 1).to(torch.int)
            support_encoded = support_audio_encoded + self.label_embedding(support_labels_int)

        elif self.args["inputs"] == "features":
            support_encoded = self.feat_encoder(support_features)  # [batch, 1, features]

        elif self.args["inputs"] == "both":
            support_audio_encoded = self.encode_audio(support_audio, is_support=True)  # [batch, time, features]
            support_labels_downsampled = self.downsample_labels(support_labels)  # [batch, time]

            support_labels_int = (support_labels_downsampled > 1).to(torch.int)
            support_audio_encoded = support_audio_encoded + self.label_embedding(support_labels_int)

            support_feats_encoded = self.feat_encoder(support_features)  # [batch, 1, features]
            support_encoded = torch.cat([support_audio_encoded, support_feats_encoded], dim=1)

        return support_encoded

    def forward_with_precomputed_support(
        self,
        support_audio_encoded: torch.Tensor,
        query_audio: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run decoder using already encoded support and new query audio.

        Returns
        -------
        logits : torch.Tensor
            Decoder predictions (batch, time)
        query_labels : torch.Tensor or None
            Downsampled query labels if provided
        """
        query_encoded = self.encode_audio(query_audio)

        len_support = support_audio_encoded.size(1)
        inp = torch.cat([support_audio_encoded, query_encoded], dim=1)
        decoded = self.decoder(inp)
        decoded = decoded[:, len_support:, :]
        # decoded = self.decoder(query_encoded, context=support_audio_encoded)

        logits = self.head(decoded).squeeze(-1)

        if query_labels is not None:
            query_labels = self.downsample_labels(query_labels)

        return logits, query_labels

    def forward(
        self,
        support_audio: torch.Tensor,
        support_labels: torch.Tensor,
        query_audio: torch.Tensor,
        support_features: Optional[torch.Tensor] = None,
        query_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of full encoder-decoder system.

        Returns
        -------
        tuple of (logits, query_labels)
        """
        support_encoded = self.encode_support(support_audio, support_labels, support_features)
        logits, query_labels = self.forward_with_precomputed_support(
            support_encoded, query_audio, query_labels=query_labels
        )
        return logits, query_labels

    def fill_holes(self, m: np.ndarray, max_hole: int) -> np.ndarray:
        """
        Fill short gaps (holes) in binary array.

        Parameters
        ----------
        m : np.ndarray
            Binary array of predictions.
        max_hole : int
            Maximum length of gap to fill.

        Returns
        -------
        np.ndarray
            Smoothed binary array.
        """
        stops = m[:-1] * ~m[1:]
        stops = np.nonzero(stops)[0]

        for stop in np.ravel(stops):
            look_forward = m[stop + 1 : stop + 1 + max_hole]
            if np.any(look_forward):
                next_start = np.amin(np.nonzero(look_forward)[0]) + stop + 1
                m[stop:next_start] = True

        return m

    def predict_selection_table_dict(
        self,
        support_audio: torch.Tensor,
        support_labels: torch.Tensor,
        query_audio: torch.Tensor,
        sr: Optional[int] = None,
        max_hole_sec: float = 0.1,
        support_features: Optional[torch.Tensor] = None,
    ) -> list[dict[str, list]]:
        """
        Generate selection table dictionaries from query predictions.

        Parameters
        ----------
        support_audio : torch.Tensor
            Support waveform.
        support_labels : torch.Tensor
            Labels for support set.
        query_audio : torch.Tensor
            Query waveform.
        sr : int, optional
            Sample rate for output time stamps.
        max_hole_sec : float
            Duration of silence gap to fill.
        support_features : torch.Tensor, optional
            Auxiliary features for support input.

        Returns
        -------
        list of dict
            Selection tables in Raven-compatible format.
        """
        # sr is output samplerate
        sr = int(self.args["sr"] // self.downsample_factor)

        logits, _ = self.forward(support_audio, support_labels, query_audio, support_features=support_features)
        preds_binary = logits > 0
        preds_binary = preds_binary.cpu().numpy()

        all_outs = []
        for i in range(np.shape(preds_binary)[0]):
            pb = preds_binary[i, :]
            if max_hole_sec > 0:
                pb = self.fill_holes(pb, int(sr * max_hole_sec))
            d = {"Begin Time (s)": [], "End Time (s)": [], "Annotation": []}

            starts = np.where((~pb[:-1]) & (pb[1:]))[0] + 1
            if pb[0]:
                starts = np.insert(starts, 0, 0)

            ends = np.where((pb[:-1]) & (~pb[1:]))[0] + 1
            if pb[-1]:
                ends = np.append(ends, len(pb))

            for start, end in zip(starts, ends, strict=False):
                d["Begin Time (s)"].append(start / sr)
                d["End Time (s)"].append(end / sr)
                d["Annotation"].append("POS")
            all_outs.append(d)

        return all_outs
