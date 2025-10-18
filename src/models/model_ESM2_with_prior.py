#!/usr/bin/env python
from torch import nn
from einops import rearrange
import torch
from torch.nn import Module
from torch import Tensor
import esm as fair_esm


class PairwiseProjector(Module):
    def __init__(
            self,
            d_model: int,
            out_ch: int,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(3*d_model, out_ch, kernel_size=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(
            self, 
            feats: Tensor
    ) -> Tensor:
        """
        Projects per-residue features to pairwise features.

        Args:
            feats: (L, d) tensor of per-residue features
        Returns:
            Tensor: (1, out_ch, L, L) pairwise features
        """

        feats_i = feats.unsqueeze(1)                # (L,1,d)
        feats_j = feats.unsqueeze(0)                # (1,L,d)
        add = feats_i + feats_j                     # (L,L,d)
        diff = (feats_i - feats_j).abs()            # (L,L,d)
        had = feats_i * feats_j                     # (L,L,d)
        out = torch.cat([add, diff, had], dim=-1)   # (L,L,3d)
        out = rearrange(out, "i j c -> 1 c i j")    # (B=1, C, L, L)
        return self.proj(out)                       # (1, out_ch, L, L)


class ResBlock2D(Module):
    def __init__(
            self,
            ch: int,
            dilation: int=1,
            pdrop: float=0.1
    ):
        super().__init__()

        pad = dilation
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Conv2d(ch, ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.GroupNorm(8, ch),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class ContactHead2D(Module):
    def __init__(
            self,
            ch: int,
            depth: int=12,
            pdrop: float=0.1
    ):
        super().__init__()
        blocks = []
        dil_cycle = [1, 2, 4, 8]
        for i in range(depth):
            d = dil_cycle[i % len(dil_cycle)]
            blocks.append(ResBlock2D(ch, dilation=d, pdrop=pdrop))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        """
        Args:
            x: (B=1, ch, L, L) tensor of pairwise features
        Returns:
            Tensor: (L, L) contact logits
        """

        x = self.blocks(x)

        return self.out(x).squeeze(1)  # (B=1, L, L) logits


class ESM2ContactWithTemplatePrior(Module):
    def __init__(
            self,
            esm_model: str="esm2_t33_650M_UR50D",
            freeze_esm: bool=True,
            cnn_channels: int=128,
            cnn_depth: int=12,
            use_esm_contact_head: bool=True,
            num_dist_bins: int=5,
            dropout: float=0.1
    ):
        super().__init__()

        assert fair_esm is not None
        self.esm_model_name = esm_model
        self.use_esm_contact_head = use_esm_contact_head

        self.esm_model, self.alphabet = getattr(fair_esm.pretrained, esm_model)()
        self.batch_converter = self.alphabet.get_batch_converter()
        # pick the last layer of the model (e.g., 30 for t30_150M, 33 for t33_650M, etc.)
        self.repr_layer = getattr(self.esm_model, "num_layers", 33)
        d_model = self.esm_model.embed_dim

        if freeze_esm:
            for p in self.esm_model.parameters():
                p.requires_grad = False

        self.pairwise_projector = PairwiseProjector(d_model, cnn_channels)
        extra_ch = 1 + num_dist_bins  # pri_contact + pri_bins
        if use_esm_contact_head:
            extra_ch += 1
        self.fuse = nn.Conv2d(cnn_channels + extra_ch, cnn_channels, kernel_size=1)
        self.head = ContactHead2D(cnn_channels, depth=cnn_depth, pdrop=dropout)

    @torch.no_grad()
    def esm_encode(
            self,
            seq: str
    ) -> Tensor:
        """
        Encodes a sequence with the Fair-ESM model to get per-residue representations.

        Args:
            seq: amino acid sequence string
        Returns:
            Tensor: (L, d) per-residue representations
        """

        batch = [("protein", seq)]
        _, _, toks = self.batch_converter(batch)  # Get token tensor from sequence
        device = next(self.parameters()).device
        toks = toks.to(device)

        # Fair-ESM forward with the correct representation layer
        out = self.esm_model(
            toks,
            repr_layers=[self.repr_layer],
            return_contacts=False  # contacts separately below
        )
        rep = out["representations"][self.repr_layer][0, 1:1 + len(seq), :]  # (L, d)

        return rep

    def forward(
            self,
            seq: str,
            pri_contact: Tensor,
            pri_bins: Tensor,
    ) -> Tensor:
        """
        Predicts contact logits from sequence and prior contact/distance information.

        Args:
            seq: amino acid sequence string
            pri_contact: (L, L) tensor of prior contact logits
            pri_bins: (L, L, B) tensor of prior distance bin one-hot encodings
        Returns:
            Tensor: (L, L) contact logits
        """

        H = self.esm_encode(seq)  # (L, d), (L, L) or None
        L = H.size(0)
        pair = self.pairwise_projector(H)  # (1, C, L, L)

        feats = [
            pair,
            pri_contact.unsqueeze(0).unsqueeze(0),  # (1, 1, L, L)
            pri_bins.permute(2, 0, 1).unsqueeze(0),  # (1, B, L, L)
        ]

        # Always provide the contact-head channel if the model was built to expect it
        if self.use_esm_contact_head:
            contact_esm = pri_contact.new_zeros((L, L))  # pad zeros to keep channels consistent
            feats.append(contact_esm.unsqueeze(0).unsqueeze(0))  # (1, 1, L, L)

        out = torch.cat(feats, dim=1)  # channels match fuse.in_channels now
        out = self.fuse(out)
        logits = self.head(out)[0]  # (L, L)
        logits = 0.5 * (logits + logits.T)  # symmetrize
        logits = logits - torch.diag(torch.diag(logits))  # zero out diagonal
        return logits
