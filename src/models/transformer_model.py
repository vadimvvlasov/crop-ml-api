import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import LayerNorm, Linear, ReLU
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

__all__ = ["TransformerModel"]


class SinCosEncoding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for SinCosEncoding.")
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        omega = torch.arange(half_dim, dtype=torch.float32)
        omega = 1.0 / (10000.0 ** (omega / half_dim))
        self.register_buffer("omega", omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.omega.device).float().unsqueeze(-1)
        out = x * self.omega
        return torch.cat((out.sin(), out.cos()), dim=-1)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=4,
        num_classes=25,
        d_model=182,
        n_head=2,
        n_layers=5,
        d_inner=128,
        activation="relu",
        dropout=0.017998950510888446,
        use_temporal: bool = True,
        use_location: bool = True,
    ):
        super(TransformerModel, self).__init__()

        enc_tag = (
            "_TL" if use_temporal and use_location
            else "_T" if use_temporal
            else "_L" if use_location
            else ""
        )
        self.modelname = (
            f"TransformerEncoder{enc_tag}_input-dim={input_dim}_num-classes={num_classes}_"
            f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_"
            f"dropout={dropout}"
        )

        encoder_layer = TransformerEncoderLayer(
            d_model, n_head, d_inner, dropout, activation
        )
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.transformerencoder = TransformerEncoder(
            encoder_layer, n_layers, encoder_norm
        )
        self.flatten = Flatten()
        self.outlinear = Linear(d_model, num_classes)

        self.use_temporal = use_temporal
        self.use_location = use_location
        if self.use_temporal:
            self.temporal_enc = SinCosEncoding(d_model)
            self.temporal_scale = nn.Parameter(torch.full((1,), 0.1))
        if self.use_location:
            self.location_enc = SinCosEncoding(d_model)
            self.location_scale = nn.Parameter(torch.full((1,), 0.1))

    def forward(
        self,
        x: torch.Tensor,
        week_of_year: torch.Tensor | None = None,
        location: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.inlinear(x)
        x = self.relu(x)

        if self.use_temporal and week_of_year is not None:
            if week_of_year.dim() == 1:
                week_expanded = week_of_year.unsqueeze(0).expand(x.size(0), -1)
            else:
                week_expanded = week_of_year
            temporal_emb = self.temporal_enc(week_expanded)
            x = x + self.temporal_scale * temporal_emb

        if self.use_location and location is not None:
            if location.dim() != 2 or location.size(1) != 2:
                raise ValueError("location must have shape (N, 2).")
            lat = location[:, 0]
            lon = location[:, 1]
            lat_enc = self.location_enc(lat)
            lon_enc = self.location_enc(lon)
            loc_emb = lat_enc + lon_enc
            x = x + self.location_scale * loc_emb.unsqueeze(1)

        x = x.transpose(0, 1)
        x = self.transformerencoder(x)
        x = x.transpose(0, 1)
        x = x.max(1)[0]
        x = self.relu(x)
        logits = self.outlinear(x)

        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)
