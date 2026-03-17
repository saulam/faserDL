"""
Context-aware multi-particle PID model for PILArNet event classification.
"""

import torch
import torch.nn as nn

from .adapted_model import pilarnet_encoder_tiny, pilarnet_encoder_base
from .pilarnet_dataset import PILARNET_PARTICLE_META_DIM


class PILArNetMultiParticleClassifier(nn.Module):
    """Classify each particle using both its local crop and event context."""

    def __init__(
        self,
        particle_encoder,
        meta_dim=PILARNET_PARTICLE_META_DIM,
        meta_dropout=0.1,
        context_layers=2,
        context_heads=12,
        context_dropout=0.1,
        num_pid_classes=5,
        head_dropout_cls=0.0,
    ):
        super().__init__()
        self.encoder = particle_encoder
        self.embed_dim = particle_encoder.embed_dim
        self.spatial_shape = particle_encoder.spatial_shape

        # The wrapped encoder's particle head is unused in the contextual model.
        for p in self.encoder.head_pid.parameters():
            p.requires_grad = False

        self.meta_proj = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.meta_dropout = nn.Dropout(meta_dropout)
        self.input_norm = nn.LayerNorm(self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=context_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=context_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=context_layers,
        )
        self.context_norm = nn.LayerNorm(self.embed_dim)
        self.head_pid = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(head_dropout_cls),
            nn.Linear(self.embed_dim, num_pid_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.meta_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for module in self.head_pid.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=2e-5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def no_weight_decay(self):
        return {f"encoder.{name}" for name in self.encoder.no_weight_decay()}

    def _pack_events(self, particle_embeddings, event_offsets):
        lengths = (event_offsets[1:] - event_offsets[:-1]).tolist()
        batch_size = len(lengths)
        max_particles = max(lengths)
        dim = particle_embeddings.size(-1)

        packed = particle_embeddings.new_zeros((batch_size, max_particles, dim))
        key_padding_mask = torch.ones(
            (batch_size, max_particles),
            dtype=torch.bool,
            device=particle_embeddings.device,
        )

        for event_i, length in enumerate(lengths):
            start = int(event_offsets[event_i].item())
            end = int(event_offsets[event_i + 1].item())
            packed[event_i, :length] = particle_embeddings[start:end]
            key_padding_mask[event_i, :length] = False

        return packed, key_padding_mask, lengths

    def _unpack_events(self, packed_logits, lengths):
        logits = []
        for event_i, length in enumerate(lengths):
            logits.append(packed_logits[event_i, :length])
        return torch.cat(logits, dim=0)

    def forward(self, x_sp, particle_meta, event_offsets):
        particle_embeddings = self.encoder.encode_pooled(x_sp)
        particle_embeddings = particle_embeddings + self.meta_dropout(
            self.meta_proj(particle_meta)
        )
        particle_embeddings = self.input_norm(particle_embeddings)

        packed, key_padding_mask, lengths = self._pack_events(
            particle_embeddings,
            event_offsets,
        )
        packed = self.context_blocks(
            packed,
            src_key_padding_mask=key_padding_mask,
        )
        packed = self.context_norm(packed)
        packed_logits = self.head_pid(packed)
        logits = self._unpack_events(packed_logits, lengths)
        return {"out_pid": logits}


def pilarnet_multiparticle_tiny(**kwargs):
    encoder_kwargs = dict(kwargs)
    model_kwargs = {
        "meta_dim": encoder_kwargs.pop("meta_dim", PILARNET_PARTICLE_META_DIM),
        "meta_dropout": encoder_kwargs.pop("meta_dropout", 0.1),
        "context_layers": encoder_kwargs.pop("context_layers", 2),
        "context_heads": encoder_kwargs.pop("context_heads", 12),
        "context_dropout": encoder_kwargs.pop("context_dropout", 0.1),
        "num_pid_classes": encoder_kwargs.pop("num_pid_classes", 5),
        "head_dropout_cls": encoder_kwargs.get("head_dropout_cls", 0.0),
    }
    encoder = pilarnet_encoder_tiny(**encoder_kwargs)
    return PILArNetMultiParticleClassifier(encoder, **model_kwargs)


def pilarnet_multiparticle_base(**kwargs):
    encoder_kwargs = dict(kwargs)
    model_kwargs = {
        "meta_dim": encoder_kwargs.pop("meta_dim", PILARNET_PARTICLE_META_DIM),
        "meta_dropout": encoder_kwargs.pop("meta_dropout", 0.1),
        "context_layers": encoder_kwargs.pop("context_layers", 2),
        "context_heads": encoder_kwargs.pop("context_heads", 12),
        "context_dropout": encoder_kwargs.pop("context_dropout", 0.1),
        "num_pid_classes": encoder_kwargs.pop("num_pid_classes", 5),
        "head_dropout_cls": encoder_kwargs.get("head_dropout_cls", 0.0),
    }
    encoder = pilarnet_encoder_base(**encoder_kwargs)
    return PILArNetMultiParticleClassifier(encoder, **model_kwargs)
