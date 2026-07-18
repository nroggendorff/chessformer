import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from diffuser import (
    build_noise_scheduler,
    build_vae,
    infer_latent_dim,
    tokens_to_image,
    DiffuserNet,
)
from encoding import BOARD_SQUARES, NUM_PIECE_TOKENS, SEQ_LEN, VOCAB_SIZE
from piece_attention import PieceAwareEncoder, kv_color_ids, query_type_ids

META_KV_COLOR = 3


def masked_policy_uncertainty(heatmap, legal_mask):
    masked = heatmap.masked_fill(~legal_mask, -1e4)
    log_p_from = F.log_softmax(torch.logsumexp(masked, dim=-1), dim=-1)
    log_p = (log_p_from[:, :, None] + F.log_softmax(masked, dim=-1)).masked_fill(
        ~legal_mask, -1e4
    )
    entropy = -(log_p.exp() * log_p).sum(dim=(-2, -1))
    max_entropy = torch.log(legal_mask.sum(dim=(-2, -1)).clamp(min=1).float())
    return (entropy / max_entropy.clamp(min=1e-6)).clamp(0.0, 1.0)


class ChessNet(nn.Module):
    ranks: torch.Tensor
    files: torch.Tensor
    meta_positions: torch.Tensor
    diffuser_offset_scale: torch.Tensor

    def __init__(
        self,
        d_model=128,
        nhead=4,
        enc_layers=2,
        heatmap_hidden=128,
        attn_rank=32,
        diffuser_hidden=256,
        diffuser_depth=4,
        diffuser_train_timesteps=100,
        diffuser_inference_steps=8,
        diffuser_fusion_enabled=True,
        diffuser_offset_scale=1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.diffuser_inference_steps = diffuser_inference_steps
        self.diffuser_fusion_enabled = diffuser_fusion_enabled
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.rank_emb, self.file_emb, self.meta_pos_emb = [
            nn.Embedding(n, d_model) for n in (8, 8, SEQ_LEN - BOARD_SQUARES)
        ]
        self.encoder = PieceAwareEncoder(
            d_model, nhead, 4 * d_model, enc_layers, attn_rank
        )
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(d_model, heatmap_hidden),
            nn.GELU(),
            nn.Linear(heatmap_hidden, BOARD_SQUARES),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, heatmap_hidden),
            nn.GELU(),
            nn.Linear(heatmap_hidden, 1),
        )
        self.vae = build_vae()
        self.diffuser = DiffuserNet(
            infer_latent_dim(self.vae), diffuser_hidden, diffuser_depth
        )
        self.diffuser_scheduler = build_noise_scheduler(diffuser_train_timesteps)
        (
            self.legal_from_emb,
            self.legal_to_emb,
            self.last_from_emb,
            self.last_to_emb,
        ) = [nn.Embedding(2, d_model) for _ in range(4)]
        self.register_buffer(
            "ranks", torch.arange(BOARD_SQUARES) // 8, persistent=False
        )
        self.register_buffer("files", torch.arange(BOARD_SQUARES) % 8, persistent=False)
        self.register_buffer(
            "meta_positions", torch.arange(SEQ_LEN - BOARD_SQUARES), persistent=False
        )
        self.register_buffer(
            "diffuser_offset_scale", torch.tensor(float(diffuser_offset_scale))
        )

    def set_diffuser_offset_scale(self, value):
        self.diffuser_offset_scale.fill_(float(value))

    def forward(
        self, board_input, legal_mask=None, use_diffuser=None, diffuser_steps=None
    ):
        use_diffuser = (
            self.diffuser_fusion_enabled if use_diffuser is None else use_diffuser
        )
        diffuser_steps = (
            self.diffuser_inference_steps if diffuser_steps is None else diffuser_steps
        )
        B = board_input.size(0)
        board_tokens, meta_tokens = (
            board_input[:, :BOARD_SQUARES],
            board_input[:, BOARD_SQUARES:SEQ_LEN],
        )
        encoded = self.encoder(
            torch.cat(
                [
                    self.token_emb(board_tokens)
                    + self.rank_emb(self.ranks.expand(B, BOARD_SQUARES))
                    + self.file_emb(self.files.expand(B, BOARD_SQUARES))
                    + self.legal_from_emb(
                        board_input[:, SEQ_LEN : SEQ_LEN + BOARD_SQUARES]
                    )
                    + self.legal_to_emb(
                        board_input[
                            :, SEQ_LEN + BOARD_SQUARES : SEQ_LEN + 2 * BOARD_SQUARES
                        ]
                    )
                    + self.last_from_emb(
                        board_input[
                            :,
                            SEQ_LEN + 2 * BOARD_SQUARES : SEQ_LEN + 3 * BOARD_SQUARES,
                        ]
                    )
                    + self.last_to_emb(board_input[:, SEQ_LEN + 3 * BOARD_SQUARES :]),
                    self.token_emb(meta_tokens)
                    + self.meta_pos_emb(
                        self.meta_positions.expand(B, SEQ_LEN - BOARD_SQUARES)
                    ),
                ],
                dim=1,
            ),
            torch.cat(
                [
                    query_type_ids(board_tokens),
                    torch.full_like(meta_tokens, NUM_PIECE_TOKENS),
                ],
                dim=1,
            ),
            torch.cat(
                [
                    kv_color_ids(board_tokens),
                    torch.full_like(meta_tokens, META_KV_COLOR),
                ],
                dim=1,
            ),
        )
        pooled = encoded.mean(dim=1)
        heatmap = self.heatmap_mlp(encoded[:, :BOARD_SQUARES])
        if use_diffuser:
            with torch.no_grad():
                offset = self._diffuser_offset(board_tokens, diffuser_steps)
            scale = (
                masked_policy_uncertainty(heatmap.detach(), legal_mask)
                if legal_mask is not None
                else heatmap.new_ones(B)
            )
            heatmap = (
                heatmap + (self.diffuser_offset_scale * scale).view(B, 1, 1) * offset
            )
        return heatmap, self.value_mlp(pooled).squeeze(-1).tanh()

    def _diffuser_offset(self, board_tokens, num_inference_steps):
        device = board_tokens.device
        cond_latent = self.vae.encode(tokens_to_image(board_tokens)).latent_dist.mode()
        latent = torch.randn_like(cond_latent)

        self.diffuser_scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.diffuser_scheduler.timesteps:
            pred_noise = self.diffuser(
                latent.flatten(1), t.expand(latent.size(0)), cond_latent.flatten(1)
            ).view_as(latent)
            latent = self.diffuser_scheduler.step(pred_noise, t, latent).prev_sample

        image = self.vae.decode(latent).sample.reshape(
            board_tokens.size(0), -1, BOARD_SQUARES
        )
        return image[
            torch.arange(board_tokens.size(0), device=device)[:, None], board_tokens
        ]


def load_checkpoint(path, device, config):
    model = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        heatmap_hidden=config.heatmap_hidden,
        attn_rank=config.attn_type_rank,
        diffuser_hidden=config.diffuser_hidden,
        diffuser_depth=config.diffuser_depth,
        diffuser_train_timesteps=config.diffuser_train_timesteps,
        diffuser_inference_steps=config.diffuser_inference_steps,
        diffuser_fusion_enabled=config.diffuser_fusion_enabled,
        diffuser_offset_scale=config.diffuser_offset_scale,
    ).to(device)
    model.load_state_dict(load_file(path, device="cpu"))
    model.eval()
    return model


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(model.state_dict(), path)
