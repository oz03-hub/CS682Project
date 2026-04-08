from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# The model used both to seed the student embeddings and as the teacher
TEACHER_MODEL_NAME = "google/bert_uncased_L-8_H-512_A-8"

# Architecture is always 3 blocks
N_BLOCKS = 3


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, T_q, D)
        kv: torch.Tensor,  # (B, T_kv, D)
        key_padding_mask: torch.Tensor | None = None,  # (B, T_kv) True=ignore
    ) -> torch.Tensor:
        residual = q
        q_norm = self.norm1(q)
        kv_norm = self.norm1(kv) if kv is not q else q_norm
        attn_out, _ = self.attn(
            q_norm, kv_norm, kv_norm, key_padding_mask=key_padding_mask
        )
        x = residual + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class FunnelBlock(nn.Module):
    """
    A single Funnel Transformer block.

    When apply_pooling=True the sequence (excluding CLS) is average-pooled
    by pool_size at the start. The first layer uses original K/V and pooled Q
    (pool_q_only=True); subsequent layers are pure self-attention on the
    shortened sequence.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float = 0.1,
        apply_pooling: bool = False,
        pool_size: int = 2,
        pool_q_only: bool = True,
    ):
        super().__init__()
        self.apply_pooling = apply_pooling
        self.pool_size = pool_size
        self.pool_q_only = pool_q_only
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_head, d_ffn, dropout) for _ in range(n_layers)]
        )

    def _avg_pool_seq(self, x: torch.Tensor, pool_size: int) -> torch.Tensor:
        """(B, T, D) -> (B, T//pool_size, D)  via average pooling."""
        B, T, D = x.shape
        T_trim = (T // pool_size) * pool_size
        return x[:, :T_trim].reshape(B, T_trim // pool_size, pool_size, D).mean(2)

    def _pool_mask(
        self, mask: torch.Tensor | None, pool_size: int
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        B, T = mask.shape
        T_trim = (T // pool_size) * pool_size
        return mask[:, :T_trim].reshape(B, T_trim // pool_size, pool_size).all(-1)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        key_padding_mask: torch.Tensor | None = None,  # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.apply_pooling:
            # Keep CLS separate; pool the rest
            cls_token = x[:, :1, :]  # (B, 1, D)
            rest = x[:, 1:, :]  # (B, T-1, D)

            cls_mask = key_padding_mask[:, :1] if key_padding_mask is not None else None
            rest_mask = (
                key_padding_mask[:, 1:] if key_padding_mask is not None else None
            )

            pooled_rest = self._avg_pool_seq(rest, self.pool_size)
            pooled_rest_mask = self._pool_mask(rest_mask, self.pool_size)

            # Q = [CLS | pooled_rest]
            q = torch.cat([cls_token, pooled_rest], dim=1)
            q_mask = (
                torch.cat([cls_mask, pooled_rest_mask], dim=1)
                if cls_mask is not None
                else None
            )

            # First layer: K/V from original (un-pooled) sequence when pool_q_only
            kv = x if self.pool_q_only else q
            kv_mask = key_padding_mask if self.pool_q_only else q_mask

            out = self.layers[0](q, kv, key_padding_mask=kv_mask)
            # kv is now pooled size too
            # first layer focuses on unpooled kv to capture context from pooling, minimizes context loss
            for layer in self.layers[1:]:
                out = layer(out, out, key_padding_mask=q_mask)
            return out, q_mask

        else:
            out = x
            for layer in self.layers:
                out = layer(out, out, key_padding_mask=key_padding_mask)
            return out, key_padding_mask


class FunnelTransformer(nn.Module):
    """
    Encoder-only Funnel Transformer with a fixed 3-block architecture.
    CLS token is kept separate from pooling throughout.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        block_layers: list[int],
        num_classes: int,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        if len(block_layers) != N_BLOCKS:
            raise ValueError()

        self.pad_token_id = pad_token_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(512, d_model) # max len is not more than 512
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Block 0: no pooling; Blocks 1 & 2: halve the sequence
        self.blocks = nn.ModuleList(
            [
                FunnelBlock(
                    n_layers=n_layers,
                    d_model=d_model,
                    n_head=n_head,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    apply_pooling=(i > 0),
                    pool_size=2,
                    pool_q_only=True,
                )
                for i, n_layers in enumerate(block_layers)
            ]
        )

        # Classification head over the final [CLS]
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T)
        attention_mask: torch.Tensor | None = None,  # (B, T) 1=attend 0=pad
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            logits      (B, num_classes)
            cls_hiddens list of 3 tensors (B, D) — CLS at end of each block;
                        used for layer-wise distillation
        """
        B, T = input_ids.shape

        if attention_mask is not None:
            pad_mask = attention_mask == 0
        else:
            pad_mask = input_ids == self.pad_token_id

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_drop(self.emb_norm(x))

        cls_hiddens: list[torch.Tensor] = []
        for block in self.blocks:
            x, pad_mask = block(x, pad_mask)
            cls_hiddens.append(x[:, 0, :])  # (B, D) — CLS at block exit

        logits = self.cls_head(cls_hiddens[-1])

        return {"logits": logits, "cls_hiddens": cls_hiddens}

    # use this method mostly
    @classmethod
    def from_bert(
        cls,
        block_layers: list[int] = [2, 2, 2],
        num_classes: int = 2,
        dropout: float = 0.1,
        model_name: str = TEACHER_MODEL_NAME,
    ) -> tuple["FunnelTransformer", object]:
        if len(block_layers) != N_BLOCKS:
            raise ValueError()

        print(f"Loading '{model_name}' for embedding initialisation ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert = AutoModel.from_pretrained(model_name)
        cfg = bert.config

        model = cls(
            vocab_size=cfg.vocab_size,
            d_model=cfg.hidden_size,
            n_head=cfg.num_attention_heads,
            d_ffn=cfg.intermediate_size,
            block_layers=block_layers,
            num_classes=num_classes,
            dropout=dropout,
            pad_token_id=tokenizer.pad_token_id,
        )

        # copy pre-trained initial embedding weights
        with torch.no_grad():
            model.token_emb.weight.copy_(bert.embeddings.word_embeddings.weight)
            src_pos = bert.embeddings.position_embeddings.weight
            n_copy = min(model.pos_emb.weight.shape[0], src_pos.shape[0])
            model.pos_emb.weight[:n_copy].copy_(src_pos[:n_copy])

        print(
            f"  Copied token embeddings  ({cfg.vocab_size} x {cfg.hidden_size})\n"
            f"  Copied positional embeddings (first {n_copy} positions)\n"
            f"  Block config: {block_layers}  |  num_classes: {num_classes}"
        )

        del bert
        return model, tokenizer

if __name__ == "__main__":
    student, tokenizer = FunnelTransformer.from_bert()
    student.eval()
    sentences = [
        "The movie was absolutely fantastic!",
        "I did not enjoy this film at all.",
    ]
    enc = tokenizer(
        sentences, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )

    with torch.no_grad():
        out = student(**enc)

    print("\nlogits shape  :", out["logits"].shape)  # (2, 2)
    print("cls_hiddens   :", [h.shape for h in out["cls_hiddens"]])
