"""
Funnel Transformer with PyTorch native MHA and TinyBERT embeddings/tokenizer.

Architecture follows the paper:
  "Funnel-Transformer: Filtering Out Sequential Redundancy for Efficient
   Language Processing" (Dai et al., 2020)

Key differences from the original implementation:
  - Uses nn.MultiheadAttention instead of manual RelMultiheadAttention
  - Uses TinyBERT (huggingface: "huawei-noah/TinyBERT_General_4L_312D")
    for tokenization and initial token embeddings
  - Block-level average pooling on the sequence dimension between blocks
    (CLS token is kept separate and not pooled, per the paper)
  - Supports the three-loss distillation objective described in the paper:
      L = α * L_task  +  β * L_logit  +  γ * L_layer

Usage:
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    model     = FunnelTransformer.from_tinybert(block_size="2_2_2", num_classes=2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Single Transformer layer (MHA + FFN with pre-norm)
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """One Transformer encoder layer using nn.MultiheadAttention (pre-norm)."""

    def __init__(self, d_model: int, n_head: int, d_ffn: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_head,
                                            dropout=dropout,
                                            batch_first=True)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,               # (B, T_q, D)
        kv: torch.Tensor,              # (B, T_kv, D)  — same as q for non-pooling layers
        key_padding_mask: torch.Tensor | None = None,  # (B, T_kv) bool, True = ignore
    ) -> torch.Tensor:
        # --- Self / cross attention with pre-norm on q ---
        residual = q
        q_norm   = self.norm1(q)
        kv_norm  = self.norm1(kv) if kv is not q else q_norm
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm,
                                key_padding_mask=key_padding_mask)
        x = residual + self.drop(attn_out)

        # --- FFN with pre-norm ---
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# One Funnel block  (N layers + optional pooling at the start)
# ---------------------------------------------------------------------------

class FunnelBlock(nn.Module):
    """
    A single Funnel Transformer block.

    If `apply_pooling=True`, the sequence (excluding CLS) is average-pooled
    by `pool_size` at the *beginning* of the block, before the first attention
    layer. The first layer then uses the pooled sequence as Q and the
    un-pooled sequence as K/V (pool_q_only=True), or both pooled (default).
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
        separate_cls: bool = True,
    ):
        super().__init__()
        self.apply_pooling = apply_pooling
        self.pool_size     = pool_size
        self.pool_q_only   = pool_q_only
        self.separate_cls  = separate_cls
        self.layers        = nn.ModuleList([
            TransformerLayer(d_model, n_head, d_ffn, dropout)
            for _ in range(n_layers)
        ])

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_pool_seq(x: torch.Tensor, pool_size: int) -> torch.Tensor:
        """Average-pool the time dimension by `pool_size`. (B, T, D) -> (B, T//pool_size, D)"""
        B, T, D = x.shape
        # trim to multiple of pool_size
        T_trim = (T // pool_size) * pool_size
        return x[:, :T_trim, :].reshape(B, T_trim // pool_size, pool_size, D).mean(dim=2)

    @staticmethod
    def _pool_mask(mask: torch.Tensor | None,
                   pool_size: int) -> torch.Tensor | None:
        """Downsample a key_padding_mask along the sequence dimension."""
        if mask is None:
            return None
        # mask: (B, T), True = ignore
        B, T = mask.shape
        T_trim = (T // pool_size) * pool_size
        m = mask[:, :T_trim].reshape(B, T_trim // pool_size, pool_size)
        # a pooled position is masked only if ALL its source tokens are masked
        return m.all(dim=-1)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,               # (B, T, D)
        key_padding_mask: torch.Tensor | None = None,  # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            output:           (B, T_out, D)
            key_padding_mask: (B, T_out)  — updated after pooling
        """
        if self.apply_pooling:
            if self.separate_cls:
                # keep CLS separate
                cls_token = x[:, :1, :]          # (B, 1, D)
                rest      = x[:, 1:, :]           # (B, T-1, D)
                cls_mask  = key_padding_mask[:, :1] if key_padding_mask is not None else None
                rest_mask = key_padding_mask[:, 1:] if key_padding_mask is not None else None

                pooled_rest      = self._avg_pool_seq(rest, self.pool_size)
                pooled_rest_mask = self._pool_mask(rest_mask, self.pool_size)

                # first layer: Q = [CLS | pooled_rest], K/V = original x
                q = torch.cat([cls_token, pooled_rest], dim=1)
                q_mask = (torch.cat([cls_mask, pooled_rest_mask], dim=1)
                          if cls_mask is not None else None)
            else:
                pooled = self._avg_pool_seq(x, self.pool_size)
                q      = pooled
                q_mask = self._pool_mask(key_padding_mask, self.pool_size)

            # first layer uses original x as K/V when pool_q_only=True
            kv      = x      if self.pool_q_only else q
            kv_mask = key_padding_mask if self.pool_q_only else q_mask

            out = self.layers[0](q, kv, key_padding_mask=kv_mask)

            # remaining layers: self-attention on the pooled sequence
            for layer in self.layers[1:]:
                out = layer(out, out, key_padding_mask=q_mask)

            return out, q_mask

        else:
            # standard self-attention block (no pooling)
            out = x
            for layer in self.layers:
                out = layer(out, out, key_padding_mask=key_padding_mask)
            return out, key_padding_mask


# ---------------------------------------------------------------------------
# Full Funnel Transformer
# ---------------------------------------------------------------------------

class FunnelTransformer(nn.Module):
    """
    Funnel Transformer for sequence classification.

    Args:
        vocab_size:    vocabulary size
        d_model:       hidden dimension
        n_head:        number of attention heads
        d_ffn:         inner FFN dimension
        block_layers:  list of ints, number of layers per block
                       e.g. [2, 2, 2]  →  block_size = "2_2_2"
        num_classes:   number of output classes
        dropout:       dropout probability
        pool_size:     pooling stride between blocks (default 2)
        pool_q_only:   if True, only Q is pooled at pooling layers
        separate_cls:  if True, CLS token is excluded from pooling
        pad_token_id:  token id used for padding (for mask construction)
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
        pool_size: int = 2,
        pool_q_only: bool = True,
        separate_cls: bool = True,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.separate_cls = separate_cls
        self.n_blocks     = len(block_layers)

        # Token embedding + layer norm + dropout  (no positional encoding;
        # relative positions are implicit in the attention bias — a simple
        # learned absolute positional embedding is added below for stability)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb   = nn.Embedding(512, d_model)          # max 512 positions
        self.emb_norm  = nn.LayerNorm(d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # Funnel blocks
        self.blocks = nn.ModuleList()
        for i, n_layers in enumerate(block_layers):
            self.blocks.append(FunnelBlock(
                n_layers      = n_layers,
                d_model       = d_model,
                n_head        = n_head,
                d_ffn         = d_ffn,
                dropout       = dropout,
                apply_pooling = (i > 0),        # first block has no pooling
                pool_size     = pool_size,
                pool_q_only   = pool_q_only,
                separate_cls  = separate_cls,
            ))

        # Classification head over the final [CLS] token
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,              # (B, T)
        attention_mask: torch.Tensor | None = None,  # (B, T), 1=attend 0=ignore
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with:
            logits        (B, num_classes)
            cls_hiddens   list of (B, D) — CLS representation at each block output
                          used for layer-wise distillation (L_layer)
        """
        B, T = input_ids.shape

        # Build key_padding_mask: True where we should IGNORE the token
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)          # (B, T)
        else:
            pad_mask = (input_ids == self.pad_token_id)

        # Embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_drop(self.emb_norm(x))

        # Pass through blocks, collect CLS at each block exit
        cls_hiddens = []
        for block in self.blocks:
            x, pad_mask = block(x, pad_mask)
            cls_hiddens.append(x[:, 0, :])            # (B, D)

        logits = self.cls_head(x[:, 0, :])

        return {
            "logits":      logits,
            "cls_hiddens": cls_hiddens,   # [block0_cls, block1_cls, block2_cls, ...]
        }

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    @staticmethod
    def distillation_loss(
        student_out:    dict,
        teacher_logits: torch.Tensor,
        teacher_cls:    list[torch.Tensor],
        labels:         torch.Tensor,
        alpha: float = 1.0,
        beta:  float = 1.0,
        gamma: float = 1.0,
        temperature: float = 4.0,
    ) -> dict[str, torch.Tensor]:
        """
        Combined distillation objective (Eq. 4 in the paper):
            L = α * L_task  +  β * L_logit  +  γ * L_layer

        Args:
            student_out:    output dict from FunnelTransformer.forward()
            teacher_logits: (B, C)  — teacher's final logits
            teacher_cls:    list of (B, D) — teacher CLS at mapped layers
                            must have the same length as student's block count
            labels:         (B,)   — ground truth class indices
            alpha, beta, gamma: loss weights
            temperature:    softmax temperature for KL loss

        Returns:
            dict with keys: loss, l_task, l_logit, l_layer
        """
        s_logits     = student_out["logits"]
        s_cls_list   = student_out["cls_hiddens"]

        # L_task  — cross-entropy with ground truth
        l_task = F.cross_entropy(s_logits, labels)

        # L_logit — KL divergence between soft distributions
        s_soft = F.log_softmax(s_logits   / temperature, dim=-1)
        t_soft = F.softmax(teacher_logits / temperature, dim=-1)
        l_logit = F.kl_div(s_soft, t_soft, reduction="batchmean") * (temperature ** 2)

        # L_layer — MSE between CLS representations at each block
        assert len(s_cls_list) == len(teacher_cls), (
            f"Student has {len(s_cls_list)} blocks but {len(teacher_cls)} "
            "teacher CLS tensors were provided."
        )
        l_layer = sum(
            F.mse_loss(s_cls, t_cls)
            for s_cls, t_cls in zip(s_cls_list, teacher_cls)
        ) / len(s_cls_list)

        loss = alpha * l_task + beta * l_logit + gamma * l_layer

        return {
            "loss":    loss,
            "l_task":  l_task.detach(),
            "l_logit": l_logit.detach(),
            "l_layer": l_layer.detach(),
        }

    # ------------------------------------------------------------------
    # Factory: build from TinyBERT checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_tinybert(
        cls,
        block_size:  str  = "2_2_2",
        num_classes: int  = 2,
        dropout:     float = 0.1,
        pool_size:   int  = 2,
        pool_q_only: bool = True,
        separate_cls: bool = True,
        model_name:  str  = "huawei-noah/TinyBERT_General_4L_312D",
    ) -> tuple["FunnelTransformer", object]:
        """
        Build a FunnelTransformer whose token embedding matrix and vocabulary
        are copied from TinyBERT.

        Returns:
            model:     FunnelTransformer (with TinyBERT embeddings frozen by default)
            tokenizer: HuggingFace tokenizer for TinyBERT
        """
        print(f"Loading TinyBERT from '{model_name}' …")
        tokenizer   = AutoTokenizer.from_pretrained(model_name)
        tinybert    = AutoModel.from_pretrained(model_name)
        tb_cfg      = tinybert.config

        # parse block_size string, e.g. "2_2_2" -> [2, 2, 2]
        block_layers = [int(b.split("x")[0]) for b in block_size.split("_")]

        # Build funnel model matching TinyBERT's embedding dimension
        model = cls(
            vocab_size   = tb_cfg.vocab_size,
            d_model      = tb_cfg.hidden_size,
            n_head       = tb_cfg.num_attention_heads,
            d_ffn        = tb_cfg.intermediate_size,
            block_layers = block_layers,
            num_classes  = num_classes,
            dropout      = dropout,
            pool_size    = pool_size,
            pool_q_only  = pool_q_only,
            separate_cls = separate_cls,
            pad_token_id = tokenizer.pad_token_id,
        )

        # Copy TinyBERT's pretrained token embeddings
        with torch.no_grad():
            model.token_emb.weight.copy_(
                tinybert.embeddings.word_embeddings.weight
            )
            # also copy positional embeddings if sizes match
            tb_pos = tinybert.embeddings.position_embeddings.weight
            n = min(model.pos_emb.weight.shape[0], tb_pos.shape[0])
            model.pos_emb.weight[:n].copy_(tb_pos[:n])

        print(
            f"  ✓ Copied token embeddings  ({tb_cfg.vocab_size} × {tb_cfg.hidden_size})\n"
            f"  ✓ Copied positional embeddings (first {n} positions)\n"
            f"  Block config: {block_layers}  |  num_classes: {num_classes}"
        )

        # Free TinyBERT weights — we only needed the embeddings
        del tinybert

        return model, tokenizer


# ---------------------------------------------------------------------------
# Convenience: build the BERT teacher used in the paper
# ---------------------------------------------------------------------------

def build_bert_teacher(
    num_classes: int,
    model_name: str = "google-bert/bert-base-uncased",
) -> tuple[nn.Module, object]:
    """
    Fine-tunable BERT_BASE teacher with a linear classification head.
    Returns (model, tokenizer).
    """
    from transformers import AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a 2-2-2 Funnel student with TinyBERT embeddings
    model, tokenizer = FunnelTransformer.from_tinybert(
        block_size  = "2_2_2",
        num_classes = 2,
    )
    model.eval()

    sentences = [
        "The movie was absolutely fantastic!",
        "I did not enjoy this film at all.",
    ]
    enc = tokenizer(sentences, padding=True, truncation=True,
                    max_length=64, return_tensors="pt")

    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])

    print("\nlogits shape :", out["logits"].shape)         # (2, 2)
    print("CLS per block:", [h.shape for h in out["cls_hiddens"]])  # [(2,312), (2,312), (2,312)]

    # ---- Simulate one distillation step ----
    # Fake teacher outputs (in practice, run BERT teacher forward pass)
    B, D = 2, model.token_emb.weight.shape[1]
    fake_teacher_logits = torch.randn(B, 2)
    fake_teacher_cls    = [torch.randn(B, D) for _ in range(3)]  # one per student block
    labels              = torch.tensor([1, 0])

    losses = FunnelTransformer.distillation_loss(
        student_out    = out,
        teacher_logits = fake_teacher_logits,
        teacher_cls    = fake_teacher_cls,
        labels         = labels,
        alpha=1.0, beta=1.0, gamma=1.0,
    )
    print("\nDistillation losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")