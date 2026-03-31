from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BERTTeacher(nn.Module):
    MODEL_NAME = "google/bert_uncased_L-8_H-512_A-8"

    def __init__(
        self,
        num_classes: int,
        mapped_layer_indices: list[int],
        dropout: float = 0.1,
        model_name: str = MODEL_NAME,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.n_layers = self.bert.config.num_hidden_layers
        self.mapped_layer_indices = mapped_layer_indices

        d_model = self.bert.config.hidden_size

        # Single linear classification head over final [CLS]
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with:
            logits       (B, num_classes)
            cls_hiddens  list[Tensor(B, D)]
        """

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=kwargs.get("token_type_ids"),
        )

        # hidden_states: tuple of (B, T, D), length = n_layers + 1
        # index 0 = after embedding, index i = after transformer layer i
        hidden_states = bert_out.hidden_states

        # Final [CLS] -> classification logits
        final_cls = self.drop(bert_out.last_hidden_state[:, 0, :])  # (B, D)
        logits = self.cls_head(final_cls)  # (B, num_classes)

        # Intermediate [CLS] at mapped layers for L_layer distillation
        cls_hiddens: list[torch.Tensor] = [
            hidden_states[idx][:, 0, :]  # (B, D)
            for idx in self.mapped_layer_indices
        ]

        out = {
            "logits": logits,
            "cls_hiddens": cls_hiddens,
        }

        return out

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def layer_mapping_info(self) -> str:
        lines = [
            f"Teacher layers: {self.n_layers}  |  Student blocks: {len(self.mapped_layer_indices)}"
            "Student block  ->  Teacher layer (hidden_states index)",
        ]

        for block_i, layer_idx in enumerate(self.mapped_layer_indices):
            lines.append(f"    Block {block_i + 1}  ->  Layer {layer_idx}")
        return "\n".join(lines)

    @classmethod
    def from_pretrained(
        cls,
        num_classes: int,
        mapped_layer_indices: list[int],
        dropout: float = 0.1,
        model_name: str = MODEL_NAME,
    ) -> tuple["BERTTeacher", object]:
        """
        Returns:
            teacher:   TinyBERTTeacher
            tokenizer: HuggingFace tokenizer (shared with FunnelTransformer student)
        """

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        teacher = cls(
            num_classes=num_classes,
            mapped_layer_indices=mapped_layer_indices,
            dropout=dropout,
            model_name=model_name,
        )
        print(teacher.layer_mapping_info())
        return teacher, tokenizer


# Smoke test
if __name__ == "__main__":
    teacher, tokenizer = BERTTeacher.from_pretrained(
        num_classes=2, mapped_layer_indices=[2, 4]
    )
    teacher.eval()

    sentences = [
        "The movie was absolutely fantastic!",
        "I did not enjoy this film at all.",
    ]
    enc = tokenizer(
        sentences, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )

    with torch.no_grad():
        out = teacher(**enc)

    print("\nlogits shape  :", out["logits"].shape)  # (2, 2)
    print("cls_hiddens   :", [h.shape for h in out["cls_hiddens"]])
