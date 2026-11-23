"""
ml/feature_extractor.py
=========================
Runs the three code views through the SHARED, frozen ModernBERT
encoder and masked-mean-pools each view's hidden states into one
768-d embedding: E_main, E_sub, E_comments.

The trainable projection layers (768->256 per view) live inside the
classifier checkpoint (ml/classifier.py), not here -- this module's
only job is producing the raw pooled encoder embeddings.
"""

from dataclasses import dataclass

import torch

from app.ml.code_splitter import CodeViews
from app.ml.modernbert_encoder import ModernBertEncoder


@dataclass
class ViewEmbeddings:
    e_main: torch.Tensor       # (hidden_size,)
    e_sub: torch.Tensor        # (hidden_size,)
    e_comments: torch.Tensor   # (hidden_size,)


def masked_mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class FeatureExtractor:
    def __init__(self, encoder: ModernBertEncoder):
        self.encoder = encoder

    def extract(self, views: CodeViews) -> ViewEmbeddings:
        texts = [views.main_code, views.sub_functions, views.comments_view]

        last_hidden_state, attention_mask = self.encoder.encode(texts)
        pooled = masked_mean_pooling(last_hidden_state, attention_mask)

        return ViewEmbeddings(e_main=pooled[0], e_sub=pooled[1], e_comments=pooled[2])
