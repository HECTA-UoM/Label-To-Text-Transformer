import torch.nn as nn
import torch
from EncDecTransformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_head: int, d_hid: int, d_k: int, d_v: int, dropout: float=0.2):

      super(EncoderLayer, self).__init__()
      self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
      self.feed_forward = PositionwiseFeedForward(d_model, d_hid, dropout)

      self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
      self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
      self.dropout1 = nn.Dropout(dropout)
      self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

      attn_output, _ = self.self_attn(x, x, x, mask)
      x = self.norm1(x + self.dropout1(attn_output))

      ff_output = self.feed_forward(x)
      x = self.norm2(x + self.dropout2(ff_output))

      return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_head: int, d_hid: int, d_k: int, d_v: int, dropout: float=0.1):

      super(DecoderLayer, self).__init__()
      self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
      self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
      self.feed_forward = PositionwiseFeedForward(d_model, d_hid, dropout)

      self.norm1 = nn.LayerNorm(d_model)
      self.norm2 = nn.LayerNorm(d_model)
      self.norm3 = nn.LayerNorm(d_model)

      self.dropout1 = nn.Dropout(dropout)
      self.dropout2 = nn.Dropout(dropout)
      self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:

      attn_output, _ = self.self_attn(x, x, x, tgt_mask)
      x = self.norm1(x + self.dropout1(attn_output))

      attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
      x = self.norm2(x + self.dropout2(attn_output))

      ff_output = self.feed_forward(x)
      x = self.norm3(x + self.dropout3(ff_output))

      return x