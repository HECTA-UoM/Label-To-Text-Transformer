import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from EncDecTransformer.Modules import ScaledDotProductAttention


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.2):
      super(PositionwiseFeedForward, self).__init__()
      self.w_1 = nn.Linear(d_in, d_hid) # position-wise
      self.w_2 = nn.Linear(d_hid, d_in) # position-wise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.w_2(F.relu(self.w_1(x)))


class MultiHeadAttention(nn.Module):

  def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int, dropout: float=0.2):
    super(MultiHeadAttention, self).__init__()

    self.n_heads = n_heads
    self.d_k = d_k
    self.d_v = d_v

    assert d_model % self.n_heads == 0

    self.w_qs = nn.Linear(d_model, n_heads * d_k)
    self.w_ks = nn.Linear(d_model, n_heads * d_k)
    self.w_vs = nn.Linear(d_model, n_heads * d_v)
    self.fc = nn.Linear(n_heads * d_v, d_model)

    self.attention = ScaledDotProductAttention(temperature=np.sqrt(d_k))

  def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:

    d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
    sz_batch, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

    # Pass through the pre-attention projection: b x lq x (n*dv)
    # Separate different heads: b x lq x n x dv
    q = self.w_qs(q).view(sz_batch, len_q, n_heads, d_k)
    k = self.w_ks(k).view(sz_batch, len_k, n_heads, d_k)
    v = self.w_vs(v).view(sz_batch, len_v, n_heads, d_v)

    # Transpose for attention dot product: b x n x lq x dv
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    if mask is not None:
        mask = mask.unsqueeze(1) # For head axis broadcasting

    q, attn = self.attention(q, k, v, mask=mask)

    # Transpose to move the head dimension back: b x lq x n x dv
    # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
    q = q.transpose(1, 2).contiguous().view(sz_batch, len_q, -1)
    q = self.fc(q)

    return q, attn
