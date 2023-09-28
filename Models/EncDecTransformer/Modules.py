import torch
import torch.nn as nn
import torch.nn.functional as F


def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.FloatTensor([d_model]))
    return pos * angle_rates

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_lookahead_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = get_angles(position, torch.arange(d_model, dtype=torch.float), d_model)

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(div_term[:, 0::2])
        pe[:, 1::2] = torch.cos(div_term[:, 1::2])

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ScaledDotProductAttention(nn.Module):

  def __init__(self, temperature: float, attn_dropout: float = 0.1):
    super(ScaledDotProductAttention, self).__init__()
    self.temperature = temperature
    self.dropout = nn.Dropout(attn_dropout)

  def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)

    attn = self.dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)

    return output, attn