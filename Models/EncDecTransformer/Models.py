import torch
import torch.nn as nn
from EncDecTransformer.Layers import EncoderLayer, DecoderLayer
from EncDecTransformer.Modules import PositionalEncoding, get_pad_mask, get_lookahead_mask


class Encoder(nn.Module):

    def __init__(self, 
        vocab_size: int, 
        n_layers: int, 
        n_head: int, 
        d_v: int,
        d_model: int, 
        d_hid: int, 
        pad_idx: int,
        keyword_max_length: int,
        dropout: float = 0.1
    ):

        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_enc = PositionalEncoding(d_model, keyword_max_length)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_hid, d_v, d_v, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask):

        enc_output = self.dropout(self.positional_enc(self.embedding(src_seq)))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layers:
            enc_output = enc_layer(enc_output, src_mask)

        return enc_output



class Decoder(nn.Module):

    def __init__(self,
        vocab_size: int,
        n_layers: int,
        n_head: int,
        d_v: int,
        d_model: int,
        d_hid: int,
        pad_idx: int,
        description_max_length: int,
        dropout: float=0.1
    ):

        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_enc = PositionalEncoding(d_model, description_max_length)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_hid, d_v, d_v, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq: torch.Tensor, tgt_mask: torch.Tensor, enc_output: torch.Tensor, src_mask:torch.Tensor) -> torch.Tensor:

        dec_output = self.dropout(self.positional_enc(self.embedding(trg_seq)))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return dec_output



class Transformer(nn.Module):
    def __init__(self,
        keyword_max_length: int,
        description_max_length: int,
        vocab_size: int,
        pad_idx: int,
        d_model: int,
        d_v: int,
        d_hid: int,
        n_head: int,
        n_layers: int,
        dropout: float=0.3
    ):

        super(Transformer, self).__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_head=n_head,
            d_v=d_v,
            d_model=d_model,
            d_hid=d_hid,
            pad_idx=pad_idx,
            keyword_max_length=keyword_max_length,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_head=n_head,
            d_v=d_v,
            d_model=d_model,
            d_hid=d_hid,
            pad_idx=pad_idx,
            description_max_length=description_max_length,
            dropout=dropout,
        )

        self.fc = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:

        src_mask = get_pad_mask(src, self.pad_idx)
        tgt_mask = get_pad_mask(tgt, self.pad_idx) & get_lookahead_mask(tgt)

        encoder_output = self.encoder(
            src_seq=src,
            src_mask=src_mask,
        )

        decoder_output = self.decoder(
            trg_seq=tgt,
            tgt_mask=tgt_mask,
            enc_output=encoder_output,
            src_mask=src_mask,
        )

        output = self.fc(decoder_output)

        return output
