''' This module will handle the text generation with beam search. '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from EncDecTransformer.Modules import get_pad_mask, get_lookahead_mask


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
        self, model, max_seq_len,
        src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        self.max_seq_len = max_seq_len

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_lookahead_mask(trg_seq)
        dec_output = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.fc(dec_output), dim=-1)


    def _get_init_state(self, src_seq, src_mask, beam_size):
        enc_output = self.model.encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step, beam_size):
        assert len(scores.size()) == 1

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self, src_seq, beam_size, nb_outputs):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1
        assert beam_size >= nb_outputs, "Beam size must be greater or equal to number of outputs."

        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, self.max_seq_len), self.trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, alpha = self.max_seq_len, self.alpha
        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask, beam_size)
            ans_idx = list(range(nb_outputs))   # default

            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step, beam_size)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).topk(nb_outputs)
                    ans_idx = ans_idx.tolist()
                    break

        return [pred[1:pred.index(self.trg_eos_idx)] if self.trg_eos_idx in pred else pred[1:] for pred in gen_seq[ans_idx].tolist()]