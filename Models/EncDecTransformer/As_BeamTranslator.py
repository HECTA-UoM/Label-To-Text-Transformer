''' This module will handle the text generation with A* beam search. '''
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from EncDecTransformer.Modules import get_pad_mask, get_lookahead_mask


class Translator(nn.Module):
    ''' Load a trained model and translate in A* beam search fashion. '''

    def __init__(
        self, transformer,
        pad_idx, sos_idx, eos_idx, max_output_length,
        alpha, nrp_length, tree_length_product, beam_size, maximal_step_probability_difference):

        super(Translator, self).__init__()

        self.max_output_length = max_output_length
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.alpha = alpha # default = 0.6
        self.nrp_length = nrp_length # default = 4
        self.tree_length_product = tree_length_product # default = 2
        self.beam_size = beam_size
        self.maximal_step_probability_difference = maximal_step_probability_difference

        # Words for below tokens :         ["*", ".", ",", "-", "&", "'", '"', "#", "$", "%", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "=", "?", "/", '(', ')']
        self.ngrams_ignore_weak_tokens =   [115, 119, 117, 118, 111, 112, 107, 108, 109, 110, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 136, 120, 113, 114]
        self.ngrams_ignore_strong_tokens = [115, 119,      118,                     109,      121, 122, 123, 124, 125, 126, 127, 128, 129, 130,           134, 136, 120          ]
        # Weak tokens can be repeated in the last 5 tokens. Strong tokens can be repeated in the token right before.

        self.transformer = transformer
        self.transformer.eval()


    def beam(self,  src: torch.Tensor, nb_outputs: int):
        # Encoder Part
        src_mask = get_pad_mask(src, self.pad_idx)
        encoder_output = self.transformer.encoder(
            src_seq=src,
            src_mask=src_mask,
        )

        # Decoder Part
        tgt = torch.tensor([self.sos_idx])
        ordered_gens, finished = [(1, tgt)], []

        while len(finished) != nb_outputs and len(ordered_gens) > 0:

            # Get highest scored sequence and remove it from list
            top_score, top_seq = ordered_gens[-1]
            ordered_gens = ordered_gens[:-1]

            # If closed, add it to finished/top sequences
            if top_seq[-1] == self.eos_idx or len(top_seq) == self.max_output_length:
                finished.append((top_score, top_seq))
                continue

            # Generate next token probabilities
            tgt_mask = get_pad_mask(top_seq.unsqueeze(0), self.pad_idx) & get_lookahead_mask(top_seq.unsqueeze(0))
            decoder_output = self.transformer.decoder(
                trg_seq=top_seq.unsqueeze(0),
                tgt_mask=tgt_mask,
                enc_output=encoder_output,
                src_mask=src_mask,
            )

            decoder_output = F.softmax(self.transformer.fc(decoder_output), dim=-1)

            # Get k most probable next tokens
            top_k_next_scores, top_k_next_tokens = decoder_output[:, -1, :].topk(self.beam_size)

            # Apply maximal step robability difference
            if self.maximal_step_probability_difference < 1:
                # Only keep probabilities that are at most x% less probable than the most probable of the Ks
                threshold = float(top_k_next_scores.max()) * (1-self.maximal_step_probability_difference)

                # Filter the tensors based on the condition
                top_k_next_tokens = top_k_next_tokens[top_k_next_scores >= threshold]
                top_k_next_scores = top_k_next_scores[top_k_next_scores >= threshold]
            else:
                top_k_next_tokens, top_k_next_scores = top_k_next_tokens[0], top_k_next_scores[0]

            # Add these to ordered_gens list
            for i in range(len(top_k_next_scores)):
                next_score, next_gen = float(top_k_next_scores[i]), top_k_next_tokens[i].unsqueeze(0)
                new_seq = torch.cat((top_seq, next_gen), dim=-1)
                new_score = top_score * next_score

                # Penalize n-gram repeats
                if len(new_seq) > 6 and self.nrp_length > 0:
                  gram, seq_to_check = new_seq[-1], new_seq[-(self.nrp_length + 1):-1]
                  # Check non-strong tokens repeat (strong tokens can be repeated in the token right before)
                  if gram not in self.ngrams_ignore_strong_tokens and gram == seq_to_check[-1]:
                    new_score = new_score**(2 - next_score*0.5)
                  # Check non-weak tokens repeat (weak tokens can be repeated in the last n tokens)
                  elif gram not in self.ngrams_ignore_weak_tokens and gram in seq_to_check:
                    new_score = new_score**(2 - next_score*0.5)

                # Add to ordered_gens list
                ordered_gens.append((new_score, new_seq))

            # Re-sort ordered_gens list and apply length normalization
            ordered_gens = sorted(ordered_gens, key=lambda x: math.log(x[0]) / (((5+len(x[1]))**self.alpha) / (6**self.alpha)), reverse=False)[-int(nb_outputs*self.tree_length_product):]


        # IMPORTANT:
        # Returns top n finished/closed outputs
        # If less than n outputs are produced, you may want to reduce maximal_step_probability_difference (less varying outputs)
        return [e[1][1:-1] for e in finished] # 1:-1 to remove start and end tokens


    def translate_sentence(self, src: torch.Tensor, nb_outputs: int):
        with torch.no_grad():
            return self.beam(src, nb_outputs)

