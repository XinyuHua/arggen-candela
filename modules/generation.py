import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.beam import Beam

def tile(x, count, dim=0):
    """Tile x on dimension `dim` `count` times."""
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1)\
         .transpose(0, 1)\
         .repeat(count, 1)\
         .transpose(0, 1)\
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def pad_list_to_array(inputs):
    """
    inputs is a list of lists like [[3], [4], [5, 6]]
    results should be torch.tensor([[3, 0, 0],[4, 0, 0], [5, 6, 0]])
    """
    max_len = max([len(x) for x in inputs])
    results = np.zeros([len(inputs), max_len], dtype='int')
    for lid, ln in enumerate(inputs):
        results[lid][:len(ln)] = ln
    return results

def pad_3d_list_to_array(inputs):
    """Pad 3d list inputs into 3d array.

    Args:
        inputs (list[list[list[int]]]): e.g. [[[1], [4,5]], [[1], [2]]]
    Returns:
        padded (numpy.ndarray)
    """
    max_len = max([len(x) for x in inputs])
    max_ph_len = 0
    for item in inputs:
        max_ph_len = max(max_ph_len, max([len(ph) for ph in item]))

    padded = np.zeros([len(inputs), max_len, max_ph_len], dtype=np.int)
    for ix, item in enumerate(inputs):
        for pid, ph in enumerate(item):
            if isinstance(ph, list):
                padded[ix][pid][:len(ph)] = ph
            else:
                padded[ix][pid][:len(ph)] = ph.cpu().data
    return padded


def make_3d_tensor_from_list_of_2d(tensor_list):
    max_dim_0 = max([item.shape[0] for item in tensor_list])
    max_dim_1 = max([item.shape[1] for item in tensor_list])
    output = torch.zeros([len(tensor_list), max_dim_0, max_dim_1], dtype=torch.long)
    for i, item in enumerate(tensor_list):
        for j, t in enumerate(item):
            output[i][j] = t
    return output

def make_tensor_template(value, size):
    template = torch.zeros(size, dtype=torch.long).cuda()
    template[0][0] = value
    return template

class DecodingStrategy:

    def __init__(self, model, vocab, args):
        self.model = model
        self.vocab = vocab
        self.quiet = args.quiet
        self.stype_map = {0: "FILL", 1: "CONTENT", 2: "EOS"}
        self.use_goldstandard_plan = args.use_goldstandard_plan

        self.max_tgt_sent = args.max_tgt_sent
        self.min_tgt_sent = args.min_tgt_sent
        self.max_tgt_token = args.max_tgt_token
        self.beam_size = args.beam_size
        self.block_ngram_repeat = args.block_ngram_repeat
        self.max_token_per_sentence = args.max_token_per_sentence
        self.max_phrase_selection_time = args.max_phrase_selection_time


    def enforce_selection_constraints(self, ph_attn_probs, selection_history, sos_mask):
        """Make sure the selection results satisfy the following constraints:

        1. no SOS will be selected after 1st sentence
        2. each selection cannot be selected for more than `self.max_phrase_selection_time`

        Args:
            ph_attn_probs (bsz x 1 x max_ph_bank_size)
            selection_history (bsz x max_ph_bank_size)
            sos_mask (bsz x max_ph_bank_size): boolean value that masks SOS phrase
                (True for SOS, False for others).
        """
        ph_attn_probs.masked_fill_(sos_mask, 0.0)
        selection_mask = selection_history >= self.max_phrase_selection_time
        ph_attn_probs.masked_fill_(selection_mask, 0.0)


    def sentence_decoding_goldstandard(self, ph_bank, ph_bank_len, ph_sel_tensor, stype):
        """Use gold-standard selection results to calculate sp states.

        Args:
            - ph_bank
            - ph_bank_len
            - ph_sel_ind
            - ph_sel_tensor
            - stype
        Output:
            - ph_sel_results (bsz x max_sent_num x
            - stype_results (bsz x max_sent_num)
            - sp_dec_outs (bsz x max_sent_num x dim)
        """
        batch_size = ph_bank.shape[0]
        max_ph_size = ph_bank.shape[2]

        ph_bank_embedded = self.model.sp_dec.embedding(ph_bank)
        ph_bank_embedded = torch.sum(ph_bank_embedded, -2)

        sp_dec_outs, _, _, _ = self.model.sp_dec(
            tgt=ph_sel_tensor, memory_bank=ph_bank_embedded,
            memory_lengths=ph_bank_len,
        )

        eos_template = make_tensor_template(value=self.vocab.eos_idx,
                                            size=(batch_size, max_ph_size))
        ph_sel_results = []
        sent_len = ph_sel_tensor.shape[1]
        stype_results = []
        for sent_id in range(sent_len):
            cur_sel_raw = ph_sel_tensor[:, sent_id, :, :]
            max_len = (cur_sel_raw.sum(-1) > 0).sum(-1).max()
            cur_sel = cur_sel_raw[:, :max_len, :]

            ph_sel_results.append(cur_sel)
            stype_results.append(stype[:, sent_id])

        return ph_sel_results, stype_results, sp_dec_outs

    def sentence_decoding_greedy(self, ph_bank, ph_bank_len):
        """Run greedy decoding on sentence decoder.
        Input:
            - ph_bank (batch_size x max_ph_num x max_ph_len)
            - ph_bank_len (batch_size)
        Output:
            - ph_sel_results: (batch_size x max_sent_num x max_ph_num)
            - stype_results: (batch_size x max_sent_num)
        """
        batch_size = len(ph_bank_len)
        max_ph_size = ph_bank.shape[2]

        # ph_bank_embedded (bsz x bank_size x emb_dim)
        ph_bank_embedded = self.model.sp_dec.embedding(ph_bank)
        ph_bank_embedded = torch.sum(ph_bank_embedded, -2)

        # starts generation with SOS symbols as initial keyphrase selection
        dec_input_ids = torch.full(size=(batch_size, 1, 1, 1),
                                   fill_value=self.vocab.sos_idx,
                                   dtype=torch.long).cuda()
        finished = torch.full(size=[batch_size], fill_value=0,
                              dtype=torch.uint8).cuda()

        eos_template = make_tensor_template(value=self.vocab.eos_idx, size=(1, max_ph_size))
        sos_template = make_tensor_template(value=self.vocab.sos_idx, size=(1, max_ph_size))

        # find SOS positions in each instance
        sos_mask = (ph_bank == sos_template).all(-1).unsqueeze(1)

        sp_hidden_states = []
        ph_sel_results = [dec_input_ids.squeeze(1)]
        stype_results = []

        # keep track of how many time each phrase has been selected so far
        selection_history = torch.zeros([batch_size, 1, ph_bank.shape[1]],
                                        dtype=torch.long).cuda()

        cur_sent_num = 0
        while cur_sent_num < self.max_tgt_sent:
            assert dec_input_ids.ndim == 4

            # ph_attn_probs: (bsz x 1 x ph_bank_len)
            sp_dec_outs, ph_attn_probs, ph_attn_logits, stype_logits = self.model.sp_dec(
                tgt=dec_input_ids, memory_bank=ph_bank_embedded,
                memory_lengths=ph_bank_len,
            )

            self.enforce_selection_constraints(ph_attn_probs, selection_history, sos_mask)

            # force finish if limit is reached
            if cur_sent_num == self.max_tgt_sent - 1:
                finished[:] = 1

            stype_preds = torch.argmax(stype_logits, dim=-1)
            # if sequence is finished, override type to EOS (2)
            stype_preds[finished==1] = 2
            stype_results.append(stype_preds.squeeze())

            selected_ph_ids = []
            cur_selected_indices = (ph_attn_probs >= 0.5)
            selection_history += cur_selected_indices
            selection_results = cur_selected_indices.view(batch_size, -1, 1) * ph_bank
            for b, cur_sel in enumerate(selection_results):
                if finished[b]:
                    selected_ph_ids.append(eos_template) # append [EOS] to all finished instances
                else:
                    cur_phrases = cur_sel[cur_sel.sum(-1) != 0]
                    # SOS cannot be selected, since sos_idx happens to be 1, we can remove
                    # the corresponding row with the following condition
                    cur_phrases = cur_phrases[cur_phrases.sum(-1) != 1]

                    # if EOS is selected after at least `self.min_tgt_sent`,
                    # set finished properly and modify selection.
                    # if EOS is selected before `self.min_tgt_sent`, remove
                    # EOS from selection.
                    if (eos_template == cur_phrases).all(-1).any():
                        if cur_sent_num > self.min_tgt_sent:
                            finished[b] = 1
                            cur_phrases = eos_template
                        else:
                            cur_phrases = cur_phrases[cur_phrases!=eos_template]

                    selected_ph_ids.append(cur_phrases)

            dec_input_ids = make_3d_tensor_from_list_of_2d(selected_ph_ids)
            ph_sel_results.append(dec_input_ids)
            dec_input_ids = dec_input_ids.unsqueeze(1).cuda()
            sp_hidden_states.append(sp_dec_outs)
            cur_sent_num += 1

            if finished.all():
                break

        sp_hidden_states = torch.cat(sp_hidden_states, 1)
        return ph_sel_results, stype_results, sp_hidden_states



    def generate(self, batch):
        """Run greedy decoding for sentence decoding (sp_dec), and beam search
        for token decoding (wd_dec)."""

        batch_size = len(batch["id"])
        enc_outs, enc_final = self.model.enc(batch["enc_src"],
                                             batch["enc_src_len"])

        memory_bank = tile(enc_outs, self.beam_size, dim=0)
        memory_lengths = tile(batch["enc_src_len"], self.beam_size)

        self.model.sp_dec.init_state(encoder_final=enc_final)
        self.model.wd_dec.init_state(encoder_final=enc_final)
        self.model.wd_dec.map_state(lambda state, dim: tile(state, self.beam_size, dim=dim))

        if self.use_goldstandard_plan:
            sp_outputs = self.sentence_decoding_goldstandard(
                ph_bank=batch['ph_bank_tensor'],
                ph_bank_len=batch['ph_bank_len_tensor'],
                ph_sel_tensor=batch['ph_sel_tensor'],
                stype=batch['sent_types']
            )
        else:
            sp_outputs = self.sentence_decoding_greedy(
                ph_bank=batch["ph_bank_tensor"],
                ph_bank_len=batch["ph_bank_len_tensor"]
            )
        ph_sel_results = sp_outputs[0]
        stype_results = sp_outputs[1]
        sp_hidden_states = sp_outputs[2]
        _, max_pred_sent_num, sp_hidden_dim = sp_hidden_states.shape

        if not self.quiet:
            # print phrase selection and sentence type results
            self.print_sp_results(ph_sel_results, stype_results)

        softmax = nn.Softmax(dim=-1)
        beams = [Beam(self.beam_size, vocab=self.vocab, min_length=10,
                      block_ngram_repeat=self.block_ngram_repeat)
                 for _ in range(batch_size)]

        # first sentence has only <SOS>, therefore should only output one token
        max_token_in_sents = [1] + [self.max_token_per_sentence for _ in range(max_pred_sent_num - 1)]

        for sent_id in range(max_pred_sent_num):
            cur_sp_dec_outs = sp_hidden_states[:, sent_id, :].unsqueeze(1)
            cur_sp_dec_outs_tile = tile(cur_sp_dec_outs, self.beam_size, 0)

            for step in range(max_token_in_sents[sent_id]):

                if all((b.done() for b in beams)):
                    break

                word_dec_input = torch.stack([b.get_current_state() for b in beams])
                word_dec_input = word_dec_input.view(-1, 1, 1).cuda()

                wd_outputs = self.model.wd_dec.forward_onestep(
                    dec_inputs=word_dec_input, enc_memory_bank=memory_bank,
                    enc_memory_len=memory_lengths,
                    sp_hidden_state=cur_sp_dec_outs_tile)
                dec_logits = wd_outputs[0]
                dec_probs = softmax(dec_logits).view(batch_size, self.beam_size, -1)

                select_indices_array = []
                for ix, beam in enumerate(beams):
                    beam.advance(probs=dec_probs[ix, :])
                    select_indices_array.append(beam.get_current_origin() + ix * self.beam_size)
                select_indices = torch.cat(select_indices_array)
                self.model.wd_dec.map_state(lambda state, dim: state.index_select(dim, select_indices))

                if not self.quiet:
                    # print results in the top beam in all instances
                    to_print = f"sentence {sent_id:<2d} step-{step:<2d} | "

                    for ix, beam in enumerate(beams):
                        cur_words = beam.get_current_state()
                        top_beam = cur_words[0].item()
                        top_beam_word = self.vocab.get_word(idx=top_beam)
                        to_print += f" {top_beam_word:<10s} | "
                    print(to_print)


        results = []
        for b in beams:
            scores, ks = b.sort_finished(minimum=1)
            hyps = []
            for i, (times, k) in enumerate(ks[:1]):
                hyp = b.get_hyp(times, k)
                hyps.append([tok_id.item() for tok_id in hyp])
            results.append(hyps)

        stype_results_str = []
        for b in range(batch_size):
            cur_stype = [stype_step[b].item() for stype_step in stype_results]
            end = cur_stype.index(2) if 2 in cur_stype else len(cur_stype)
            cur_stype = cur_stype[:end]
            stype_results_str.append([self.stype_map[item] for item in cur_stype])

        ph_sel_results_str = []
        for b in range(batch_size):
            cur_ph_sel = [ph_step[b] for ph_step in ph_sel_results[1:]]
            cur_ph_sel_str = []
            for sent in cur_ph_sel:
                sent = sent[sent.sum(-1) > 0]
                sent_str = [' '.join(self.vocab.decode(item)) for item in sent]
                cur_ph_sel_str.append(sent_str)
                if sent_str == ['EOS']:
                    break
            ph_sel_results_str.append(cur_ph_sel_str)

        return results, stype_results_str, ph_sel_results_str


    def print_sp_results(self, ph_sel_results, stype_results):
        """Print phrase selection and sentence type results to console.

        Args:
            ph_sel_results (List[tensor]): by sentence id, each element is of
                shape (bsz x max_ph_sel x max_ph_len)
            stype_results (List[tensor]): by sentence id, each element is of
                shape (bsz)
        """
        batch_size = len(stype_results[0])

        ph_sel_words = [[] for _ in range(batch_size)]
        stype_words = [[] for _ in range(batch_size)]

        for sent_id, (ph_sel, stype) in enumerate(zip(ph_sel_results, stype_results)):
            # ph_sel (bsz x max_ph_sel x max_ph_len)
            # stype (bsz)
            for b in range(batch_size):
                cur_ph_sel = ph_sel[b]
                # remove paddings
                cur_ph_sel = cur_ph_sel[cur_ph_sel.sum(-1) > 0]
                cur_ph_sel_str = [' '.join(self.vocab.decode(item)) for item in cur_ph_sel]
                cur_stype = stype[b]
                if cur_stype == 2:
                    continue

                ph_sel_words[b].append(cur_ph_sel_str)
                stype_words[b].append(self.stype_map[cur_stype.item()])

        for b in range(batch_size):
            to_print = f"INSTANCE-{b}\n"
            for sid, (ph, stype) in enumerate(zip(ph_sel_words[b], stype_words[b])):
                to_print += f"sent {sid} type={stype}, selected phrase={', '.join(ph)}\n"
            print(to_print + "\n")
