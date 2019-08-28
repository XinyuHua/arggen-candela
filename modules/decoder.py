import torch
import torch.nn as nn
import utils
from modules import attention


class RNNDecoderBase(nn.Module):
    def __init__(self, hidden_size, embedding, emb_size, out_vocab_size):
        super(RNNDecoderBase, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = embedding
        self.emb_size = emb_size
        self.state = {}

        self.LSTM = nn.LSTM(input_size=emb_size,
                            hidden_size=hidden_size,
                            num_layers=2,
                            dropout=0.2,
                            batch_first=True,
                            bias=True)
        self.init_attn()
        self.readout = nn.Linear(hidden_size, out_vocab_size, bias=True)

    def init_attn(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def init_state(self, encoder_final):
        """ Init decoder state with last state of the encoder """

        def _fix_enc_hidden(hidden):
            hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
            return hidden

        self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final])
        batch_size = self.state["hidden"][0].size(1)


class WordDecoder(RNNDecoderBase):
    """
    word decoder
    s^r_i = g^r(s^r_{i-1}, tanh(W1x_{i-1} + W2s^p_j + W3c^r_{i-1} + b^r))
    c^r_{i-1} = attn(h, s^r_{i-1})
    output = softmax(W^ro tanh(W^c[c^r_{i-1}; s^r_{i-1}]))
    """

    def __init__(self, hidden_size, word_emb, word_emb_size, word_vocab_size):
        super(WordDecoder, self).__init__(hidden_size,
                                          word_emb,
                                          word_emb_size,
                                          word_vocab_size)
        self.word_transformation = nn.Linear(word_emb_size, word_emb_size, bias=True)
        self.planner_transformation = nn.Linear(hidden_size, word_emb_size, bias=True)

        return

    def init_attn(self):
        """
        Initialize attention for word decoder
        """
        self.enc_attn = attention.GlobalAttention(query_dim=self.hidden_size,
                                                  key_dim=self.hidden_size,
                                                  attn_type="bilinear")
        return

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])

    def forward_onestep(self, dec_inputs, enc_memory_bank, enc_memory_len, sp_hidden_state):
        """
        run forward on word decoder for one step
        Args:
            dec_inputs: [batch_size x 1]
            enc_memory_bank: [batch_size x max_ph_num]
            enc_memory_len: batch_size,
            sp_hidden_state: [batch_size x hz]
        """
        dec_inputs = dec_inputs.squeeze(-1)
        word_emb = self.embedding(dec_inputs)

        word_transformed = self.word_transformation(word_emb)
        planner_transformed = self.planner_transformation(sp_hidden_state)
        merged_inputs = word_transformed + planner_transformed

        rnn_input = torch.tanh(merged_inputs)

        rnn_output, dec_state = self.LSTM(rnn_input, self.state["hidden"])
        self.state["hidden"] = dec_state

        dec_outs, enc_attn, _ = self.enc_attn(
            rnn_output.contiguous(),
            enc_memory_bank.contiguous(),
            memory_lengths=enc_memory_len)
        readouts = self.readout(dec_outs)
        return readouts, dec_outs, enc_attn

    def forward(self, tgt_words, tgt_word_len, enc_memory_bank,
                enc_memory_len, sent_planner_output, sent_id_template,
                sent_mask_template):
        """
        The word decoder forward path depends on both encoder hidden states
        and the corresponding hidden states of sentence planner.
        Args:
            tgt_words (LongTensor): the sequence of word ids for current batch
                                    [batch_size x max_rr_len]
            tgt_word_len (LongTensor): the length information for the equence
                                    [batch_size]
            enc_memory_bank (FloatTensor): hidden states for encoder
                                    [batch_size x max_op_len x enc_hidden_size]
            enc_memory_len (LongTensor): the length of the encoder states
                                    [batch_size]
            sent_planner_output (FloatTensor): the output states of sentence planner
                                    [batch_size x max_rr_sent_cnt x dec_out_dim]
            sent_id_template (LongTensor): sentence id for each token
                                    [batch_size x max_rr_len]
            sent_mask_template (LongTensor): mask for token padding
                                    [batch_size x max_rr_len]
        """

        max_rr_len = sent_id_template.size(1)
        sent_planner_output_dim = sent_planner_output.size(-1)
        sent_id_template_expanded = sent_id_template.unsqueeze(dim=-1) \
            .expand(-1, max_rr_len,
                    sent_planner_output_dim)

        token_distributed_sent_planner_output = torch.gather(
            sent_planner_output, 1, sent_id_template_expanded)
        # at this point, the size of the following tensor should be:
        # [batch_size x max_rr_len x sp_out_dim]
        token_distributed_sent_planner_output_masked = sent_mask_template.float() \
                                                       * token_distributed_sent_planner_output

        word_emb = self.embedding(tgt_words)
        merged_inputs = self.word_transformation(word_emb) \
                        + self.planner_transformation(token_distributed_sent_planner_output_masked)
        rnn_input = torch.tanh(merged_inputs)
        rnn_output, dec_state = self.LSTM(rnn_input, self.state["hidden"])
        self.rnn_output = rnn_output
        dec_outs, enc_attn, _ = self.enc_attn(
            rnn_output.contiguous(),
            enc_memory_bank.contiguous(),
            memory_lengths=enc_memory_len)
        readouts = self.readout(dec_outs)

        return dec_state, dec_outs, enc_attn, readouts


class SentencePlanner(RNNDecoderBase):
    """
    sentence planner decoder
    s^p_j = g^p(s^p_{j-1}, c^p_{j-1})
    c^p_{j-1} = attn(v^c_k, s^p_{j-1})
    output = softmax(W^{out} x
    """

    def __init__(self, hidden_size, word_emb, word_emb_size):
        self.word_emb_size = word_emb_size
        super(SentencePlanner, self).__init__(hidden_size, word_emb, word_emb_size, 3)

    def forward(self, tgt, memory_bank, memory_lengths=None):
        dec_state, dec_outs, attns, attn_raw, readouts = self._run_forward_pass(
            tgt, memory_bank, memory_lengths)

        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        return dec_outs, attns, attn_raw, readouts

    def _run_forward_pass(self, ph_sel, phrase_bank, phrase_lengths=None):
        """
        Args:
            ph_sum_emb (FloatTensor): a tensor of phrase embeddings for each
                                      RR sentence
                                     [batch x max_sent_num x ph_emb_dim]
            phrase_bank (FloatTensor): embeddings for phrase collections
                            [batch x len x nfeats]
            phrase_lengths (LongTensor): the lengths of phrase collections
        Returns:
            dec_state (Tensor): final hidden state from the decoder.
            dec_outs ([FloatTensor]): an array of output of every time step
                                      from the decoder.
            ph_attns ([FloatTensor]): phrase attention Tensor array of every
                                      time step from the decoder.
        """
        ph_sel_emb = self.embedding(ph_sel)
        ph_sel_emb = torch.sum(ph_sel_emb, -2)
        ph_sum_emb = torch.sum(ph_sel_emb, -2)
        rnn_output, dec_state = self.LSTM(ph_sum_emb, self.state["hidden"])
        self.rnn_output = rnn_output

        ph_batch, ph_len, _ = ph_sum_emb.size()

        output_batch, output_len, _ = rnn_output.size()

        utils.aeq(ph_len, output_len)
        utils.aeq(ph_batch, output_batch)

        dec_outs, ph_attn, ph_attn_raw = self.ph_attn(
            rnn_output.contiguous(),
            phrase_bank.contiguous(),
            memory_lengths=phrase_lengths,
            use_softmax=False
        )

        readouts = self.readout(dec_outs)

        # dec_outs: [batch_size x max_sent_num x ph_emb_dim]
        # readouts: [batch_size x max_sent_num x ph_vocab_size]

        return dec_state, dec_outs, ph_attn, ph_attn_raw, readouts

    def init_attn(self):
        """
        Initialize attention for sentence planner decoder
        """
        self.ph_attn = attention.GlobalAttention(query_dim=self.hidden_size,
                                                 key_dim=self.word_emb_size,
                                                 attn_type="bilinear")
        return