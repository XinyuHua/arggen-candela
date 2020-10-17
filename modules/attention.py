import torch
import torch.nn as nn
import utils


class GlobalAttention(nn.Module):

    def __init__(self, query_dim, key_dim):
        super(GlobalAttention, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.linear_in = nn.Linear(query_dim, key_dim, bias=False)
        self.linear_out = nn.Linear(query_dim + key_dim, query_dim, False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()


    def score(self, h_t, h_s):
        """
        Args:
            h_t (FloatTensor): sequence of queries [batch x tgt_len x h_t_dim]
            h_s (FloatTensor): sequence of sources [batch x src_len x h_s_dim]
        Returns:
            raw attention scores for each src index [batch x tgt_len x src_len]
        """

        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        utils.aeq(src_batch, tgt_batch)

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
        h_s_ = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s_)

    def forward(self, query, memory_bank, memory_lengths=None, use_softmax=True):
        """
        Args:
            query (FloatTensor): query vectors [batch x tgt_len x dim]
            memory_bank (FloatTensor): source vectors [batch x src_len x dim]
            memory_lengths (LongTensor): source context lengths [batch]
            use_softmax (bool): use softmax to produce alignment score,
                otherwise use sigmoid for each individual one
        Returns:
            (FloatTensor, FloatTensor)
            computed attention weighted average: [batch x tgt_len x dim]
            attention distribution: [batch x tgt_len x src_len]
        """
        '''
        print("memory_bank:")
        print(memory_bank.size())
        '''

        if query.dim == 2:
            one_step = True
            query = query.unsqueeze(1)
        else:
            one_step = False

        src_batch, src_len, src_dim = memory_bank.size()
        query_batch, query_len, query_dim = query.size()
        utils.aeq(src_batch, query_batch)
        #utils.aeq(src_dim, query_dim)

        align = self.score(query, memory_bank)

        if memory_lengths is not None:
            mask = utils.sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1).long()
            align.masked_fill_((1 - mask).bool(), -float('inf'))

        if use_softmax:
            align_vectors = self.softmax(align.view(src_batch * query_len, src_len))
            # align_vectors = F.softmax(align.view(src_batch * query_len, src_len), -1)
            align_vectors = align_vectors.view(src_batch, query_len, src_len)
        else:
            align_vectors = self.sigmoid(align)
            # align_vectors = F.sigmoid(align)



        c = torch.bmm(align_vectors, memory_bank)
        # c is the attention weighted context representation
        # [batch x tgt_len x hidden_size]

        concat_c = torch.cat([c, query], 2).view(src_batch * query_len, src_dim + query_dim)
        attn_h = self.linear_out(concat_c).view(src_batch, query_len, query_dim)
        attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            batch_, dim_ = attn_h.size()
            utils.aeq(src_batch, batch_)
            utils.aeq(src_dim, dim_)
            batch_, src_l_ = align_vectors.size()
            utils.aeq(src_batch, batch_)
            utils.aeq(src_len, src_l_)

        else:

            batch_, target_l_, dim_ = attn_h.size()
            utils.aeq(target_l_, query_len)
            utils.aeq(batch_, query_batch)
            utils.aeq(dim_, query_dim)

            batch_, target_l_, source_l_ = align_vectors.size()
            utils.aeq(target_l_, query_len)
            utils.aeq(batch_, query_batch)
            utils.aeq(source_l_, src_len)

        return attn_h, align_vectors, align