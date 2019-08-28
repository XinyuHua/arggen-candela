import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class GlobalAttention(nn.Module):
    """
    attention types:
        bilinear: H_j^T W_a q (bilinear)
        mlp: v_a^T tanh(W_a q + U_a h_j)
    """


    def __init__(self, query_dim, key_dim, attn_type="bilinear"):
        super(GlobalAttention, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        assert attn_type in ["bilinear", "mlp"]

        self.attn_type = attn_type

        if self.attn_type == "bilinear":
            self.linear_in = nn.Linear(query_dim, key_dim, bias=False)
            self.linear_out = nn.Linear(query_dim + key_dim, query_dim, False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
            self.linear_out = nn.Linear(dim * 2, dim, True)

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
        #utils.aeq(src_dim, tgt_dim)

        if self.attn_type == "bilinear":
            h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
            h_t_ = self.linear_in(h_t_)
            h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
            h_s_ = h_s.transpose(1, 2)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

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
        '''
        print("memory_lengths:")
        print(memory_lengths.size())
        print(memory_lengths)
        '''

        if memory_lengths is not None:
            mask = utils.sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)
            align.masked_fill_(1 - mask, -float('inf'))
        '''
        print("align:")
        print(align)
        print(align.size())
        '''
        if use_softmax:
            align_vectors = F.softmax(align.view(src_batch * query_len, src_len), -1)
            align_vectors = align_vectors.view(src_batch, query_len, src_len)
        else:
            align_vectors = F.sigmoid(align)
        '''
        print("align after normalize:")
        print(align_vectors)
        print("align_vectors:")
        print(align_vectors)
        print(align_vectors.size())
        print("memory_bank:")
        print(memory_bank)
        print(memory_bank.size())
        '''

        c = torch.bmm(align_vectors, memory_bank)
        # c is the attention weighted context representation
        # [batch x tgt_len x hidden_size]
        '''
        print("c:")
        print(c.size())
        print("query:")
        print(query.size())
        '''

        concat_c = torch.cat([c, query], 2).view(src_batch * query_len, src_dim + query_dim)
        '''
        print("concat_c:")
        print(concat_c.size())
        '''
        attn_h = self.linear_out(concat_c).view(src_batch, query_len, query_dim)
        if self.attn_type == "bilinear":
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