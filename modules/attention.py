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


    def score(self, query, key):
        """
        Multiplicative attention (Luong attention):
        align = (query * W)^T * key (shape: bsz x tgt_len x src_len, where
        tgt_len is the sequence length of query, src_len is the sequence length
        of key.)

        Args:
            query (FloatTensor): sequence of queries [batch x tgt_len x query_dim]
            key (FloatTensor): sequence of sources [batch x src_len x key_dim]
        Returns:
            raw attention scores for each src index [batch x tgt_len x src_len]
        """
        k_bsz, k_len, k_dim = key.shape
        q_bsz, q_len, q_dim = query.shape

        assert k_bsz == q_bsz
        assert k_dim == self.key_dim
        assert q_dim == self.query_dim

        query = query.view(-1, q_dim)
        query_transformed = self.linear_in(query).view(q_bsz, q_len, k_dim)
        key_transpose = key.transpose(1, 2)

        align_raw = torch.bmm(query_transformed, key_transpose)
        return align_raw

    def forward(self, query, memory_bank, memory_lengths=None, use_softmax=True):
        """
        Args:
            query (FloatTensor): query vectors [batch x tgt_len x q_dim]
            memory_bank (FloatTensor): source vectors [batch x src_len x k_dim]
            memory_lengths (LongTensor): source context lengths [batch]
            use_softmax (bool): use softmax to produce alignment score,
                otherwise use sigmoid for keyphrase selection
        Returns:
            attn_h (FloatTensor, batch x tgt_len x k_dim): weighted value vectors after attention
            attn_vectors (FloatTensor, batch x tgt_len x src_len) : normalized attention scores
            align (FloatTensor, batch x tgt_len x src_len): raw attention scores used for loss calculation
        """


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
            align_vectors = align_vectors.view(src_batch, query_len, src_len)
        else:
            align_vectors = self.sigmoid(align)

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