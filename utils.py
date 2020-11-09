import numpy as np
import torch
import glob
import os

DATA_DIR = "./data/"
WEMB_DIR = "./embeddings/glove.6B.300d.txt"

def find_ckpt_path(exp_name, epoch_id):
    """Find model checkpoint path given the exp_name and epoch_id"""
    if epoch_id == -1:
        glob_pattern = "epoch_*_"
    else:
        glob_pattern = f"epoch_{epoch_id}_"

    candidates = glob.glob(f"checkpoints/{exp_name}/{glob_pattern}*")

    def get_epoch_id_from_path(path):
        file_name = os.path.basename(path)
        e_id = file_name.split("_")[1]
        return int(e_id)

    sorted_cands = sorted(candidates, key=lambda x: get_epoch_id_from_path(x),
                          reverse=True)
    return sorted_cands[0]

def load_glove_emb(vocab):
    """
    Load pre-trained GloVe embeddings for words in the vocabulary. If no
    embedding exists, initialize randomly.
    Args:
        vocab (Vocab): a Vocabulary object contains
    """

    random_init = np.random.uniform(-.25, .25, [len(vocab), 300])
    random_init[0] = np.zeros(300) # padding

    for ln in open(WEMB_DIR):
        lsplit = ln.strip().split('\t')
        if len(lsplit) < 300:
            continue

        word = lsplit[0]
        word_idx = vocab.get_idx(word)
        if word_idx == vocab.unk_idx:
            continue

        vec = np.array([float(x) for x in lsplit[1:]])
        random_init[word_idx] = vec
    return random_init

def collate_tokens(values, pad_idx):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][:len(v)])
    return res

def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            r = {key: _apply(value) for key, value in x.items()}
            return r
            # return {
            #     key: _apply(value)
            #     for key, value in x.items()
            # }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments)

def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def get_sequence_mask_from_length(seq_len, max_len):
    if max_len is None:
        max_len = seq_len.data.max()

    batch_size = seq_len.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if seq_len.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_len_expand = (seq_len.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_len_expand