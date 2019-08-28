# Author: Xinyu Hua
"""utility functions for model training and testing"""

import torch
import json
import os
import logging
import time
import numpy as np
import pickle as pkl

PAD_id = 0
SOS_id = 1
SEP_id = 2
EOS_id = 3
UNK_id = 4
PH_PAD_id = 0
PH_SOS_id = 1
PH_EOS_id = 2

setup_configs = {
    "oracle": {"src": ["op", "rr_psg_kp", "rr_psg", "tid"], "tgt": ["rr"]},
    "system": {"src": ["op", "op_psg_kp", "op_psg"], "tgt": ["rr"]},
}

DATA_DIR = os.environ["DATA_DIR_PREFIX"]
EXP_DIR = os.environ["EXP_DIR_PREFIX"]
WEMB_DIR = os.environ["WEMB_DIR"]

def load_test_data(setup="oracle", demo=False):
    path = DATA_DIR + "%s_test.jsonl" % setup
    dataset = dict()
    dataset["test"] = {"src": {item:[] for item in setup_configs[setup]["src"]},
                       "tgt": {item:[] for item in setup_configs[setup]["tgt"]}}

    ln_cnt = 0
    for ln in open(path):
        cur_obj = json.loads(ln)
        ln_cnt += 1

        for comp in ["src", "tgt"]:

            for item in dataset["test"][comp]:
                dataset["test"][comp][item].append(cur_obj[item])

        if demo and ln_cnt >= 100:
            break
    logging.info("Test data loaded. %d pairs in total" % (ln_cnt))
    return dataset


def load_train_data(demo=False):
    set_type_path = {set_type: DATA_DIR + ("oracle_%s.jsonl" % set_type)
                        for set_type in ["train", "dev"]}
    dataset = dict()
    dataset["train"] = {"src": {item: [] for item in setup_configs["oracle"]["src"]},
                        "tgt": {item: [] for item in setup_configs["oracle"]["tgt"]}}

    dataset["dev"] = {"src": {item: [] for item in setup_configs["oracle"]["src"]},
                      "tgt": {item: [] for item in setup_configs["oracle"]["tgt"]}}

    for set_type in ["train", "dev"]:
        ln_cnt = 0
        logging.info("loading %d data..." % set_type)

        for ln in open(set_type_path[set_type]):
            cur_obj = json.loads(ln)
            ln_cnt += 1

            for comp in ["src", "tgt"]:
                for item in dataset[set_type][comp]:
                    dataset[set_type][comp][item].append(cur_obj[item])

            if demo and ln_cnt >= 100:
                break

        logging.info("%s data loaded, %d samples in total" % (set_type, ln_cnt))
    return dataset

def load_vocab():
    path = DATA_DIR + "vocab.txt"

    word2id = dict()
    id2word = list()

    for ln in open(path):
        wid, word, freq = ln.strip().split("\t")
        word2id[word] = int(wid)
        id2word.append(word)

        if len(id2word) == 50000:
            break

    return word2id, id2word


def load_glove_emb(word2id):
    """
    Load Glove embedding for words in the vocabulary, if no embedding exist, initialize randomly
     Params:
      `word2id`: a dictionary mapping word to id
    """

    random_init = np.random.uniform(-.25, .25, len(word2id), 300)
    random_init[0] = np.zeros(300)

    for ln in open(WEMB_DIR):
        lsplit = ln.strip().split("\t")
        if len(lsplit) < 300:
            continue

        word = lsplit[0]

        if not word in word2id:
            continue

        wid = word2id[word]
        vec = np.array([float(x) for x in lsplit[1:]])
        random_init[wid] = vec
    return random_init


def pad_text_id_list_into_array(batch_text_lists, max_len=500, add_start=True, sos_id=SOS_id, eos_id=EOS_id):
    """
    Pad text id list into array.
     Params:
      `batch_text_lists`: a list of word ids without adding SOS or EOS
      `max_len`: maximum allowed length for words (including SOS and EOS)
      `add_start`: boolean, denotes whether to add "SOS" at the beginning of the sequence, used for decoder
      `sos_id`: integer word id for SOS token
      `eos_id`: integer word id for EOS token
    """

    batch_size = len(batch_text_lists)
    max_word_num_in_batch = max([len(x) + 1 for x in batch_text_lists])
    if add_start:
        max_word_num_in_batch += 1

    max_word_num_in_batch = min(max_word_num_in_batch, max_len)

    word_inputs = np.zeros([batch_size, max_word_num_in_batch]).astype("float32")
    word_targets = np.zeros([batch_size, max_word_num_in_batch]).astype("float32")
    word_count = np.zeros(batch_size)

    for sample_id, sample in enumerate(batch_text_lists):
        if add_start:
            truncated_sample = sample[:max_word_num_in_batch - 1]
            input_sample = [sos_id] + truncated_sample
            target_sample = truncated_sample + [eos_id]
            word_count[sample_id] = len(truncated_sample + 1)
        else:
            truncated_sample = sample[:max_word_num_in_batch]
            input_sample = truncated_sample
            target_sample = truncated_sample
            word_count[sample_id] = len(truncated_sample)

        word_inputs[sample_id][:len(input_sample)] = input_sample
        word_targets[sample_id][:len(target_sample)] = target_sample
    return word_inputs, word_targets, word_count
