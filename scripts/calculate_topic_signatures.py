"""Identify topic signature words in a document using log-likelihood ratio 
test. The original description can be found in:

@inproceedings{lin-hovy-2000-automated,
title = "The Automated Acquisition of Topic Signatures for Text Summarization",
author = "Lin, Chin-Yew  and
  Hovy, Eduard",
booktitle = "{COLING} 2000 Volume 1: The 18th International Conference on Computational Linguistics",
year = "2000",
url = "https://www.aclweb.org/anthology/C00-1072",
}

"""

import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm

def compute_likelihood_ratio(c_1, c_12, c_2, p, p_1, p_2, N):
    """
    """
    def log_L(k, n, x):
        """
        """
        return k * np.log(x) + (n-k) * np.log(1 - x)

    return log_L(c_12, c_1, p) \
            + log_L(c_2 - c_12, N - c_1, p) \
            - log_L(c_12, c_1, p_1) \
            - log_L(c_2 - c_12, N - c_1, p_2)



class TopicSignatureConstruction(object):

    def __init__(self, lemma_data_path, output_path):
        self.data_path = lemma_data_path
        self.output_path = output_path

        self.lemma_data = []
        self.doc2freq = dict()
        self.doc_total_words = dict()
        self.total_freq = defaultdict(int)
        self.total_words = 0

        self.stopwords = [w.lower() for w in open('dat/stopwords.txt')]


    def load_data(self):
        for ln in open(self.data_path):
            cur_obj = json.loads(ln)
            cur_id = cur_obj['id']
            cur_lemma = cur_obj['lemma']

            for word in cur_lemma:
                if word.lower() in self.stopwords:
                    continue

                self.total_freq[word] += 1
                self.total_words += 1

                if cur_id not in self.doc2freq:
                    self.doc2freq[cur_id] = defaultdict(int)
                self.doc2freq[cur_id][word] += 1

                if cur_id not in self.doc_total_words:
                    self.doc_total_words[cur_id] = 0
                self.doc_total_words[cur_id] += 1

        print(f'{len(self.doc_total_words)} documents loaded')


    def calculate_llr(self):
        """Calculate log-likelihood ratio"""
        self.doc_word2ratio = {doc_id: defaultdict(float) \
                                       for doc_id in self.doc2freq}

        for word in tqdm(self.total_freq):
            if self.total_freq[word] < 10:
                continue

            c_2 = self.total_freq[word]
            N = self.total_words
            p = c_2 / N
            for doc_id in self.doc2freq:
                c_12 = self.doc2freq[doc_id][word]
                if c_12 == 0:
                    continue

                c_1 = self.doc_total_words[doc_id]
                p_1 = c_12 / c_1
                p_2 = (c_2 - c_12) / (N - c_1)
                if c_2 == c_12:
                    cur_ratio = 0
                else:
                    cur_ratio = -2 * compute_likelihood_ratio(c_1, c_12, c_2, p, p_1, p_2, N=N)
                self.doc_word2ratio[doc_id][word] = cur_ratio



    def write_to_disk(self):
        fout = open(self.output_path, 'w')
        for doc_id, w2ratio in self.doc_word2ratio.items():
            ret_obj = {'id': doc_id, 'ratio_ranked_words': []}
            for item in sorted(w2ratio.items(), key=lambda x: x[1], reverse=True):
                output_tuple = (item[0], item[1], self.doc2freq[doc_id][item[0]])
                ret_obj['ratio_ranked_words'].append(output_tuple)
            fout.write(json.dumps(ret_obj) + '\n')
        fout.close()


if __name__=='__main__':
    lemma_path = 'dat/cmv_op_lemma.jsonl'
    output_path = 'dat/cmv_op_llr.jsonl'

    ts_construction = TopicSignatureConstruction(lemma_data_path=lemma_path,
                                                 output_path=output_path)
    ts_construction.load_data()
    ts_construction.calculate_llr()
    ts_construction.write_to_disk()

