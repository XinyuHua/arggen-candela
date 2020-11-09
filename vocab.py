
class Vocab:

    def __init__(self, vocab_path):
        self.word2id = dict()
        self.id2word = list()

        for ln in open(vocab_path):
            wid, word, freq = ln.strip().split("\t")
            self.word2id[word] = int(wid)
            self.id2word.append(word)

            if len(self.id2word) == 50000:
                break

        self.size = len(self.id2word)
        self.pad_idx = 0
        self.sos_idx = 1
        self.sep_idx = 2
        self.eos_idx = 3
        self.unk_idx = 4
        self.special_token_idx = [0, 1, 2, 3, 4]

    def __len__(self):
        return self.size

    def get_word(self, idx):
        if idx > 0 and idx < self.size:
            return self.id2word[idx]
        else:
            raise IndexError("{} index invalid!".format(idx))

    def get_idx(self, word):
        return self.word2id[word] if word in self.word2id else self.unk_idx

    def encode(self, word_list):
        return [self.get_idx(w) for w in word_list]

    def decode(self, idx_list):
        return [self.get_word(idx) for idx in idx_list if idx > 0]