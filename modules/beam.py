import torch

class Beam:

    def __init__(self, beam_size, vocab, min_length, block_ngram_repeat=0):
        self.beam_size = beam_size
        self.scores = torch.zeros(beam_size, dtype=torch.float32).cuda()
        self.all_scores = []

        self.prev_ks = []

        self.pad_idx = vocab.pad_idx
        self.eos_idx = vocab.eos_idx
        self.sos_idx = vocab.sos_idx

        # whether EOS topped the beam
        self.eos_top = False

        self.next_ys = [torch.LongTensor(beam_size).fill_(self.pad_idx)]
        self.next_ys[0][0] = self.sos_idx


        self.finished = []
        self.min_length = min_length
        self.block_ngram_repeat = block_ngram_repeat


    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.next_ys[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def advance(self, probs):
        """Given model's output probabilities, generate for all beams."""
        num_words = probs.shape[1]
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(probs)):
                probs[k][self.eos_idx] = -1e20

        if len(self.prev_ks) > 0:
            beam_scores = probs + self.scores.unsqueeze(1).expand_as(probs)
            for i in range(self.next_ys[-1].shape[0]):
                if self.next_ys[-1][i] == self.eos_idx:
                    beam_scores[i] = -1e20

            if self.block_ngram_repeat > 0:
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].shape[0]):
                    hyp = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -1e20
        else:
            beam_scores = probs[0]

        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.beam_size, 0, True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores

        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        for i in range(self.next_ys[-1].shape[0]):
            if self.next_ys[-1][i] == self.eos_idx:
                self.finished.append((self.scores[i], len(self.next_ys) - 1, i))

        # when top-of-beam is EOS, end
        if self.next_ys[-1][0] == self.eos_idx:
            self.all_scores.append(self.scores)
            self.eos_stop = True


    def done(self):
        return self.eos_top and len(self.finished) >= 1

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            while len(self.finished) < minimum:
                self.finished.append((self.scores[i], len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """Walk back to construct the full hypothesis"""
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return hyp[::-1]

