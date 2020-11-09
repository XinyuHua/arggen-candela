import json
import torch
import utils
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import Counter


class ArgumentGenerationDataset(Dataset):

    def __init__(self, args, set_type, vocab):
        super().__init__()
        self.set_type = set_type
        self.encode_passage = args.encode_passage
        self.max_op_token = args.max_op_token
        self.max_passage_token = args.max_passage_token
        self.max_tgt_token = args.max_tgt_token
        self.max_tgt_sent = args.max_tgt_sent
        self.max_phrase_per_sent = args.max_phrase_per_sent
        self.max_phrase_bank_size = args.max_phrase_bank_size

        self.vocab = vocab

        self.ID = []
        self.src_op = []
        self.tgt_counterarg = []
        self.tgt_sent_id = [] # keep track of the sentence id for each token
        self.tgt_kp = []
        self.passage = []
        self.phrase_sel_inputs = []
        self.phrase_sel_indicators = []
        self.phrase_bank = []
        self.sentence_type = []

        self.load_data()
        self.size = len(self.ID)

    def __len__(self):
        return self.size


    def load_data(self):
        data_path = utils.DATA_DIR + f"{self.set_type}.jsonl"

        tid = Counter()
        for ln in tqdm(open(data_path), desc=f"Loading {self.set_type} data..."):
            cur_obj = json.loads(ln)

            cur_id = cur_obj['id']
            tid[cur_id] += 1
            long_id = (cur_id, tid[cur_id])
            self.ID.append(long_id)

            cur_src = cur_obj["op"][:self.max_op_token]
            cur_src = self.vocab.encode(cur_src)
            self.src_op.append(cur_src)

            # load target counterarg and keyphrase
            cur_sent_types = [0] # SOS as filler
            cur_tgt = [self.vocab.sos_idx]
            cur_tgt_sent_id = [0]
            selected_phrase = [[(self.vocab.sos_idx,)]]
            phz_bank = set()

            # add SOS and EOS as a special keyphrase, this is needed so that
            # during inference time the model can starts from scratch, and
            # knows when to stop.
            phz_bank.add((self.vocab.eos_idx,))
            phz_bank.add((self.vocab.sos_idx,))

            for sent_ix, sent in enumerate(cur_obj["target_counterarg"]):
                cur_tgt.extend(self.vocab.encode(sent["tokens"]))
                cur_tgt_sent_id.extend([sent_ix + 1 for _ in sent['tokens']])
                sent_phz = [ph.split() for ph in sent['selected_keyphrases']][:self.max_phrase_per_sent]
                sent_phz_id = [tuple(self.vocab.encode(ph)) for ph in sent_phz]
                selected_phrase.append(sent_phz_id)
                if len(phz_bank) < self.max_phrase_bank_size:
                    phz_bank.update(sent_phz_id)

                cur_sent_types.append(1 if sent['is_content'] else 0)

                if sent_ix == self.max_tgt_sent - 1:
                    break

            selected_phrase.append([(self.vocab.eos_idx,)])
            cur_tgt = cur_tgt[:self.max_tgt_token]
            cur_tgt.append(self.vocab.eos_idx)

            cur_tgt_sent_id = cur_tgt_sent_id[:self.max_tgt_token]
            cur_tgt_sent_id.append(cur_tgt_sent_id[-1] + 1)

            self.tgt_counterarg.append(cur_tgt)
            self.tgt_sent_id.append(cur_tgt_sent_id)
            self.phrase_sel_inputs.append(selected_phrase)

            # 0 for filler, 1 for content, 2 for EOS
            # cur_sent_types.append(2)
            self.sentence_type.append(cur_sent_types)

            # build phrase bank
            phz_bank = list(phz_bank)
            phrase_selection_indicator = [] # ids in the phrase bank
            for sent_ix, sent_ph in enumerate(selected_phrase):
                cur_sent_indicator = [0 for _ in phz_bank]
                for item in sent_ph:
                    if tuple(item) in phz_bank:
                        cur_sent_indicator[phz_bank.index(tuple(item))] = 1

                phrase_selection_indicator.append(cur_sent_indicator)
            # dummy last sentence with EOS only
            # phrase_selection_indicator.append([(self.vocab.eos_idx)])

            self.phrase_bank.append(phz_bank)
            self.phrase_sel_indicators.append(phrase_selection_indicator)

            # load passages and keyphrase
            cur_passages = []
            for item in cur_obj["target_retrieved_passages"]:
                tokens = []
                for sent in item['sentences']:
                    tokens.extend(self.vocab.encode(sent))
                tokens.append(self.vocab.sep_idx)
                cur_passages.extend(tokens)
            cur_passages = cur_passages[:self.max_passage_token]
            self.passage.append(cur_passages)


    def __getitem__(self, index):
        sample = dict(
            id=self.ID[index],
            op=self.src_op[index],
            tgt=self.tgt_counterarg[index],
            tgt_sent_ids=self.tgt_sent_id[index],
            passage=self.passage[index],
            phrase_bank=self.phrase_bank[index],
            phrase_bank_sel_ind=self.phrase_sel_indicators[index],
            phrase_bank_sel=self.phrase_sel_inputs[index],
            sentence_type=self.sentence_type[index],
        )
        return sample


    def collater(self, samples):
        batch = dict()
        batch['id'] = [s['id'] for s in samples]
        batch_size = len(batch['id'])

        # src (op + passage)
        src_tensors = []
        src_len = []
        for ix in range(batch_size):
            cur_src = samples[ix]['op']
            if self.encode_passage:
                cur_src += samples[ix]['passage']
            src_tensors.append(torch.LongTensor(cur_src))
            src_len.append(len(cur_src))
        batch['enc_src'] = utils.collate_tokens(values=src_tensors,
                                                pad_idx=self.vocab.pad_idx)
        batch['enc_src_len'] = torch.LongTensor(src_len)

        # target: dec_in and dec_out
        # sentence_types
        tgt_len = [len(s['tgt']) for s in samples]
        max_tgt_len = max(tgt_len)
        sent_nums = [len(s['sentence_type']) for s in samples]
        max_sent_nums = max(sent_nums)

        dec_inputs = []
        dec_targets = []
        dec_lens = []
        dec_sent_ids = []
        dec_mask = torch.zeros([batch_size, max_tgt_len - 1], dtype=torch.long)

        # add EOS sentence type
        sent_types = torch.full([batch_size, max_sent_nums + 1], fill_value=2, dtype=torch.long)

        for ix, item in enumerate(samples):
            cur_tgt = item['tgt']
            dec_inputs.append(torch.LongTensor(cur_tgt[:-1]))
            dec_targets.append(torch.LongTensor(cur_tgt[1:]))
            dec_lens.append(len(cur_tgt) - 1)
            dec_mask[ix][:len(cur_tgt) - 1] = 1

            dec_sent_ids.append(torch.LongTensor(item['tgt_sent_ids'][:-1]))
            sent_types[ix][:sent_nums[ix]] = torch.LongTensor(item["sentence_type"])

        batch['dec_in'] = utils.collate_tokens(values=dec_inputs,
                                               pad_idx=self.vocab.pad_idx)
        batch['dec_out'] = utils.collate_tokens(values=dec_targets,
                                                pad_idx=self.vocab.pad_idx)
        batch['dec_in_len'] = torch.LongTensor(dec_lens)
        batch['dec_mask'] = dec_mask
        batch['dec_sent_id'] = utils.collate_tokens(values=dec_sent_ids,
                                                    pad_idx=0)
        batch["sent_types"] = sent_types


        # pad phrase bank, for each sample the phrase bank is a 2D list
        # the result would be a 3D list
        phrase_banks = [s['phrase_bank'] for s in samples]
        phrase_sizes = [[len(ph) for ph in s] for s in phrase_banks]

        max_ph_num = max([len(x) for x in phrase_sizes])
        max_ph_len = max([max([len(p) for p in bank]) for bank in phrase_banks])
        phrase_bank_tensor = torch.zeros([batch_size, max_ph_num, max_ph_len], dtype=torch.long)
        for ix in range(batch_size):
            cur_ph_bank = phrase_banks[ix]
            for j, ph in enumerate(cur_ph_bank):
                phrase_bank_tensor[ix][j][:len(ph)] = torch.LongTensor(list(ph))
        batch['ph_bank_tensor'] = phrase_bank_tensor
        batch['ph_bank_len_tensor'] = torch.LongTensor([len(x) for x in phrase_banks])

        # create padded tensor for phrase selection indicators
        # sample[`phrase_bank_sel`] is a 3D list [sample_id, sent_id, phrase_id]
        phrase_sel = [s['phrase_bank_sel_ind'] for s in samples]
        sent_num = [len(x) for x in phrase_sel]
        phrase_sel_ind_tensor = torch.zeros([batch_size, max(sent_num), max_ph_num],
                                        dtype=torch.long)
        for ix in range(batch_size):
            cur_sel = phrase_sel[ix]
            for sent_ix, sent_sel in enumerate(cur_sel):
                phrase_sel_ind_tensor[ix, sent_ix, : len(sent_sel)] = torch.LongTensor(sent_sel)
        batch["ph_sel_ind_tensor"] = phrase_sel_ind_tensor

        # 3d list, batch_size x sent_num x ph_num x ph_len
        phrase_sel = [s['phrase_bank_sel'] for s in samples]
        max_ph_len = 0
        max_ph_per_sent = 0
        for sample in phrase_sel:
            ph_lens = [[len(ph) for ph in sent] for sent in sample]
            max_ph_per_sent = max(max_ph_per_sent, max([len(item) for item in ph_lens]))
            max_ph_len = max(max_ph_len,
                             max([max(item) if len(item) > 0 else 0 for item in ph_lens]))

        phrase_sel_tensor = torch.zeros([batch_size, max(sent_num), max_ph_per_sent, max_ph_len],
                                        dtype=torch.long)
        for six, sample in enumerate(phrase_sel):
            for sent_ix, sent in enumerate(sample):
                for ph_ix, ph in enumerate(sent):
                    phrase_sel_tensor[six, sent_ix, ph_ix, :len(ph)] = torch.LongTensor(ph)
        batch['ph_sel_tensor'] = phrase_sel_tensor

        return batch