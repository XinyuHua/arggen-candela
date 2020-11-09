# Author: Xinyu Hua
# Last modified: 2020-11-04
import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.model import Candela
from option import get_inference_parser

from modules.generation import DecodingStrategy
import utils
from vocab import Vocab
from arggen_data import ArgumentGenerationDataset

def main():

    parser = get_inference_parser()
    args = parser.parse_args()

    vocab = Vocab(utils.DATA_DIR + "vocab.txt")

    test_dataset = ArgumentGenerationDataset(args=args,
                                             set_type="oracle_test.toy",
                                             vocab=vocab)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collater)

    ckpt_path = utils.find_ckpt_path(args.exp_name, args.epoch_id)
    model = Candela.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()

    decoding_strategy = DecodingStrategy(model=model, vocab=vocab, args=args)
    fout = open(f"output/{args.exp_name}_epoch={args.epoch_id}.jsonl", "w")

    test_tqdm = tqdm(enumerate(test_dataloader),
                     total=len(test_dataset) / args.batch_size)
    for batch_ix, batch in test_tqdm:
        batch = utils.move_to_cuda(batch)
        batch_size = len(batch['id'])

        with torch.no_grad():
            output, stype_results, ph_sel_results = decoding_strategy.generate(batch)

        for b in range(batch_size):
            cur_tok_ids_raw = output[b][0]
            cur_tok_ids_no_special = [item for item in cur_tok_ids_raw if item not in vocab.special_token_idx]
            cur_output_tokens_raw = vocab.decode(cur_tok_ids_raw)
            cur_output_str = " ".join(vocab.decode(cur_tok_ids_no_special))

            enc_src_len = batch['enc_src_len'][b]
            enc_src = batch['enc_src'][b][:enc_src_len]
            enc_src = vocab.decode(enc_src)

            output_obj = {
                "id": batch['id'][b],
                "op": " ".join(enc_src),
                "output_tokens": cur_output_tokens_raw,
                "output": cur_output_str,
                "sentence_types": stype_results[b],
                "phrase_selection": ph_sel_results[b],
            }
            fout.write(json.dumps(output_obj) + "\n")
    fout.close()

if __name__=='__main__':
    main()
