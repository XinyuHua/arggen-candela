# Author: Xinyu Hua
"""Program entry for training and test"""


import os
import sys
import glob
import argparse
import time
import random
import logging
import utils
import torch
import torch.nn as nn
from torch import optim

from modules import encoder
from modules import decoder
from modules.model import HierarchicalSeq2seq
from modules.trainer import trainEpoch, validation

parser = argparse.ArgumentParser(description="main.py")
parser.add_argument("--batch_size", type=int, default=32, help="batch size for training and validation")
parser.add_argument("--load_full", action="store_true", help="whether to load the full CMV politics dataset or only 20 sample threads")
parser.add_argument("--n_epochs", type=int, default=20, help="total epochs for training")
parser.add_argument("--exp_name", type=str, default="demo", help="name for experiment directory")
parser.add_argument("--learning_rate", type=float, default=0.15, help="initial learning rate for optimizer")
parser.add_argument("--init_accum", type=float, default=0.1, help="initial accum")
parser.add_argument("--hidden_size", type=int, default=512, help="dimension of RNN hidden states")
parser.add_argument("--attn_size", type=int, default=512, help="dimension of attention")
parser.add_argument("--pret_encoder_path", type=str, default=utils.PRET_PATH + "op-enc/epoch_2-2_train_loss_3.981.tar", help="name for experiment directory")
parser.add_argument("--pret_decoder_path", type=str, default=utils.PRET_PATH + "psg-dec_no_type_keep_on_rr/epoch_9_train_loss_3.915_valid_ppl_56.531.tar", help="path to pretrained decoder")

parser.add_argument("--ph_emb_size", type=int, default=100, help="dimension of word embedding")
parser.add_argument("--word_emb_size", type=int, default=300, help="dimension of word embedding")
parser.add_argument("--max_op_words", type=int, default=200, help="maximum allowed op length, if an OP is longer, it will be truncated to this many words.")
parser.add_argument("--max_rr_words", type=int, default=50, help="maximum allowed RR length, if a RR is longer, it will be truncated to this many words.")
parser.add_argument("--max_sent_num", type=int, default=5, help="maximum number of sentences in RR to be considered")
parser.add_argument("--max_ph_num", type=int, default=10, help="maximum number of phrases in each RR sentence")
parser.add_argument("--max_ph_bank_size", type=int, default=20, help="maximum number of phrases in each RR paragraph")
parser.add_argument("--save_freq", type=int, default=1, help="the number of checkpoints to drop in each epoch")

opt = parser.parse_args()

def run_training(model, data, optimizer, word2id, id2word, opt):
    """
    run training program: check if checkpoints exist in the specified
    experiment path, if so directly load the checkpoint and continue training,
    otherwise start from scratch
    """

    done_epoch = 0
    ckpt_path = opt.ckpt_path + opt.exp_name + "/"
    ckpt_name_base = "ckpt-"
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_name_lst = glob.glob(ckpt_path + "epoch*")

    if len(ckpt_name_lst) > 0: # continue training from saved checkpoint
        ckpt_lst_sorted = sorted(ckpt_name_lst, key=lambda x:\
                eval(os.path.basename(x).split("_")[1]),\
                reverse=True)
        ckpt_loaded = torch.load(ckpt_lst_sorted[0])
        done_epoch = ckpt_loaded["epoch"]
        print("start training from epoch %d" % done_epoch)
        print(ckpt_path)
        model.enc.load_state_dict(ckpt_loaded["encoder_state_dict"])
        model.sp_dec.load_state_dict(ckpt_loaded["sp_decoder_state_dict"])
        model.wd_dec.load_state_dict(ckpt_loaded["wd_decoder_state_dict"])
        optimizer.load_state_dict(ckpt_loaded["optimizer"])

    else:
        print("start training from scratch...")
        print("loading pre-trained weigths...")
        pret_enc = torch.load(opt.pret_encoder_path)["model_state_dict"]
        for name in model.enc.state_dict().keys():
            if not "l0" in name:continue
            model.enc.state_dict()[name].copy_(pret_enc[name.lower()])

        pret_dec = torch.load(opt.pret_decoder_path)["model_state_dict"]
        for name in model.wd_dec.state_dict().keys():
            if "l0" in name or "readout" in name:
                model.wd_dec.state_dict()[name].copy_(pret_dec[name.lower()])
        print("pre-trained layers loaded")


    def collect_all_data_components(raw_data):
        data_dict = dict()
        if opt.encode_passage:
            data_dict["src_inputs"], data_dict["src_lens"] = utils.encoding_concat_text_to_id_lists(
                raw_data["src"]["op"],
                raw_data["src"]["rr_psg"],
                word2id,
                max_length_op=opt.max_op_words,
                max_length_psg=opt.max_psg_words,
            )

        else:
            data_dict["src_inputs"], data_dict["src_lens"] = utils.encoding_text_to_id_lists(
                raw_data["src"]["op"],
                word2id,
                max_length_op=opt.max_op_words,
            )

        data_dict["ph_sel_inputs"], data_dict["ph_bank"], data_dict["ph_bank_sel_ind"] = \
            utils.encode_ph_sel_to_word_ids(sample_list=raw_data["tgt"]["rr_psg_kp_sel"],
                                            word2id=word2id,
                                            max_kp_num=opt.max_kp_num,
                                            max_sent_num=opt.max_sent_num,
                                            max_bank_size=opt.max_bank_size)
        data_dict["rr_word_ids"], data_dict["rr_sent_ids"], data_dict["sent_type"], \
            data_dict["sent_len"], data_dict["sent_num"] = utils.encode_sentence_and_type_to_list(
                sample_list=raw_data["tgt"]["rr"],
                sentence_type_list=raw_data["tgt"]["rr_stype"],
                word2id=word2id,
                max_sent_num=opt.max_sent_num,
                max_word_cnt=opt.max_rr_words,
        )

        return data_dict

    train_data_dict = collect_all_data_components(data["train"])
    val_data_dict = collect_all_data_components(data["dev"])


    steps_per_epoch = len(src_inputs) // opt.batch_size
    fout_log = open("log/%s.txt" % opt.exp_name,'w')

    for n_epoch in range(1, opt.n_epochs + 1):

        logging.info("starting Epoch %d" % (n_epoch + done_epoch))
        model.train()

        avg_tr_loss = train_epoch(model, train_data_dict, opt, optimizer, n_epoch + done_epoch)

        with torch.no_grad():
            model.eval()
            avg_val_loss, cs_acc, st_acc = valid_epoch(model, val_data_dict, opt, n_epoch + done_epoch, id2word)

        ckpt_name = ckpt_path + "epoch_%d_train_%.3f_val_%.3f_st_acc_%.3f.tar"\
                % (n_epoch + done_epoch, avg_tr_loss["total"], avg_val_loss["total"], st_acc)

        torch.save({
            "embedding": model.word_emb.state_dict(),
            "encoder": model.enc.state_dict(),
            "wd_decoder_state_dict": model.wd_dec.state_dict(),
            "sp_decoder_state_dict": model.sp_dec.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": done_epoch + n_epoch},
            ckpt_name)
    fout_log.close()

    return



def run_inference(model, data, word2id, id2word, opt):
    if opt.model == "no_psg":
        src_inputs, src_lens = utils.encoding_text_to_id_lists(
                data["src"]["op"],
                word2id,
                max_length_op=opt.max_op_words)

    else:
        src_inputs, src_lens = utils.encoding_concat_text_to_id_lists(
                data["src"]["op"],
                data["src"]["rr_psg"],
                word2id,
                max_length_op=opt.max_op_words,
                max_length_rr=opt.max_rr_words)

    if opt.eval_setup == "oracle":
        ph_sel_inputs, ph_bank, ph_bank_sel_indicator = utils.encoding_ph_sel_to_word_ids(
                data["src"]["rr_psg_kp"],
                word2id,
                max_sent_num=opt.max_sent_num,
                max_kp_num=opt.max_kp_num)

        sent_type_target, sent_type_target_len = utils.encoding_sent_type_to_list(
            data["tgt"]["rr_stype"],
            max_sent_num=opt.max_sent_num)

        rr_inputs, rr_sent_ids, rr_slen = utils.encoding_output_to_list(
            data["tgt"]["rr"],
            word2id,
            max_word_cnt=opt.max_rr_words)

    elif opt.eval_setup == "system":
        _, ph_bank, _ = utils.encoding_ph_sel_to_word_ids(
                data["src"]["passage_kp"],
                word2id,
                max_sent_num=opt.max_sent_num,
                max_kp_num=opt.max_kp_num)


    ckpt_path = opt.ckpt_path + opt.exp_name + "/"
    ckpt_to_load = ckpt_path + opt.model_file
    ckpt_id = opt.model_file.split("_")[1]
    ckpt_loaded = torch.load(ckpt_to_load)
    model.enc.load_state_dict(ckpt_loaded["encoder_state_dict"])
    model.sp_dec.load_state_dict(ckpt_loaded["sp_decoder_state_dict"])
    model.wd_dec.load_state_dict(ckpt_loaded["wd_decoder_state_dict"])

    fout = open("infer_out/%s_epoch_%s_%s_%d.jsonlist"\
            % (opt.exp_name, ckpt_id, opt.eval_setup, opt.chunk_id), 'w')

    n_iters = len(src_inputs) // opt.batch_size
    t0 = time.time()

    for it in range(n_iters + 1):
        if it % 10 == 0:
            print("%d interations finished, time spent: %.4f secs" \
                % (it, time.time() - t0))

        if it == n_iters:
            cur_batch_ids = [ix for ix in range(it * opt.batch_size, len(src_inputs))]
            opt.batch_size = len(cur_batch_ids)
            if opt.batch_size == 0:
                break

        else:
            cur_batch_ids = [ix for ix in range(it * opt.batch_size, (it + 1) * opt.batch_size)]

        src_lens = [(ix, len(src_inputs[ix])) for ix in cur_batch_ids]
        ix_sorted_with_src_lens = sorted(src_lens, key=lambda x:x[1], reverse=True)

        src_seqs = []
        ph_bank_seqs = []
        tid_seqs = []
        stype_tgt_seqs = []
        arg_tgt_seqs = []

        for ix, _ in ix_sorted_with_src_lens:
            src_seqs.append(src_inputs[ix])
            tid_seqs.append(data["src"]["tid"][ix])
            ph_bank_seqs.append(ph_bank[ix])

            if opt.eval_setup == "oracle":
                arg_tgt_seqs.append(rr_inputs[ix])
                stype_tgt_seqs.append(sent_type_target)
            else:
                arg_tgt_seqs.append([])
                stype_tgt_seqs.append([])

        src_input_array, src_input_len = utils.pad_text_id_list_into_array_test(src_seqs)
        src_input_tensor = torch.tensor(src_input_array, dtype=torch.long).cuda()
        src_len_tensor = torch.tensor(src_input_len, dtype=torch.long).cuda()

        ph_bank_array, ph_bank_len = utils.pad_phrase_bank_into_tensor(\
                ph_bank_seqs, opt.max_kp_bank_size)

        ph_bank_tensor = torch.tensor(ph_bank_array, dtype=torch.long).cuda()
        ph_bank_len_tensor = torch.tensor(ph_bank_len, dtype=torch.long).cuda()

        with torch.no_grad():
            results = infer_batch(model, src_input_tensor, src_len_tensor,
                    ph_bank_tensor, ph_bank_len_tensor, id2word, opt)

        write_to_file(results, arg_tgt_seqs, src_seqs, fout, id2word, tid_seqs)


    print("beam search finished. time elapsed: %.2f secs" % (time.time() - t0))
    return



def main():
    opt.load_full = not opt.debug
    word2id, id2word = utils.load_vocab()
    glove_emb = utils.load_glove_emb(word2id)

    word_emb = nn.Embedding.from_pretrained( \
        torch.tensor(glove_emb, dtype=torch.float))

    model = HierarchicalSeq2seq(word_emb=word_emb,
                                word_emb_dim=300,
                                word_vocab_size=len(id2word)).cuda()

    if opt.mode == "train":

        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, \
                                         model.parameters()),
                                  lr=opt.learning_rate,
                                  initial_accumulator_value=opt.init_accum)

        train_data = utils.load_train_data(demo=opt.demo)

        run_training(model, train_data, optimizer, word2id, id2word, opt)

    elif opt.mode == "inference":

        test_data = utils.load_test_data(setup=opt.setup, demo=opt.demo)
        run_inference(model, test_data, word2id, id2word, opt)




if __name__=="__main__":
    main()