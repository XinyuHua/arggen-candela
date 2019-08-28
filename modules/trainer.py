# Author: Xinyu Hua
# Last modified: 2019-3-26
""" training and validation """
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import utils


def prepare_batch(data_dict, opt):
    """
    pad each list data in data_dict and convert into tensors
    """
    tensor_dict = {}
    src_input_array, _, src_input_lens = utils.pad_text_id_list_into_array(
        batch_text_lists=data_dict["src_seqs"],
        max_len=opt.max_op_words,
        add_start=False)

    tensor_dict["src_input_tensor"] = torch.tensor(src_input_array, dtype=torch.long).cuda()

    tensor_dict["src_len_tensor"] = torch.tensor(src_input_lens, dtype=torch.long).cuda()

    ph_bank_array, ph_bank_len = utils.pad_phrase_bank_into_tensor(
        data_dict["ph_bank_seqs"], opt.max_kp_bank_size)
    ph_bank_max_len = int(max(ph_bank_len))

    tensor_dict["ph_bank_len_array"] = ph_bank_len

    tensor_dict["ph_bank_tensor"] = \
        torch.tensor(ph_bank_array, dtype=torch.long).cuda()

    tensor_dict["ph_bank_len_tensor"] = \
        torch.tensor(ph_bank_len, dtype=torch.long).cuda()

    ph_sel_array, ph_sel_len = utils.pad_phrase_list_into_tensor(
        data_dict["ph_sel_seqs"],
        MAX_SENT_NUM=opt.max_sent_num,
        MAX_PH_NUM=opt.max_kp_num)
    ph_sel_max_len = int(max(ph_sel_len))

    tensor_dict["ph_sel_tensor"] = torch.tensor( \
        ph_sel_array, dtype=torch.long).cuda()

    tensor_dict["ph_sel_len_tensor"] = torch.tensor( \
        ph_sel_len, dtype=torch.long).cuda()
    tensor_dict["ph_sel_len_array"] = ph_sel_len

    ph_bank_sel_ind_array = utils.pad_ph_bank_sel_ind_list( \
        data_dict["ph_bank_sel_seqs"],
        ph_sel_max_len,
        ph_bank_max_len)
    tensor_dict["ph_bank_sel_ind_tensor"] = torch.tensor(
        ph_bank_sel_ind_array, dtype=torch.long).cuda()

    tensor_dict["stype_tensor"] = torch.tensor(
        utils.pad_stype_into_tensor(data_dict["stype_seqs"],
                                    MAX_SENT_NUM=opt.max_sent_num),
        dtype=torch.long).cuda()

    tensor_dict["stype_len_tensor"] = torch.tensor( \
        data_dict["stype_len_seqs"], dtype=torch.long).cuda()

    rr_inputs_array, rr_targets_array, rr_len_array = \
        utils.pad_text_id_list_into_array(data_dict["rr_seqs"],
                                          opt.max_rr_words)

    tensor_dict["rr_inputs_tensor"] = torch.tensor( \
        rr_inputs_array, dtype=torch.long).cuda()
    tensor_dict["rr_targets_tensor"] = torch.tensor( \
        rr_targets_array, dtype=torch.long).cuda()

    tensor_dict["rr_len_tensor"] = torch.tensor( \
        rr_len_array, dtype=torch.long).cuda()

    _, rr_sent_id_array, _ = utils.pad_text_id_list_into_array(
        data_dict["rr_sent_ids_seqs"], MAX_LEN=opt.max_rr_words,
        add_start=False, EOS=0)

    # CAUTION: because we have a limit for max number of sentences and
    # max number of words in the root reply. These two limits can cause
    # troubles, when there are more sentences kept in the words array
    # than the sentence array, then the model won't know what to take
    # from the sentence planner decoder. The hack here is to always
    # take the 0-th sentence's states, therefore we need to convert all
    # sentence ids that are larger than the limit into 0s
    for ln_id, item in enumerate(rr_sent_id_array):

        for sid in range(len(item)):
            if item[sid] >= opt.max_sent_num:
                rr_sent_id_array[ln_id][sid] = opt.max_sent_num - 1

    tensor_dict["rr_sent_id_tensor"] = torch.tensor( \
        rr_sent_id_array, dtype=torch.long).cuda()
    rr_mask_array = np.zeros((opt.batch_size, rr_sent_id_array.shape[1]))
    for ln_id, item in enumerate(rr_len_array):
        rr_mask_array[ln_id][:int(item)] = 1
    rr_mask_tensor = torch.tensor(rr_mask_array, dtype=torch.long)
    tensor_dict["rr_mask_tensor"] = rr_mask_tensor.unsqueeze(-1).cuda()
    return tensor_dict


def valid_epoch(model, src_inputs, src_lens, ph_sel_inputs, ph_bank_sel_ind,
                ph_bank, sent_type_target, sent_type_target_len, rr_inputs,
                rr_sent_ids, rr_slen, opt, n_epoch, id2word, fout_log):
    n_iters = len(src_inputs) // opt.batch_size
    print_every = n_iters // 1
    correct_cnt = 0
    total_pred = 0
    total_loss = 0
    total_attn_loss = 0
    total_ppl = 0

    for it in range(n_iters):
        cur_batch_ids = [ix for ix in \
                         range(it * opt.batch_size, (it + 1) * opt.batch_size)]
        cur_src_lens = [(ix, len(src_inputs[ix])) for ix in cur_batch_ids]
        ix_sorted_with_src_lens = sorted(cur_src_lens, key=lambda x: x[1], reverse=True)

        src_seqs = []
        src_len_seqs = []
        ph_bank_seqs = []
        ph_sel_seqs = []
        ph_bank_sel_ind_seqs = []
        stype_seqs = []
        stype_len_seqs = []
        rr_seqs = []
        rr_sent_ids_seqs = []
        rr_len_seqs = []

        for ix, _ in ix_sorted_with_src_lens:
            src_seqs.append(src_inputs[ix])
            src_len_seqs.append(src_lens[ix])
            ph_sel_seqs.append(ph_sel_inputs[ix])
            ph_bank_seqs.append(ph_bank[ix])
            ph_bank_sel_ind_seqs.append(ph_bank_sel_ind[ix])
            stype_seqs.append(sent_type_target[ix])
            stype_len_seqs.append(sent_type_target_len[ix])
            rr_seqs.append(rr_inputs[ix])
            rr_sent_ids_seqs.append(rr_sent_ids[ix])
            rr_len_seqs.append(rr_slen[ix])

        data_dict = {"src_seqs": src_seqs, "src_len_seqs": src_len_seqs,
                     "ph_bank_seqs": ph_bank_seqs, "ph_sel_seqs": ph_sel_seqs,
                     "ph_bank_sel_seqs": ph_bank_sel_ind_seqs,
                     "stype_seqs": stype_seqs,
                     "stype_len_seqs": stype_len_seqs,
                     "rr_seqs": rr_seqs,
                     "rr_sent_ids_seqs": rr_sent_ids_seqs,
                     "rr_len_seqs": rr_len_seqs}

        tensor_data_dict = prepare_batch(data_dict, opt)

        with torch.no_grad():

            st_readouts, wd_readouts, sp_attn, sp_attn_logits = model(tensor_data_dict)

            mask = utils.get_matrix_mask(tensor_data_dict["ph_bank_sel_ind_tensor"].size(),
                                         tensor_data_dict["ph_sel_len_array"],
                                         tensor_data_dict["ph_bank_len_array"])
            mask = torch.tensor(mask).cuda()

            wd_loss, st_loss, attn_loss = model.compute_losses(st_readouts, wd_readouts,
                                                               tensor_data_dict["stype_tensor"],
                                                               tensor_data_dict["rr_targets_tensor"],
                                                               sp_attn,
                                                               tensor_data_dict["ph_bank_sel_ind_tensor"],
                                                               mask,
                                                               tensor_data_dict["stype_len_tensor"],
                                                               tensor_data_dict["rr_len_tensor"])
            avg_ppl = model.compute_ppl(wd_readouts,
                                        tensor_data_dict["rr_targets_tensor"],
                                        tensor_data_dict["rr_len_tensor"])

        if it % print_every == 0:
            model_probs = F.softmax(wd_readouts, dim=-1)
            model_pred = torch.argmax(model_probs, dim=-1)
            fout_log.write("EPOCH: %d ITER: %d\n" % (n_epoch, it))
            for s_ix, sample in enumerate(rr_seqs):
                cur_tgt_seq = [id2word[wid] for wid in sample]
                cur_pred_seq = [id2word[wid] for wid in model_pred[s_ix]]
                fout_log.write("GOLD STANDARD:")
                fout_log.write(" ".join(cur_tgt_seq))
                fout_log.write("\n")
                fout_log.write("PREDICTION:")
                fout_log.write(" ".join(cur_pred_seq))
                fout_log.write("\n" + "-" * 50 + "\n")

        total_loss += wd_loss
        total_attn_loss += attn_loss
        total_ppl += avg_ppl

        stype_pred_probs = F.softmax(st_readouts, dim=-1)
        stype_pred = torch.argmax(stype_pred_probs, dim=-1)

        # compute sentence type prediction accuracy
        correctness = stype_pred == tensor_data_dict["stype_tensor"]
        mask = utils.get_sequence_mask_from_length(
            tensor_data_dict["ph_sel_len_tensor"], stype_pred.size(1))
        correct_cnt += (correctness * mask).sum().data.cpu().numpy()
        total_pred += tensor_data_dict["ph_sel_len_tensor"].sum().data.cpu().numpy()
    avg_loss = total_loss / n_iters
    avg_attn_loss = total_attn_loss / n_iters
    acc = correct_cnt / total_pred
    print("validation finished, avg XENT loss: %.3f\tsentence type accuracy: %.3f" % (avg_loss, acc))
    print("Attention loss: %.3f" % avg_attn_loss)
    ppl = total_ppl / n_iters
    return avg_loss, acc, ppl


def train_epoch(model, n_iters, src_inputs, src_lens, ph_sel_inputs,
                ph_bank_sel_indicator, ph_bank, sent_type_target, sent_type_target_len,
                rr_inputs, rr_sent_ids, rr_slen, optimizer, opt, n_epoch):
    """
    train an epoch with minibatching
    Args:
        model: model to train
        n_iters: number of iterations per epoch
        src_inputs:
        src_lens:
        ph_sel_inputs:
        ph_bank_sel_indicator
        ph_bank
        sent_type_target
        sent_type_target_len
        rr_inputs
        rr_sent_ids
        rr_slen
        optimizer
        opt
        epoch
    Returns:
        avg_loss: average cross entropy loss over the batch
    """

    start_time = time.time()
    sequence_order = random.sample(range(len(src_inputs)), len(src_inputs))

    print_loss = 0
    print_attn_loss = 0
    print_every = n_iters // 3
    total_loss = 0
    save_iters = n_iters // opt.save_freq
    saved_ckpts = 0

    for it in range(n_iters):

        src_seqs = []
        src_len_seqs = []
        ph_bank_seqs = []
        ph_sel_seqs = []
        ph_bank_sel_seqs = []
        stype_seqs = []
        stype_len_seqs = []
        rr_seqs = []
        rr_sent_ids_seqs = []
        rr_len_seqs = []

        sampled_ids = []
        for sample_id in range(opt.batch_size):
            try:
                idx = sequence_order.pop()
            except IndexError:
                idx = random.randint(0, len(src_inputs))
            sampled_ids.append((idx, len(src_inputs[idx])))
        sorted_ids = sorted(sampled_ids, key=lambda x: x[1], reverse=True)

        for item in sorted_ids:
            idx = item[0]
            src_seqs.append(src_inputs[idx])
            src_len_seqs.append(src_lens[idx])
            ph_bank_seqs.append(ph_bank[idx])
            ph_sel_seqs.append(ph_sel_inputs[idx])
            ph_bank_sel_seqs.append(ph_bank_sel_indicator[idx])
            stype_seqs.append(sent_type_target[idx])
            stype_len_seqs.append(sent_type_target_len[idx])
            rr_seqs.append(rr_inputs[idx])
            rr_sent_ids_seqs.append(rr_sent_ids[idx])
            rr_len_seqs.append(rr_slen[idx])

        data_dict = {"src_seqs": src_seqs, "src_len_seqs": src_len_seqs,
                     "ph_bank_seqs": ph_bank_seqs, "ph_sel_seqs": ph_sel_seqs,
                     "ph_bank_sel_seqs": ph_bank_sel_seqs,
                     "stype_seqs": stype_seqs,
                     "stype_len_seqs": stype_len_seqs,
                     "rr_seqs": rr_seqs,
                     "rr_sent_ids_seqs": rr_sent_ids_seqs,
                     "rr_len_seqs": rr_len_seqs}

        tensor_data_dict = prepare_batch(data_dict, opt)
        model.zero_grad()

        st_readout, wd_readout, sp_attn, sp_attn_logits = model(tensor_data_dict)
        mask = utils.get_matrix_mask(tensor_data_dict["ph_bank_sel_ind_tensor"].size(),
                                     tensor_data_dict["ph_sel_len_array"],
                                     tensor_data_dict["ph_bank_len_array"])
        mask = torch.tensor(mask).cuda()

        wd_loss, st_loss, attn_loss = model.compute_losses(st_readout,
                                                           wd_readout,
                                                           tensor_data_dict["stype_tensor"],
                                                           tensor_data_dict["rr_targets_tensor"],
                                                           sp_attn,
                                                           tensor_data_dict["ph_bank_sel_ind_tensor"],
                                                           mask,
                                                           tensor_data_dict["stype_len_tensor"],
                                                           tensor_data_dict["rr_len_tensor"])
        model_loss = wd_loss + st_loss + attn_loss
        print_loss += wd_loss
        total_loss += wd_loss
        print_attn_loss += attn_loss

        if it % print_every == 0 and it > 0:
            print("%d-th iter finished (%d in total), loss on word dec: %.4f" \
                  % (it, n_iters, print_loss / print_every))
            print("average attn loss: %.4f" % (print_attn_loss / print_every))
            print_loss = 0
            print_attn_loss = 0

        if it % save_iters == 0 and it > 0 and saved_ckpts < opt.save_freq - 1:
            n_th_ckpt = it // save_iters
            print("  dropping %d-th checkpoints in %d-th epoch..." \
                  % (n_th_ckpt, n_epoch))
            ckpt_path = opt.ckpt_path + opt.exp_name + "/"
            ckpt_name = ckpt_path + "epoch_%d-%d_train_%.3f.tar" \
                        % (n_epoch, n_th_ckpt, total_loss / it)

            torch.save({
                "encoder_state_dict": model.enc.state_dict(),
                "wd_decoder_state_dict": model.wd_dec.state_dict(),
                "sp_decoder_state_dict": model.sp_dec.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": n_epoch},
                ckpt_name)

        model_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

    return total_loss / n_iters

