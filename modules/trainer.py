import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from tqdm import tqdm


def compute_perplexity(token_logits, token_targets, pad_token_id):
    """calculate PPL as exp(probs_normalized)."""
    vocab_size = token_logits.shape[-1]
    log_probs_flat = F.log_softmax(token_logits.view(-1, vocab_size), dim=-1)

    target_flat = token_targets.view(-1, 1)
    losses_flat = - torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*token_targets.size())

    mask = token_targets.eq(pad_token_id)
    seq_len = token_targets.shape[1] -  mask.sum(1)
    losses.masked_fill_(mask, 0)

    losses_sum = losses.sum(dim=1)
    probs_normalized = losses_sum / seq_len.float()
    probs_normalized = torch.min(probs_normalized, 100 * torch.ones(probs_normalized.size(), dtype=torch.float).cuda())
    ppl_lst = torch.exp(probs_normalized)
    return torch.mean(ppl_lst)


def compute_losses(
        token_logits,
        token_targets,
        pad_token_id,
        sentence_type_logits=None,
        sentence_type_targets=None,
        ph_bank_attn=None,
        ph_bank_len=None,
        ph_bank_sel_ind_targets=None,
    ):
    """Calcuate three types of losses:

    1. token level cross-entropy:
    2. sentence type cross-entropy (optional):
    3. phrase selection binary cross-entropy (optional):

    Args:
        token_logits: (batch_size x max_tgt_len x vocab_size)
        token_targets: (batch_size x max_tgt_len)
        sentence_type_logits: (batch_size x max_sent_num x 3)
        sentence_type_targets: (batch_size x max_sent_num)
        ph_bank_attn: (batch_size x max_sent_num x max_ph_bank_size)
        ph_bank_len: (batch_size)
        ph_bank_sel_ind_targets: (batch_size x max_sent_num)
    """
    token_ce_loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    losses = {"token_loss": token_ce_loss_fct(
                    token_logits.view(-1, token_logits.shape[-1]),
                    token_targets.view(-1)),
              "token_ppl": compute_perplexity(token_logits, token_targets, pad_token_id),
    }

    if sentence_type_logits is not None and sentence_type_targets is not None:
        stype_ce_loss_fct = nn.CrossEntropyLoss(ignore_index=2)
        losses["sentence_type_loss"] = stype_ce_loss_fct(
            sentence_type_logits.view(-1, sentence_type_logits.shape[-1]),
            sentence_type_targets.view(-1),
        )

    if ph_bank_attn is not None and ph_bank_sel_ind_targets is not None:
        loss_all = F.binary_cross_entropy(input=ph_bank_attn, target=ph_bank_sel_ind_targets.float(),
                                          reduction="none")
        # create phrase bank mask, this is necessary because different samples
        # can have different phrase bank size
        batch_size = ph_bank_attn.shape[0]
        ph_bank_size = ph_bank_attn.shape[-1]
        ph_bank_mask = torch.zeros([batch_size, 1, ph_bank_size]).to(loss_all.device)
        for ix, val in enumerate(ph_bank_len):
            ph_bank_mask[ix, 0, :val] = 1.0

        masked_loss = loss_all * ph_bank_mask.float()
        losses["phrase_selection_loss"] = masked_loss.sum() / ph_bank_mask.sum()

    return losses


def train_epoch(model, train_dataloader, args, optimizer, vocab, tb_logger):
    """train an epoch with minibatch"""
    total_losses = {
        "total": 0,
        "token_ppl": 0,
        "token_loss": 0,
        "sentence_type_loss": 0,
        "phrase_selection_loss": 0,
    }
    n_iters = 0

    train_tqdm = tqdm(enumerate(train_dataloader), total=len(train_dataloader.dataset)/args.batch_size)
    for batch_ix, batch in train_tqdm:
        batch = utils.move_to_cuda(batch)
        stype_logits, token_logits, ph_attn, _ = model(batch)

        losses = compute_losses(token_logits=token_logits,
                                token_targets=batch["dec_out"],
                                pad_token_id=vocab.pad_idx,
                                sentence_type_logits=stype_logits,
                                sentence_type_targets=batch["sent_types"],
                                ph_bank_attn=ph_attn,
                                ph_bank_len=batch["ph_bank_len_tensor"],
                                ph_bank_sel_ind_targets=batch["ph_sel_ind_tensor"])

        model_loss = losses['token_loss'] + \
                     args.gamma * losses["sentence_type_loss"] + \
                     args.eta * losses["phrase_selection_loss"]
        model_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.global_steps += 1
        n_iters += 1

        for loss_type in losses:
            total_losses[loss_type] += losses[loss_type].item()
        total_losses["total"] += model_loss.item()

        if batch_ix % 50 == 0:
            tb_logger.add_scalar("train_loss_total", model_loss.item(),
                                 model.global_steps)
            tb_logger.add_scalar("train_loss_token", losses["token_loss"].item(),
                                 model.global_steps)
            tb_logger.add_scalar("train_loss_sentence_type", losses["sentence_type_loss"].item(),
                                 model.global_steps)
            tb_logger.add_scalar("train_loss_phrase_selection", losses["phrase_selection_loss"].item(),
                                 model.global_steps)
            tb_logger.add_scalar("train_PPL", losses["token_ppl"].item(),
                                 model.global_steps)

            train_tqdm.set_postfix_str('train_loss={:.2f}'.format(model_loss.item()), refresh=False)

    return {loss_type: loss_val/n_iters for loss_type, loss_val in total_losses.items()}


def valid_epoch(model, valid_dataloader, args, vocab, tb_logger):
    total_losses = {
        "total": 0,
        "token_ppl": 0,
        "token_loss": 0,
        "sentence_type_loss": 0,
        "phrase_selection_loss": 0,
    }
    n_iters = 0
    for batch_ix, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader.dataset)/args.batch_size):
        batch = utils.move_to_cuda(batch)
        stype_logits, token_logits, ph_attn, _ = model(batch)

        losses = compute_losses(token_logits=token_logits,
                                token_targets=batch["dec_out"],
                                pad_token_id=vocab.pad_idx,
                                sentence_type_logits=stype_logits,
                                sentence_type_targets=batch["sent_types"],
                                ph_bank_attn=ph_attn,
                                ph_bank_len=batch["ph_bank_len_tensor"],
                                ph_bank_sel_ind_targets=batch["ph_sel_ind_tensor"])

        model_loss = losses['token_loss'] + \
                     args.gamma * losses["sentence_type_loss"] + \
                     args.eta * losses["phrase_selection_loss"]

        for loss_type in losses:
            total_losses[loss_type] += losses[loss_type].item()
        total_losses["total"] += model_loss.item()
        n_iters += 1

    tb_logger.add_scalar("valid_loss_total",
                         total_losses["total"] / n_iters,
                         model.global_steps)
    tb_logger.add_scalar("valid_loss_token",
                         total_losses["token_loss"] / n_iters,
                         model.global_steps)
    tb_logger.add_scalar("valid_loss_sentence_type",
                         total_losses["sentence_type_loss"] / n_iters,
                         model.global_steps)
    tb_logger.add_scalar("valid_loss_phrase_selection",
                         total_losses["phrase_selection_loss"] / n_iters,
                         model.global_steps)
    tb_logger.add_scalar("valid_PPL",
                         total_losses["token_ppl"] / n_iters,
                         model.global_steps)
    return {loss_type: loss_val/n_iters for loss_type, loss_val in total_losses.items()}
 
