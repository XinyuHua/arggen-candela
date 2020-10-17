# Author: Xinyu Hua
# Last modified: 2020-10-07
import os
import time
import logging
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from modules.model import Candela
from option import get_training_parser
from vocab import Vocab
from arggen_data import ArgumentGenerationDataset
from modules.trainer import train_epoch, valid_epoch
import utils

def main():

    vocab = Vocab(utils.DATA_DIR + "vocab.txt")
    glove_emb = utils.load_glove_emb(vocab)
    word_emb = nn.Embedding.from_pretrained(
        torch.tensor(glove_emb, dtype=torch.float)
    )

    parser = get_training_parser()
    args = parser.parse_args()

    model = Candela(word_emb=word_emb,
                    word_emb_dim=300,
                    word_vocab_size=len(vocab)).cuda()
    optimizer = optim.Adagrad(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.learning_rate,
                        initial_accumulator_value=args.init_accum)


    args.train_set = "train" if not args.debug else "train-toy"
    args.valid_set = "dev" if not args.debug else "train-toy"

    train_dataset = ArgumentGenerationDataset(args=args,
                                              set_type=args.train_set,
                                              vocab=vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collater)

    valid_dataset = ArgumentGenerationDataset(args=args,
                                              set_type=args.valid_set,
                                              vocab=vocab)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  collate_fn=valid_dataset.collater)

    tb_logger = SummaryWriter(f'runs/{args.exp_name}')

    for n_epoch in range(1, args.max_epochs):
        print("starting epoch {}".format(n_epoch))
        model.train()
        train_loss_info = train_epoch(model, train_dataloader, args, optimizer,
                                      vocab, tb_logger)
        with torch.no_grad():
            model.eval()
            val_loss_info = valid_epoch(model, valid_dataloader, args, vocab,
                                        tb_logger)
        print("train loss: {:.2f} PPL: {:.2f}\tvalid loss: {:.2f} PPL: {:.2f}"
              .format(train_loss_info["total"],
                      train_loss_info["token_ppl"],
                      val_loss_info["total"],
                      val_loss_info["token_ppl"]))

        if n_epoch % args.save_freq == 0:

            if not os.path.exists(f"checkpoints/{args.exp_name}/"):
                os.makedirs(f"checkpoints/{args.exp_name}/")
            checkpoint_path = "checkpoints/{}/epoch_{}_train_{:.3f}_val_{:.3f}.tar"\
                .format(args.exp_name, n_epoch, train_loss_info["total"],
                        val_loss_info["total"])
            torch.save({
                "encoder": model.enc.state_dict(),
                "wd_decoder": model.wd_dec.state_dict(),
                "sp_decoder": model.sp_dec.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": n_epoch,
                }, checkpoint_path)

    return

if __name__=="__main__":
    main()
