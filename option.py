import argparse

def _add_common_args(parser):
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--debug", action="store_true",
                        help="Load 10 sample instances for debug.")
    parser.add_argument("--exp-name", type=str, required=True)

    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Dimension of RNN hidden states.")

    parser.add_argument("--encode-passage", action="store_true",
                        help="whether to include passages as part of encoder input.")
    parser.add_argument("--max-tgt-sent", type=int, default=5,
                        help="Maximum number of sentences to consider (in target).")
    parser.add_argument("--min-tgt-sent", type=int, default=2,
                        help="Minimum number of sentences to consider (in target).")
    parser.add_argument("--max-tgt-token", type=int, default=100,
                        help="Maximum number of tokens to consider (in target), if longer do truncation.")
    parser.add_argument("--max-op-token", type=int, default=100,
                        help="Maximum number of tokens in src (OP), if longer do truncation.")
    parser.add_argument("--max-passage-token", type=int, default=300,
                        help="Maximum number of tokens in src (passage), if longer do truncation.")
    parser.add_argument("--max-phrase-per-sent", type=int, default=10,
                        help="Maximum number of phrases in each sentence (target).")
    parser.add_argument("--max-phrase-bank-size", type=int, default=20,
                        help="Maximum number of phrases in the phrase bank (target paragraph).")


def _add_training_specific_args(parser):
    parser.add_argument("--max-epochs", type=int, default=20,
                        help="total number of epochs to train.")
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--init-accum", type=float, default=0.1)
    parser.add_argument("--save-freq", type=int, default=3,
                        help="the number of checkpoints to save in each epoch.")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Loss coefficient for sentence type prediction.")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="Loss coefficient for phrase selection.")


def _add_inference_specific_args(parser):
    parser.add_argument("--epoch-id", type=int, default=-1,
                        help="The epoch id of checkpoint to load, use -1 to "
                             "load the latest one.")
    parser.add_argument("--max-phrase-selection-time", type=int, default=2)
    parser.add_argument("--max-token-per-sentence", type=int, default=30)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--block-ngram-repeat", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use-goldstandard-plan", action="store_true")

def get_training_parser():
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    _add_training_specific_args(parser)
    return parser

def get_inference_parser():
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    _add_inference_specific_args(parser)
    return parser
