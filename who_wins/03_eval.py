import argparse
import collections
import csv
import pickle
import torch
import torch.nn as nn
import transformers

from contextlib import nullcontext
from torch.optim import AdamW
from transformers import BertTokenizer, RobertaTokenizer

import who_wins_lib

seed = 34
torch.manual_seed(seed)
import random
random.seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
import numpy as np
np.random.seed(seed)


DEVICE = "cuda"


parser = argparse.ArgumentParser(
    description="Evaluate BERT model for DISAPERE classification tasks"
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="path to overall data directory",
)

parser.add_argument(
    "-e",
    "--eval_subset",
    type=str,
    choices="train dev test predict".split(),
    help="subset to evaluate",
)


def argmax(l):
    return l.index(max(l))


def do_eval(tokenizer, model, config, eval_subset):
    """Evaluate without backpropagating."""
    if eval_subset == who_wins_lib.DEV:
      dev_first_split = False
    else:
      dev_first_split = None
    data_loader = who_wins_lib.create_data_loader(
        config,
        eval_subset,
        tokenizer,
        dev_first_split=dev_first_split,
    )

    # Get best model
    model.load_state_dict(
        torch.load(f"checkpoints/{config.config_name}/best_bert_model.bin")
    )
    results = who_wins_lib.train_or_eval(
        who_wins_lib.EVAL, model, data_loader, DEVICE, return_probs=True
    )

    with open(f"results/{config.config_name}_{eval_subset}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow("identifier label true_label".split() + config.labels)

        for i in results:
            for identifier, probs, target in zip(*i):
                writer.writerow([identifier, argmax(probs), target] + probs)


def main():
    args = parser.parse_args()

    config = who_wins_lib.read_config(args.config)
    if config.model_name == "roberta-base":
      tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    else:
      tokenizer = BertTokenizer.from_pretrained(config.model_name)

    model = who_wins_lib.Classifier(len(config.labels), config.model_name).to(DEVICE)
    model.loss_fn.to(DEVICE)

    do_eval(tokenizer, model, config, args.eval_subset)


if __name__ == "__main__":
    main()
