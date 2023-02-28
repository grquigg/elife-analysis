import argparse
import collections
import os
import pickle
import torch
import transformers

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


parser = argparse.ArgumentParser(
    description="Train BERT model for DISAPERE classification tasks"
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="config name",
)

# Hyperparameters
DEVICE = "cuda"
EPOCHS = 10
PATIENCE = 5
LEARNING_RATE = 2e-5

HistoryItem = collections.namedtuple(
    "HistoryItem", "epoch train_acc train_loss val_acc val_loss".split()
)

Example = collections.namedtuple("Example", "identifier text target".split())


def do_train(tokenizer, model, config):
    """Train on train set, validating on validation set."""

    (
        train_data_loader,
        val_data_loader,
    ) = who_wins_lib.build_data_loaders(config, tokenizer)

    # Optimizer and scheduler (boilerplatey)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    checkpoint_dir = f"checkpoints/{config.config_name}/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = []
    best_accuracy = 0
    best_accuracy_epoch = None

    # EPOCHS is the maximum number of epochs we will run.
    for epoch in range(EPOCHS):

        # If no improvement is seen in PATIENCE iterations, we quit.
        if best_accuracy_epoch is not None and epoch - best_accuracy_epoch > PATIENCE:
            break

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        # Run train_or_eval ono train set in TRAIN mode, backpropagating
        train_acc, train_loss = who_wins_lib.train_or_eval(
            who_wins_lib.TRAIN,
            model,
            train_data_loader,
            DEVICE,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Run train_or_eval on validation set in EVAL mode
        val_acc, val_loss = who_wins_lib.train_or_eval(
            who_wins_lib.EVAL, model, val_data_loader, DEVICE
        )

        # Recording metadata
        history.append(HistoryItem(epoch, train_acc, train_loss, val_acc, val_loss))

        # Save the model parameters if this is the best model seen so far
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_bert_model.bin")
            best_accuracy = val_acc
            best_accuracy_epoch = epoch

        with open(f"{checkpoint_dir}/history.pkl", "wb") as f:
            pickle.dump(history, f)


def main():
    args = parser.parse_args()

    config = who_wins_lib.read_config(args.config)
    if config.model_name == "roberta-base":
      tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    else:
      tokenizer = BertTokenizer.from_pretrained(config.model_name)

    model = who_wins_lib.Classifier(len(config.labels), config.model_name).to(DEVICE)
    model.loss_fn.to(DEVICE)

    do_train(tokenizer, model, config)


if __name__ == "__main__":
    main()
