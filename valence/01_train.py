import argparse
import collections
import pickle
import torch
import torch.nn as nn
import transformers
import glob

from contextlib import nullcontext
from torch.optim import AdamW
from transformers import BertTokenizer

import classification_lib


parser = argparse.ArgumentParser(
    description="Train BERT model for DISAPERE classification tasks"
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to overall data directory",
)

parser.add_argument(
    "-t",
    "--task",
    type=str,
    help="train eval or predict",
)

# Hyperparameters
DEVICE = "cuda"
EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 2e-5

HistoryItem = collections.namedtuple(
    "HistoryItem", "epoch train_acc train_loss val_acc val_loss".split()
)

Example = collections.namedtuple("Example", "identifier text target".split())


def do_train(tokenizer, model, task_dir, multi_train=False):
    """Train on train set, validating on validation set."""
    if(multi_train):
       (
           train_data_loader,
           val_data_loader,
       ) = classification_lib.build_data_loader_multitask(task_dir, tokenizer)
    # Optimizer and scheduler (boilerplatey)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

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
        train_acc, train_loss = classification_lib.train_or_eval(
            classification_lib.TRAIN,
            model,
            train_data_loader,
            DEVICE,
            optimizer=optimizer,
            scheduler=scheduler,
            multi_train=multi_train
        )

        # Run train_or_eval on validation set in EVAL mode
        val_acc, val_loss = classification_lib.train_or_eval(
            classification_lib.EVAL, model, val_data_loaders, DEVICE, multi_train=multi_train
        )

        # Recording metadata
        history.append(HistoryItem(epoch, train_acc, train_loss, val_acc, val_loss))

        # Save the model parameters if this is the best model seen so far
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), f"{task_dir}/ckpt/best_bert_model.bin")
            best_accuracy = val_acc
            best_accuracy_epoch = epoch

        with open(f"{task_dir}/ckpt/history.pkl", "wb") as f:
            pickle.dump(history, f)


def main():
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(classification_lib.PRE_TRAINED_MODEL_NAME)
    if(args.task == "all"):
        tasks = []
        all_labels = []
        dirs = []
        for filename in glob.glob(f"{args.data_dir}/*/"):
            task = filename.split("/")[1]
            labels = classification_lib.get_label_list(args.data_dir, task)
            tasks.append(task)
            all_labels.append(labels)
            task_dir = classification_lib.make_checkpoint_path(args.data_dir, task)
            dirs.append(task_dir)
        model = classification_lib.MultiTaskClassifier(all_labels).to(DEVICE)
        for i in range(len(tasks)):
            model.loss[i].to(DEVICE)
        do_train(tokenizer, model, dirs, multi_train=True)
    else:
        task_dir = classification_lib.make_checkpoint_path(args.data_dir, args.task)
        model = classification_lib.Classifier(len(labels)).to(DEVICE)
        model.loss_fn.to(DEVICE)
        do_train(tokenizer, model, task_dir)


if __name__ == "__main__":
    main()
