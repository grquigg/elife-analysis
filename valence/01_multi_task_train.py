import argparse
import collections
import pickle
import torch
import torch.nn as nn
import transformers
import glob

from contextlib import nullcontext
from torch.optim import AdamW
from transformers import BertTokenizer, AutoTokenizer

import classification_lib
import hsln

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
    nargs="*",
    help="train eval or predict",
)

parser.add_argument(
    "-m",
    "--model_path",
    type=str,

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

def do_train(tokenizer, model, data_dir, task_dir, multi_train=False):
    """Train on train set, validating on validation set."""
    print("TRAIN")
    (
       train_data_loader,
       val_data_loader,
    ) = hsln.build_data_loader_multitask(data_dir, task_dir, tokenizer)
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
        train_acc, train_loss = hsln.train_or_eval(
            classification_lib.TRAIN,
            model,
            train_data_loader,
            DEVICE,
            optimizer=optimizer,
            scheduler=scheduler,
            num_tasks = len(task_dir)
        )

        # Run train_or_eval on validation set in EVAL mode
        val_acc, val_loss = hsln.train_or_eval(
            classification_lib.EVAL, model, val_data_loader, DEVICE, num_tasks=len(task_dir)
        )

        # Recording metadata
        history.append(HistoryItem(epoch, train_acc, train_loss, val_acc, val_loss))
        print(data_dir)
        # Save the model parameters if this is the best model seen so far
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "disapere_data/multi_task/{}/ckpt/best_bert_model.bin".format("_".join(task_dir)))
            with open("disapere_data/multi_task/{}/ckpt/history.pkl".format("_".join(task_dir)), "wb") as f:
                pickle.dump(history, f)
            best_accuracy = val_acc
            best_accuracy_epoch = epoch


def main():
    args = parser.parse_args()
    print(args.task)
    tokenizer = AutoTokenizer.from_pretrained(hsln.PRE_TRAINED_MODEL_NAME)
    labels = hsln.get_label_list(args.data_dir, args.task)
    task_dir = classification_lib.make_checkpoint_path(args.data_dir, "_".join(args.task))
    model = hsln.BertHSLN(labels, args.task, 0.6).to(DEVICE)
    if(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
    do_train(tokenizer, model, args.data_dir, args.task, multi_train=True)

if __name__ == "__main__":
    main()
