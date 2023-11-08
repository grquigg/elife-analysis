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
import multi_classification_lib
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

# Hyperparameters
DEVICE = "cpu"
EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 2e-5

HistoryItem = collections.namedtuple(
    "HistoryItem", "epoch train_acc train_loss val_acc val_loss".split()
)

Example = collections.namedtuple("Example", "identifier text target".split())


def do_train(tokenizer, model, tasks, data_dir):
    """Train on train set, validating on validation set."""
    #TO-DO: add dataloader such that it reduces the amount of repeated code necessary 
    #we give the list of tasks to the data_loader rather than a specific path to a file
    (
       train_data_loader,
       val_data_loader,
    ) = hsln.build_data_loader_multitask(data_dir, tasks, tokenizer)
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
            scheduler=scheduler
        )

        # Run train_or_eval on validation set in EVAL mode
        val_acc, val_loss = classification_lib.train_or_eval(
            classification_lib.EVAL, model, val_data_loader, DEVICE, multi_train=multi_train
        )

        # Recording metadata
        history.append(HistoryItem(epoch, train_acc, train_loss, val_acc, val_loss))

        # Save the model parameters if this is the best model seen so far
        if val_acc > best_accuracy:
            task_name = "_".join(task_dir)
            print(task_name)
            torch.save(model.state_dict(), f"disapere_data/{task_name}/ckpt/best_bert_model.bin")
            with open(f"disapere_data/{task_name}/ckpt/history.pkl", "wb") as f:
                pickle.dump(history,f)
        else:
            torch.save(model.state_dict(), f"{task_dir}/ckpt/best_bert_model.bin")
            with open(f"{task_dir}/ckpt/history.pkl", "wb") as f:
                pickle.dump(history, f)
            best_accuracy = val_acc
            best_accuracy_epoch = epoch


def main():
    args = parser.parse_args()
    all_text, all_labels = [], []
    tasks = args.task
    for task in tasks:
        text, labels = multi_classification_lib.get_text_and_labels(args.data_dir, task, "train", get_labels=True)
        all_text += text
        all_labels += labels
    print(len(all_text))
    print(len(all_labels))
    tokenizer = AutoTokenizer.from_pretrained(hsln.PRE_TRAINED_MODEL_NAME)
    labels = hsln.get_label_list(args.data_dir, tasks)
    task_dir = classification_lib.make_checkpoint_path(args.data_dir, "_".join(tasks)+"mod")
    model = hsln.BertHSLN(labels, tasks, 0.6).to(DEVICE)
    # if(args.model_path):
    #     model.load_state_dict(torch.load(args.model_path))
    print(f"Task_dir: {task_dir}\nData_dir: {args.data_dir}")
    do_train(tokenizer, model, tasks, args.data_dir)


if __name__ == "__main__":
    main()
