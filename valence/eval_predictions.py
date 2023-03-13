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
from sklearn.metrics import confusion_matrix
import classification_lib

DEVICE = "cuda"


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
    "-e",
    "--eval_subset",
    type=str,
    choices="train dev test".split(),
    help="subset to evaluate",
)


parser.add_argument(
    "-t",
    "--task",
    nargs="*",
    type=str,
    help="train eval or predict",
)

parser.add_argument(
    "-c",
    "--ckpt",
    type=str,
    help="path to the model checkpoint"
)
def do_multitask_eval(tokenizer, model, task_dir, eval_subset, model_path):
    data_loader = classification_lib.create_multitask_data_loader(task_dir, eval_subset, tokenizer)
    model.load_state_dict(torch.load(model_path))
    print(isinstance(model, collections.OrderedDict))
    predictions, labels = classification_lib.train_or_eval(
        classification_lib.EVAL, model, data_loader, DEVICE, multi_train=True, return_preds=True
    )

    print(type(predictions))
    task1_pred = []
    task1_actual = []
    task2_pred = []
    task2_actual = []
    for i in range(0, len(predictions), 2):
        for j in range(len(predictions[i][1])):
            task1_pred.append(predictions[i][1][j])
            task1_actual.append(labels[i][1][j])
        for n in range(len(predictions[i+1][1])):
            task2_pred.append(predictions[i+1][1][n])
            task2_actual.append(labels[i+1][1][n])
    conf1 = confusion_matrix(task1_actual, task1_pred)
    print(conf1)
    conf2 = confusion_matrix(task2_actual, task2_pred)
    print(conf2)


def do_eval(tokenizer, model, task_dir, eval_subset):
    """Evaluate (on dev set?) without backpropagating."""
    data_loader = classification_lib.create_data_loader(
        task_dir,
        eval_subset,
        tokenizer,
    )

    # Get best model
    model.load_state_dict(torch.load(f"{task_dir}/ckpt/best_bert_model.bin"))
    acc, loss = classification_lib.train_or_eval(
        classification_lib.EVAL, model, data_loader, DEVICE
    )

    print(f"Accuracy: {acc} Loss: {loss}")


def main():
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(classification_lib.PRE_TRAINED_MODEL_NAME)
    if(len(args.task) > 1):
        print("Multi task")
        print(args.task)
        tasks = []
        all_labels = []
        dirs = []
        for task in args.task:
            labels = classification_lib.get_label_list(args.data_dir, task)
            tasks.append(task)
            all_labels.append(labels)
            task_dir = classification_lib.make_checkpoint_path(args.data_dir, task)
            dirs.append(task_dir)
        model = classification_lib.MultiTaskClassifier(all_labels).to(DEVICE)
        for i in range(len(args.task)):
            model.loss[i].to(DEVICE)
    else:
        labels = classification_lib.get_label_list(args.data_dir, args.task)
        model = classification_lib.Classifier(len(labels)).to(DEVICE)
        
        task_dir = classification_lib.make_checkpoint_path(args.data_dir, args.task)
        model.loss_fn.to(DEVICE)
    if(len(args.task) > 1):
        do_multitask_eval(tokenizer, model, tasks, args.eval_subset, args.ckpt)
    else:
        do_eval(tokenizer, model, task_dir, args.eval_subset)


if __name__ == "__main__":
    main()
