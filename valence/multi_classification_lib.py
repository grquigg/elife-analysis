from contextlib import nullcontext
import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.utils import class_weight

BATCH_SIZE = 32
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda"

TASK_SPACE = {"pol": 0, "asp": 1, "epi": 2}
tokenizer_fn = lambda tok, text: tok.encode_plus(
    text,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding="max_length",
    max_length=256,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)

def get_text_and_labels(data_dir, task, subset, get_labels=False):

  texts = []
  labels = []
  with open(f"{data_dir}/{subset}_{task}", "r") as f:
    for line in f:
      example = json.loads(line)
      texts.append(example["sentences"])
      if get_labels:
        labels.append({task: example[task]})
  if not get_labels:
    labels = None

  return texts, labels

class MultiTaskClassificationDataset(Dataset):

    def __init__(self, task_dirs, subset, tokenizer, max_len=512):
        print(task_dirs)
        self.identifier_list = []
        self.text_list = []
        self.targets_indices_list = []
        self.targets_list = []
        for i in range(len(task_dirs)):
            (
                texts,
                target_indices,
            ) = get_text_and_labels(task_dirs[i], subset, get_labels=True)
            target_set = set(target_indices)
            assert list(sorted(target_set)) == list(range(len(target_set)))
            eye = np.eye(
                len(target_set), dtype=np.float64
            )  # An identity matrix to easily switch to and from one-hot encoding.
            targets = [eye[int(i)] for i in target_indices]
            self.identifier_list.append(identifiers)
            self.text_list.append(texts)
            self.targets_indices_list.append(target_indices)
            self.targets_list.append(targets)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text_list[0])

    def __getitem__(self, item):
        text = str(self.text_list[0][item])
        targets = [self.targets_list[i][item] for i in range(len(self.targets_list))]

        encoding = tokenizer_fn(self.tokenizer, text)

        return {
            "reviews_text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": [torch.tensor(targets[i], dtype=torch.float64) for i in range(len(targets))],
            "target_indices": [self.targets_indices_list[i][item] for i in range(len(self.targets_indices_list))],
            "identifier": [self.identifier_list[i][item] for i in range(len(self.identifier_list))],
        }

def create_multitask_data_loader(task_dirs, subset, tokenizer):
    ds = MultiTaskClassificationDataset(
        task_dirs,
        subset,
        tokenizer
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)