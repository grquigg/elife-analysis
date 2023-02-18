from contextlib import nullcontext
import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tqdm
from transformers import BertTokenizer, BertModel

TRAIN, EVAL, PREDICT, DEV, TEST = "train eval predict dev test".split()
MODES = [TRAIN, EVAL, PREDICT]

BATCH_SIZE = 4
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"

# Wrapper around the tokenizer specifying the details of the BERT input
# encoding.
tokenizer_fn = lambda tok, text: tok.encode_plus(
    text,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)

def make_identifier(review_id, index):
    return f"{review_id}|||{index}"


def get_text_and_labels(task_dir, subset, get_labels=False):

  texts = []
  identifiers = []
  labels = []
  with open(f"{task_dir}/{subset}.jsonl", "r") as f:
    for line in f:
      example = json.loads(line)
      texts.append(example["text"])
      identifiers.append(example["identifier"])
      if get_labels:
        labels.append(example["label"])
  if not get_labels:
    labels = None

  return identifiers, texts, labels

class MultiTaskClassificationDataset(Dataset):

    def __init__(self, task_dirs, subset, tokenizer, max_len=512):
        self.identifier_list = []
        self.text_list = []
        self.targets_indices_list = []
        self.targets_list = []
        for i in range(len(task_dirs)):
            (
                identifiers,
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

class ClassificationDataset(Dataset):
    """A torch.utils.data.Dataset for binary classification."""

    def __init__(self, task_dir, subset, tokenizer, max_len=512):
        (
            self.identifiers,
            self.texts,
            self.target_indices,
        ) = get_text_and_labels(task_dir, subset, get_labels=True)
        target_set = set(self.target_indices)
        assert list(sorted(target_set)) == list(range(len(target_set)))
        eye = np.eye(
            len(target_set), dtype=np.float64
        )  # An identity matrix to easily switch to and from one-hot encoding.
        self.targets = [eye[int(i)] for i in self.target_indices]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]

        encoding = tokenizer_fn(self.tokenizer, text)

        return {
            "reviews_text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.float64),
            "target_indices": self.target_indices[item],
            "identifier": self.identifiers[item],
        }

def create_data_loader(task_dir, subset, tokenizer):
    """Wrap a DataLoader around a PolarityDetectionDataset.

    While the dataset manages the content of the data, the data loader is more
    concerned with how the data is doled out, and is the connection between the
    dataset and the model.
    """
    ds = ClassificationDataset(
        task_dir,
        subset,
        tokenizer=tokenizer,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)


def build_data_loaders(task_dir, tokenizer):
    """Build train and dev data loaders from a structured data directory.

    TODO(nnk): Investigate why there is no test data loader.
    """
    return (
        create_data_loader(
            task_dir,
            TRAIN,
            tokenizer,
        ),
        create_data_loader(
            task_dir,
            DEV,
            tokenizer,
        ),
    )

def create_multitask_data_loader(task_dirs, subset, tokenizer):
    ds = MultiTaskClassificationDataset(
        task_dirs,
        subset,
        tokenizer
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)

def build_data_loader_multitask(task_dirs, tokenizer):
    return (
        create_multitask_data_loader(
            task_dirs,
            TRAIN,
            tokenizer,
        ),
        create_multitask_data_loader(
            task_dirs,
            DEV,
            tokenizer,
        ),
    )
class MultiTaskClassifier(nn.Module):
    def __init__(self, num_classes_array):
        super(MultiTaskClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = []
        self.loss = []
        for i in range(len(num_classes_array)):
            self.out.append(nn.Linear(self.bert.config.hidden_size, len(num_classes_array[i])))
            if len(num_classes_array[i]) == 2:
                self.loss.append(nn.BCEWithLogitsLoss())  # Not sure if this is reasonable
            else:
                self.loss.append(nn.CrossEntropyLoss())

    def forward(self, input_ids, attention_mask, task_id):  # This function is required
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(bert_output["pooler_output"])
        return self.out[task_id](output)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)
        if num_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()  # Not sure if this is reasonable
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):  # This function is required
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(bert_output["pooler_output"])
        return self.out(output)



def get_label_list(data_dir, task):
    #task can be a list of tasks
    print(data_dir)
    print(task)
    with open(f"{data_dir}/{task}/metadata.json", "r") as f:
        return json.load(f)["labels"]


def make_checkpoint_path(data_dir, task):
    task_dir = f"{data_dir}/{task}/"
    ckpt_dir = f"{task_dir}/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    return task_dir

def train_or_eval(
    mode,
    model,
    data_loader,
    device,
    return_preds=False,
    optimizer=None,
    scheduler=None,
    multi_train = False
):
  """Do a forward pass of the model, backpropagating only for TRAIN passes.
  """
  assert mode in [TRAIN, EVAL]
  is_train = mode == TRAIN
  if is_train:
    model = model.train() # Put the model in train mode
    context = nullcontext()
    # ^ This is so that we can reuse code between this mode and eval mode, when
    # we do have to specify a context
    assert optimizer is not None # Required for backprop
    assert scheduler is not None # Required for backprop
  else:
    model = model.eval() # Put the model in eval mode
    context = torch.no_grad() # Don't backpropagate

  results = []
  losses = []
  correct_predictions = 0
  n_examples = len(data_loader.dataset)

  with context:
    if(multi_train):
        for d in tqdm.tqdm(data_loader):
            input_ids, attention_mask = [
               d[k].to(device) # Move all this stuff to gpu
               for k in "input_ids attention_mask".split()
            ]
            for i in range(len(d["targets"])):
                task_id = i
                task_id = torch.tensor(task_id, dtype=torch.int32, device="cuda:0")
                targets = d["targets"][i].to(device)
                target_indices = d["target_indices"][i].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, task_id=task_id)
                _, preds = torch.max(outputs, dim=1)

                if return_preds:
                # If this is being run as part of prediction, we need to return the
                # predicted indices. If we are just evaluating, we just need loss and/or
                # accuracy
                    results.append((d["identifier"], preds.cpu().numpy().tolist()))

                # We need loss for both train and eval
                loss = model.loss[i](outputs, targets)
                losses.append(loss.item())
                if is_train:
                # Backpropagation steps
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Counting correct predictions in order to calculate accuracy later
                correct_predictions += torch.sum(preds == target_indices)
        #second option: build a custom dataloader
    else:
        for d in tqdm.tqdm(data_loader): # Load batchwise
          input_ids, attention_mask, targets, target_indices = [
              d[k].to(device) # Move all this stuff to gpu
              for k in "input_ids attention_mask targets target_indices".split()
          ]

          outputs = model(input_ids=input_ids, attention_mask=attention_mask)
          # ^ this gives logits
          _, preds = torch.max(outputs, dim=1)
          # TODO(nnk): make this argmax!
          if return_preds:
          # If this is being run as part of prediction, we need to return the
          # predicted indices. If we are just evaluating, we just need loss and/or
          # accuracy
            results.append((d["identifier"], preds.cpu().numpy().tolist()))

          # We need loss for both train and eval
          loss = model.loss_fn(outputs, targets)
          losses.append(loss.item())

          # Counting correct predictions in order to calculate accuracy later
          correct_predictions += torch.sum(preds == target_indices)

          if is_train:
            # Backpropagation steps
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

  if return_preds:
    return results
  else:
    # Return accuracy and mean loss
    return correct_predictions.double().item() / n_examples, np.mean(losses)


