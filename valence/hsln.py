
from contextlib import nullcontext
import torch
import json
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from models import CRFOutputLayer

#most of this code is going to be the same from classification_lib as well, but some additions to potentially keep everything organized
TRAIN, EVAL, PREDICT, DEV, TEST = "train eval predict dev test".split()
MODES = [TRAIN, EVAL, PREDICT]

BATCH_SIZE = 1
SENTENCE_BATCH_SIZE = 32
PRE_TRAINED_MODEL_NAME = "allenai/scibert_scivocab_uncased"

#hard code each task's respective none values
NULLS = {
"ms_aspect": 9,
"polarity": 0,
"review_action": 7
}
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

def extract_reviews(in_dir, tasks, subset):
    reviews = {}
    with open(f"{in_dir}/{subset}.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            review_id, sentence_index = example["identifier"].split("|||")
            if review_id not in reviews:
                reviews[review_id] = {
                    "sentences": [],
                    "labels": []
                }
            reviews[review_id]["sentences"].append(example["text"])
            reviews[review_id]["labels"].append([example[label] for label in tasks])
    return reviews

def get_text_and_labels(in_dir, tasks, subset, get_labels=True):
    reviews = extract_reviews(in_dir, tasks, subset)
    identifiers = []
    texts = []
    labels = []
    for key, value in reviews.items():
        identifiers.append(key)
        texts.append(reviews[key]["sentences"])
        labels.append(reviews[key]["labels"])
    if(not get_labels):
        labels = None
    return identifiers, texts, labels

def create_multitask_data_loader(in_dir, task_dirs, subset, tokenizer):
    ds = MultiTaskReviewDataset(
        in_dir,
        subset,
        task_dirs,
        tokenizer
    )
    print("Dataset successfully loaded")
    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1)

def build_data_loader_multitask(in_dir, task_dirs, tokenizer):
    return (
        create_multitask_data_loader(
            in_dir,
            task_dirs,
            TRAIN,
            tokenizer,
        ),
        create_multitask_data_loader(
            in_dir,
            task_dirs,
            DEV,
            tokenizer,
        ),
    )


def get_label_list(data_dir, tasks):
    #task can be a list of tasks
    print(data_dir)
    print(tasks)
    
    with open(f"{data_dir}/metadata.json", "r") as f:
        data = json.load(f)
        return [data["labels"][task] for task in tasks]

class MultiTaskReviewDataset(Dataset): #This dataset is different from the one in clssification_lib because it's entries are the full reviews

    def __init__(self, input_dir, subset, tasks, tokenizer, max_len=256):
        self.review_list = []
        self.targets_indices_list = [[] for _ in tasks]
        self.targets_list = [[] for _ in tasks]
        self.text = []
        self.tokenizer = tokenizer
        self.label_list = get_label_list(input_dir, tasks)
        self.eyes = [np.eye(len(labels), dtype=np.float64) for labels in self.label_list]
        #this should be done in one pass
        (
            identifiers,
            texts,
            target_indices,
        ) = get_text_and_labels(input_dir, tasks, subset, get_labels=True)
        input_index = 0
        self.identifier_list = []
        for i in range(len(target_indices)):
            review = target_indices[i] #target indices of review i
            for j in range(0, max(len(review) - SENTENCE_BATCH_SIZE, 1)):
                labels = review[j:min(j+SENTENCE_BATCH_SIZE, len(review))]
                sentences = [torch.zeros((SENTENCE_BATCH_SIZE, max_len), dtype=torch.int32) for _ in range(2)]
                t_list = [[] for _ in tasks]
                indices_list = [[] for _ in tasks]
                if(len(labels) == 0):
                    continue
                for val in range(SENTENCE_BATCH_SIZE): #tokenize everything at the start first
                    idx = j+val
                    if(idx == len(review)):
                        #slice both Tensors in sentences so we don't have "empties"
                        sentences[0] = sentences[0][0:val]
                        sentences[1] = sentences[1][0:val]
                        break
                    encoded = tokenizer_fn(self.tokenizer, texts[i][idx])
                    sentences[0][val,:] = encoded["attention_mask"]
                    sentences[1][val,:] = encoded["input_ids"]
                for k in range(len(tasks)):
                    y = [labels[idx][k] for idx in range(len(labels))]
                    t_list[k].append(y)
                    indices_list[k] = [self.eyes[k][y[idx]] for idx in range(len(y))]
                for k in range(len(tasks)):
                    self.targets_list[k].append(t_list[k])
                    self.targets_indices_list[k].append(indices_list[k])
                self.text.append(sentences)
                self.identifier_list.append(identifiers[i])
        self.max_len = max_len
        print(self.targets_indices_list[0][0])
        print(self.targets_list[0][0])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return {
            "input_ids": self.text[item][1],
            "attention_mask": self.text[item][0],
            "targets": [torch.tensor(self.targets_list[i][item], dtype=torch.int64) for i in range(len(self.targets_list))],
            "target_indices": [torch.tensor(np.array(self.targets_indices_list[i][item])) for i in range(len(self.targets_indices_list))],
            "identifier": self.identifier_list[item],
        }
#adapted from the sequential-sentence-classification
class CRFPerTaskOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, tasks, labels):
        super(CRFPerTaskOutputLayer, self).__init__()

        self.per_task_output = torch.nn.ModuleDict()
        for i in range(len(tasks)):
            self.per_task_output[tasks[i]] = CRFOutputLayer(in_dim=in_dim, num_labels=len(labels[i]))


    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        if(labels is not None):
            _, batch, max_sequence = labels.shape
            labels = labels.view(batch, max_sequence) 
        output = self.per_task_output[task](x, mask, labels)
        output_without_lbs = self.per_task_output[task](x, mask)
        output["predicted_label"] = output_without_lbs["predicted_label"]
        if output_all_tasks:
            output["task_outputs"] = []
            for t, task_output in self.per_task_output.items():
                task_result = task_output(x, mask)
                task_result["task"] = t
                output["task_outputs"].append(task_result)
        return output

class BertHSLN(nn.Module):
    def __init__(self, labels, tasks, dropout):
        super(BertHSLN, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=dropout)
        self.tasks = tasks
        self.crf = CRFPerTaskOutputLayer(self.bert.config.hidden_size, tasks, labels)
        
    #each sentence should be a tuple of (input_ids, attention_masks)
    def forward(self, input_ids, attention_mask, labels, task_id):
        documents, sentences, tokens = input_ids.shape
        task_id = task_id.item()
        bert_output = self.bert(input_ids=input_ids[0], attention_mask=attention_mask[0])["pooler_output"]
        if(self.dropout):
            bert_output = self.dropout(bert_output)
        bert_output = bert_output.view(documents, sentences, -1)
        return self.crf(self.tasks[task_id], bert_output, None, labels, output_all_tasks=False)

    def get_predictions(self, input_ids, attention_mask, labels, task_id):
        task_id = task_id.item()
        bert_output = self.bert(input_ids=input_ids[0], attention_mask=attention_mask[0])["pooler_output"]
        return F.softmax(self.linear(bert_output), dim=1)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

#verify labels to make sure that we are actually training on something that has concrete labels associated with it and not just all nones
def verify_labels(labels, task):
    none_val = NULLS[task]
    all_none = torch.full(labels.size(), none_val)
    return not torch.equal(labels, all_none)
    
#first take a look at how the training loop with work
#should have labels divided up per task
def train_or_eval(
    mode,
    model,
    data_loader,
    device,
    return_preds=False,
    optimizer=None,
    scheduler=None,
    num_tasks=1
):
    assert mode in [TRAIN, EVAL]
    losses = torch.zeros((num_tasks,len(data_loader)))
    correct_predictions = [0 for _ in range(num_tasks)]
    total_sentences_per_task = [0 for _ in range(num_tasks)]
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
    with context:
        idx = 0
        for entry in tqdm.tqdm(data_loader):
            for i in range(num_tasks):
              
                task_id = i
                task_id = torch.tensor(task_id, dtype=torch.int32, device="cuda")
                targets = entry["targets"][i].to(device)
                sentence_embeddings = entry["input_ids"].to(device)
                attention_masks = entry["attention_mask"].to(device)

                target_indices = entry["target_indices"][i].to(device)
                outputs = model(sentence_embeddings, attention_masks, targets, task_id=task_id)
                loss = outputs["loss"] / targets.size()[2]
                losses[i][idx] = loss.item()
                if(verify_labels(entry["targets"][i], model.tasks[i])):
                    if(is_train):
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    preds = outputs["predicted_label"].to(device)
                    correct_predictions[i] += torch.sum(preds == targets)
                    total_sentences_per_task[i] += targets.size()[2]
                else:
                    if(is_train):
                        print("Do not propogate loss")
                    else:
                        preds = outputs["predicted_label"].to(device)
                        correct_predictions[i] += torch.sum(preds == targets)
                        total_sentences_per_task[i] += targets.size()[2]
        for i in range(num_tasks):
            accuracy = correct_predictions[i] / total_sentences_per_task[i]
            print("Accuracy for task {}: {}".format(i, accuracy))
    return sum(correct_predictions) / (np.sum(total_sentences_per_task)), torch.sum(losses)

