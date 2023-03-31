import torch
import json
import tqdm
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset

#most of this code is going to be the same from classification_lib as well, but some additions to potentially keep everything organized
TRAIN, EVAL, PREDICT, DEV, TEST = "train eval predict dev test".split()
MODES = [TRAIN, EVAL, PREDICT]

BATCH_SIZE = 16
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"

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
    with open(f"{in_dir}/multi_task/{subset}.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            review_id, sentence_index = example["identifier"].split("|||")
            if review_id not in reviews:
                reviews[review_id] = {
                    "sentences": {},
                    "labels": {}
                }
            reviews[review_id]["sentences"][sentence_index] = example["text"]
            reviews[review_id]["labels"][sentence_index] = [example[label] for label in tasks]
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

def get_label_list(data_dir, tasks):
    #task can be a list of tasks
    print(data_dir)
    print(tasks)
    with open(f"{data_dir}/metadata.json", "r") as f:
        return [json.load(f)["labels"][task] for task in tasks]

class MultiTaskReviewDataset(Dataset): #This dataset is different from the one in clssification_lib because it's entries are the full reviews

    def __init__(self, input_dir, tasks, tokenizer, max_len=512):
        self.review_list = []
        self.targets_indices_list = []
        self.targets_list = []
        #this should be done in one pass
        (
            identifiers,
            texts,
            target_indices,
        ) = get_text_and_labels(input_dir, tasks, subset, get_labels=True)
        target_set = set(target_indices)
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


class BertHSLN(nn.Module):
    def __init__(self, num_tasks=1):
        super(BertHSLN, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.bidirect_hidden = 768
        self.bilstm1 = nn.LSTM(self.bert.config.hidden_size, self.bidirect_hidden, num_layers=1, bidirectional=True)
    #each sentence should be a tuple of (input_ids, attention_masks)
    def forward(self, sentences, task_id):
        task_id = task_id.item()
        bert_ouputs = [] #initialize empty array for BERT outputs
        for input_ids, attention_mask in embeddings:
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"]
            bert_outputs.append(bert_output)
        #transform bert_ouputs into the proper shape
        #feed bert_outputs into the bilstm1
        out = self.bilstm1(bert_outputs)

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
        for entry in tqdm.tqdm(data_loader):
            for i in range(num_tasks):
                sentence_embeddings, attention_masks = [d[k].to(device) for k in "input_ids attention_mask".split()]
                task_id = i
                task_id = torch.tensor(task_id, dtype=torch.int32, device="cuda")
                targets = d["targets"][i].to(device)
                target_indices = d["target_indices"][i].to(device)
                outputs = model(embeddings=sentence_embeddings, attention_mask=attention_masks, task_id=task_id)
                _, preds = torch.max(outputs, dim=1)

