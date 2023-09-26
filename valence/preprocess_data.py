import argparse
import collections
import glob
import tarfile
import json
import os
import stanza
import numpy as np
import math

parser = argparse.ArgumentParser(description="")
parser.add_argument("-d", "--data_dir", default="", type=str, help="")

SPLITS = [0.7, 0.1, 0.2]
# ==== DISAPERE

disapere_pol_map = {"none": "non", "pol_negative": "neg", "pol_positive": "pos"}

disapere_asp_map = {
    "arg_other": "non",
    "asp_clarity": "clr",
    "asp_meaningful-comparison": "mng",
    "asp_motivation-impact": "mot",
    "asp_originality": "org",
    "asp_replicability": "rep",
    "asp_soundness-correctness": "snd",
    "asp_substance": "sbs",
    "none": "non",
}


def get_disapere_labels(sent):
    labels = {
        "pol": disapere_pol_map[sent["polarity"]],
        "asp": disapere_asp_map[sent["aspect"]],
        "rev": sent["review_action"]
    }
    labels["epi"] = (
        "epi" if sent["review_action"] in ["arg_request", "arg_evaluative"] else "nep"
    )
    return labels


def preprocess_disapere(data_dir, write=False, return_data = False):
    #upload all entries in train, dev and test 
    for subset in "train dev test".split():
        #data
        lines = []
        for filename in glob.glob(f"{data_dir}/raw/disapere/{subset}/*.json"):
            with open(filename, "r") as f:
                obj = json.load(f)
                #review_id
                review_id = obj["metadata"]["review_id"]
                review = {"review_id": review_id, "sentences": []}
                for sent in obj["review_sentences"]:
                    #sentence_index: the index of the sentence in the context of the full review
                    sentence = {"sentence_index": sent["sentence_index"], "text": sent["text"]}
                    for task, label in get_disapere_labels(sent).items():
                        if(task not in review):
                            review[task] = []
                        review[task].append(label)
                    review["sentences"].append(sentence)
                lines.append(review)
        if(write):
            for task, examples in lines.items():
                output_dir = f"{data_dir}/labeled/{task}/{subset}/"
                os.makedirs(output_dir, exist_ok=True)
                with open(f"{output_dir}/disapere.jsonl", "w") as f:
                    f.write("\n".join(json.dumps(e) for e in examples))
    if(return_data):
        return lines

# ==== AMPERE

ampere_epi_map = {
    "non-arg": "nep",
    "evaluation": "epi",
    "request": "epi",
    "fact": "nep",
    "reference": "nep",
    "quote": "nep",
}
LABELS = {
    "epi": {
        "nep": 0,
        "epi": 1
    },
    "pol": {
        "non": 0,
        "neg": 1,
        "pos": 2
    },
    "asp": {
        "non": 0,
        "clr": 1,
        "mng": 2,
        "mot": 3,
        "org": 4,
        "rep": 5,
        "snd": 6,
        "sbs": 7
    }
}

def preprocess_ampere(data_dir):
    examples = []
    for filename in glob.glob(f"{data_dir}/raw/ampere/*.txt"):
        review_id = filename.split("/")[-1].rsplit(".", 1)[0].split("_")[0]
        review = {"review_id": review_id, "sentences": [], "epi": []}
        with open(filename, "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                label, sentence = line.strip().split("\t", 1)
                review["epi"].append(ampere_epi_map[label])
                review["sentences"].append({"sentence_index": i, "text": sentence})
        examples.append(review)
    return examples
# ==== ReviewAdvisor

SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize", use_gpu=True)
TOLERANCE = 7

revadv_label_map = {
    "positive": "pos",
    "negative": "neg",
    "clarity": "clr",
    "meaningful_comparison": "mng",
    "motivation": "mot",
    "originality": "org",
    "replicability": "rep",
    "soundness": "snd",
    "substance": "sbs",
}

Sentence = collections.namedtuple("Sentence", "interval text")


def tokenize(text):
    doc = SENTENCIZE_PIPELINE(text)
    sentences = []
    for sentence in doc.sentences:
        start = sentence.to_dict()[0]["start_char"]
        end = sentence.to_dict()[-1]["end_char"]
        sentences.append(Sentence(Interval(start, end), sentence.text))
    return sentences


def label_sentences(sentences, label_obj):
    labels = [list() for _ in range(len(sentences))]
    for label_start, label_end, label in label_obj:
        label_interval = Interval(label_start, label_end)
        for i, sentence in enumerate(sentences):
            if label_interval == sentence.interval:
                labels[i].append(label)
            elif (
                label_start > sentence.interval.upper_bound
                or label_end < sentence.interval.lower_bound
            ):
                pass
            else:
                overlap = sentence.interval & label_interval
                if overlap.upper_bound - overlap.lower_bound > TOLERANCE:
                    labels[i].append(label)
    return labels


def load_revadv_reviews(data_dir):
    reviews = {}
    #load reviews into a dictionary with their respective ids as the keys
    with open(f"{data_dir}/raw/revadv/review_with_aspect.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if(obj["id"] not in reviews):
                reviews[obj["id"]] = []
            reviews[obj["id"]].append(obj)
    return reviews

#this will basically be a function to create a "sparse" dataset similar to what is done for AMPERE and DISAPERE above
def process_revadv_labels(reviews, labels, data_dir, context_width=1):
    dataset = []
    #for each review in the labelled dataset
    count = 0
    for id, review in labels.items():
        count += 1
        #review candidates is the full text of each review that has the label id
        review_candidates = reviews[id]
        #we need tokenize the full text of each review and put it in the "sentences" attribute for each individual review
        for i in range(len(review_candidates)):
            review_candidates[i]["sentences"] = []
            doc = SENTENCIZE_PIPELINE(review_candidates[i]["text"])
            for sentence in doc.sentences:
                review_candidates[i]["sentences"].append(sentence.text)
            for key in labels[id].keys():
                if key == "text":
                    continue
                review_candidates[i][key] = ["non"]*len(review_candidates[i]["sentences"])
        #next we need to determine which of the labelled sentences corresponds to which actual review and store that info into a dictionary
        for number in labels[id]["text"].keys():
            #again, since multiple sentences can map to the same sentence number for each review, 
            #labels[id]["text"][number] is a list
            texts = labels[id]["text"][number]
            for index, text in enumerate(texts):
                #search for the candidates that match the text of the sentence word for word
                candidates = [i for i in range(len(review_candidates)) 
                                if number < len(review_candidates[i]["sentences"]) 
                                and text == review_candidates[i]["sentences"][number]]
                #and then we append number to the list of labelled sentences sorted by review
                for can in candidates:
                    for key in labels[id].keys():
                        if key == "text":
                            continue
                        review_candidates[can][key][number] = labels[id][key][number][index]
        for i in range(len(review_candidates)):
            dataset.append(review_candidates[i])
    with open(f"{data_dir}/raw/revadv/all/revadv.jsonl", "w") as f:
        f.write("\n".join(json.dumps(e) for e in dataset))
    return dataset

def load_revadv_labels(data_dir):
    entries = []
    with open(f"{data_dir}/raw/revadv/all/revadv.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            entry["review_id"] = entry["id"]
            entries.append(entry)
    return entries

def process_revadv(data_dir):
    if(os.path.exists(f"{data_dir}/raw/revadv/all/revadv.jsonl")):
        data = load_revadv_labels(data_dir)
    else:
        print("Parse sentences")
        #load the full reviews of each conference
        reviews = load_revadv_reviews(data_dir)
        #load labelled data using the process_revadv_labels function
        labelled_reviews = preprocess_revadv(data_dir)
        data = process_revadv_labels(reviews, labelled_reviews, data_dir)
    return data

def preprocess_revadv(data_dir):
    reviews = {}
    with open(f"{data_dir}/raw/revadv/asp_train.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            items = obj["identifier"].split("|")
            _, _, review_id, sentence_number = items
            sentence_number = int(sentence_number)
            if(review_id not in reviews):
                reviews[review_id] = {"text": collections.defaultdict(list), "asp": collections.defaultdict(list)}
            reviews[review_id]["text"][sentence_number].append(obj["text"])
            reviews[review_id]["asp"][sentence_number].append(obj["label"])
    with open(f"{data_dir}/raw/revadv/pol_train.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            items = obj["identifier"].split("|")
            _, _, review_id, sentence_number = items
            sentence_number = int(sentence_number)
            if "pol" not in reviews[review_id]:
                reviews[review_id]["pol"] = collections.defaultdict(list)
            reviews[review_id]["pol"][sentence_number].append(obj["label"])
    return reviews
    # for task, examples in lines.items():
    #     with open(f"{data_dir}/labeled/{task}/train/revadv.jsonl", "w") as f:
    #         f.write("\n".join(json.dumps(e) for e in examples))


def prepare_unlabeled_iclr_data(data_dir):
    lines = collections.defaultdict(list)
    for filename in glob.glob(f"{data_dir}/raw/iclr/*.json"):
        with open(filename, "r") as f:
            obj = json.load(f)
            review_id = obj["identifier"]
            identifier_prefix = f"iclr|predict|{review_id}|"
            for i, sent in enumerate(tokenize(obj["text"])):
                for task in "epi pol asp".split():
                    lines[task].append(
                        {
                            "identifier": f"{identifier_prefix}{i}",
                            "text": sent.text,
                            "label": None,
                        }
                    )
    for task, examples in lines.items():
        output_dir = f"{data_dir}/unlabeled/{task}/predict/"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/iclr.jsonl", "w") as f:
            f.write("\n".join(json.dumps(e) for e in examples))

def get_entropy(arr):
    total_entropy = 0
    total = np.sum(arr)
    for el in arr:
        total_entropy += -(el / total) * np.log2(el / total)
    return total_entropy

#we want to maximize the difference in average entropy between the sets
def findOptimal(labels, num_samples, num_classes, task):
    n = len(labels)
    num_examples = [int(n*s) for s in SPLITS]
    total = sum(num_examples)
    best_entropy = 0
    best_splits = []
    best_dists = None
    if(total != n): #remove extra entries from test
        num_examples[0] -= (n - total)
    for i in range(num_samples):
        perm = np.random.permutation(n)
        dists = np.zeros((3,num_classes))
        train_set = [(labels[p], p) for p in perm[0:num_examples[0]]]
        dev_set = [(labels[p], p) for p in perm[num_examples[0]:num_examples[0]+num_examples[1]]]
        test_set = [(labels[p], p) for p in perm[num_examples[1]:num_examples[1]+num_examples[2]]]
        for el in train_set:
            dists[0] += el[0]["counts"]
        for el in dev_set:
            dists[1] += el[0]["counts"]
        for el in test_set:
            dists[2] += el[0]["counts"]
        total_train = get_entropy(dists[0])
        total_dev = get_entropy(dists[1])
        total_test = get_entropy(dists[2])
        if(task != "epi"):
            total_train = get_entropy(dists[0][1:])
            total_dev = get_entropy(dists[1][1:])
            total_test = get_entropy(dists[2][1:])
        entropy = total_train + total_test + total_dev
        if(entropy > best_entropy):
            best_entropy = entropy
            best_splits = [train_set, dev_set, test_set]
            best_dists = dists
    print(best_entropy)
    print(best_dists)
    return best_splits

def convert_to_one_hots(labels, task, num_labels):
    arr = np.zeros((num_labels,), dtype=np.int16)
    CLASS_LABELS = LABELS[task]
    for label in labels:
        arr[CLASS_LABELS[label]] += 1
    return arr

#there are two things that this function does:
#1. given a document where a large amount of sentences don't have labels for the specific task,
#strip down the document into sentences that are labelled
#2. provide surrounding context for each sentence that does have a label

#NOTE: sequences of one or more consecutive labels are considered to be one "larger" document.
#we will need to see how this type of thing impacts performance
def cleanSparseLabels(data, task, context_width = 1):
    d = []
    for i in range(len(data)):
        #j is the index of the sentence
        j = 0
        sentence_index = 1 #more like an entry index than an actual sentence index
        while(j < len(data[i][task])):
            if(data[i][task][j] != "non"): #if the current sentence actually has a label associated with it
                # print(data[i])
                entry = {"sentences": [], task: [], "review_id": data[i]["review_id"] + "||" + str(sentence_index)}
                min_context = max(j-context_width, 0) #prevent array indexing from going out of bounds
                entry["sentences"] += data[i]["sentences"][min_context:j+1]
                entry[task] += data[i][task][min_context:j+1]
                j += 1
                #while the sentences after the current sentence have a label, add them to the current entry
                while(j < len(data[i][task]) and data[i][task][j] != "non"):
                    entry["sentences"].append(data[i]["sentences"][j])
                    entry[task].append(data[i][task][j])
                    j+=1
                #add ending context
                entry["sentences"] += data[i]["sentences"][j:j+context_width]
                entry[task] += data[i][task][j:j+context_width]
                d.append(entry)
                sentence_index += 1
            j += 1
    return d

def filterLabels(data, task):
    d = [entry for entry in data if task in entry]
    return d


def getLabels(data, task, num_classes):
    #filter out examples that have labels for the specific task
    d = filterLabels(data, task)
    if(task == "asp" or task == "pol"): #we don't need to do this for epistemic since there isn't sparse data with these labels
            #removes all of the sentences that don't have labels
            #cuts up the data and returns "mini-documents"
            d = cleanSparseLabels(d, task)

    #convert each new document into one hot vectors
    new_data = []
    for entry in d:
        # el = {"review_id": entry["review_id"]}
        el = {}
        el["counts"] = convert_to_one_hots(entry[task], task, num_classes)
        el["sentences"] = entry["sentences"]
        el[task] = entry[task]
        new_data.append(el)
    return new_data

def findSplits(data, task="epi", num_samples=100, num_classes=3, save=False):
    #reorganize the labels in a way that better fits resampling
    labels = getLabels(data, task, num_classes)
    #find the splits that minimize the average KL divergence
    data = findOptimal(labels, num_samples, num_classes, task)
    entries = ["train", "dev", "test"]
    for i in range(len(entries)):
        for entry in data[i]:
            if type(entry[0]["counts"]) != list:
                entry[0]["counts"] = entry[0]["counts"].tolist()
        with open(f"{entries[i]}_{task}", "w") as f:
            f.write("\n".join(json.dumps(entry[0]) for entry in data[i]))
    return data

def main():

    DATA_DIR = "data"

    print("Preprocessing DISAPERE")
    #focus on disapere first
    #DISAPERE has all three labels: aspect, polarity, and epistemic

    #data should be a list of python dictionaries and should have the following attributes:
    #sentences: list of dictionaries, one for each sentence
    #   each dictionary should have a sentence index and the text of the sentence

    #pol, asp, epi: the sequences of labels corresponding to the document itself
    #(I don't know what rev is)
    disapere = preprocess_disapere(DATA_DIR, write=False, return_data=True)
    # print(data[0]["sentences"][0]["sentence_index"])

    #AMPERE has only epistemic
    print("Preprocessing AMPERE")
    ampere = preprocess_ampere(DATA_DIR)
    # print(ampere[0]["sentences"][0].keys())

    #rev_adv has both aspect and polarity associated with its labels
    print("Preprocessing ReviewAdvisor")
    rev_adv = process_revadv(DATA_DIR)
    #merge datasets together
    full_data = rev_adv + ampere + disapere
    # #find a train test dev split with low entropy
    tasks = {"asp": 8, "epi": 2, "pol": 3}
    for key, value in tasks.items():
        print(f"Finding splits for {key}")
        splits = findSplits(full_data, task=key, num_samples = 1000, num_classes=value)

    

if __name__ == "__main__":
    main()

