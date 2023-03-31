import argparse
import collections
import glob
import json
import os

import classification_lib
from transformers import BertTokenizer

def polarity_exists(sentence):
    return 0 if sentence["polarity"] == "none" else 1

def polarity(sentence):
    return "none" if "polarity" not in sentence else sentence["polarity"]


def review_action(sentence):
    return sentence["review_action"]


def ms_aspect(sentence):
    return "none" if "aspect" not in sentence else sentence["aspect"]


TASK_MAP = {
    "polarity_exists": polarity_exists,
    "review_action": review_action,
    "ms_aspect": ms_aspect,
    'polarity': polarity,
}

parser = argparse.ArgumentParser(description="Extract DISAPERE data")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to main DISAPERE directory (should contain final_dataset/ as a subdirectory)",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to output directory (will be created if necessary)",
)
parser.add_argument(
    "-t",
    "--task",
    choices=TASK_MAP.keys(),
    help="name of the label extractor to apply to each sentence",
)

parser.add_argument(
    "-f",
    "--full_review",
    type=bool
)

def analyze_tokens(docs, tokenizer): 
    tokenizer_fn = lambda tok, text: tok.encode_plus(
        text,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding=False, #THIS IS DIFFERENT. DO NOT TRAIN WITH THIS CODE
        max_length=512,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    #do this in two passes
    #first pass: get the maximum value of tokens per sentence
    max_length = 0
    for value in docs.values():
        sentences = value["sentences"]
        for sentence in sentences:
            encoding = tokenizer_fn(tokenizer, sentence)
            max_length = max(encoding["input_ids"].size()[1], max_length)
    print("Maximum embedding length: {}".format(max_length))

def analyze_documents(docs):
    print("Analyze docs")
    review_init_count = {}
    review_final_count = {}
    context_count = {}
    has_printed = False
    for value in docs.values():
        labels = value["labels"]
        if(not has_printed):
           print(labels)
           has_printed = True
        init = tuple(labels[0])
        if init not in review_init_count:
            review_init_count[init] = 0
        review_init_count[init] += 1
        if len(labels) > 1:
            final = tuple(labels[-1])
            if final not in review_final_count:
                review_final_count[final] = 0
            review_final_count[final] += 1
        for i in range(1, len(labels)):
            labels[i-1], labels[i] = tuple(labels[i-1]), tuple(labels[i])
            if(labels[i-1] not in context_count):
                context_count[labels[i-1]] = {}
            if(labels[i] not in context_count[labels[i-1]]):
                context_count[labels[i-1]][labels[i]] = 0
            context_count[labels[i-1]][labels[i]] += 1
    print(review_init_count)
    print(review_final_count)
    print(context_count)
    print("Total docs: {}".format(len(docs.items())))

def main():
    #tokenizer code
    tokenizer = BertTokenizer.from_pretrained(classification_lib.PRE_TRAINED_MODEL_NAME)
    args = parser.parse_args()

    sentences = collections.defaultdict(list)
    docs = {}
    for subset in "train dev test".split():
        for filename in glob.glob(f"{args.data_dir}/{subset}/*.json"):
            with open(filename, "r") as f:
                obj = json.load(f)
            review_id = obj["metadata"]["review_id"]
            docs[review_id] = {"sentences": [], "labels": []}
            for i, sentence in enumerate(obj["review_sentences"]):
                    identifier = classification_lib.make_identifier(review_id, i)
                    sentences[subset].append(
                        (
                            identifier,
                            sentence["text"],
                            TASK_MAP["ms_aspect"](sentence),
                            TASK_MAP["polarity"](sentence),
                            TASK_MAP["review_action"](sentence)
                        )
                    )
                    docs[review_id]["sentences"].append(sentence["text"])
                    docs[review_id]["labels"].append([TASK_MAP[task](sentence) for task in ("ms_aspect", "polarity", "review_action")])
    analyze_documents(docs)
    analyze_tokens(docs, tokenizer)
    task_output_dir = f"{args.output_dir}/multi_task/"
    os.makedirs(task_output_dir, exist_ok=True)
    label_map1 = list(sorted(set(t[2] for t in sum(sentences.values(), []))))
    label_map2 = list(sorted(set(t[3] for t in sum(sentences.values(), []))))
    label_map3 = list(sorted(set(t[4] for t in sum(sentences.values(), []))))
    with open(f"{task_output_dir}/metadata.json", "w") as f:
        json.dump({"labels": {"ms_aspect": label_map1, "polarity": label_map2, "review_action": label_map3}}, f)
    for subset, subset_sentences in sentences.items():
        print(subset)
        with open(f"{task_output_dir}/{subset}.jsonl", "w") as f:
            for identifier, text, label1, label2, label3 in subset_sentences:
                f.write(
                    json.dumps(
                        {
                            "identifier": identifier,
                            "text": text,
                            "ms_aspect": label_map1.index(label1),
                            "polarity": label_map2.index(label2),
                            "review_action": label_map3.index(label3)
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
