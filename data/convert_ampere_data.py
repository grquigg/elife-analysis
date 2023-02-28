import argparse
import collections
import glob
from interval import Interval
import json
import os
import sys
import stanza
import tqdm


parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--input_dir', default='', type=str, help='')
parser.add_argument('-o', '--output_dir', default='', type=str, help='')

SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")
TOLERANCE = 7

label_map = {
    "non-arg": "arg_other",
    "evaluation": "arg_evaluative",
    "request": "arg_request",
    "fact": "arg_fact",
    "reference": "arg_fact",
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


def build_dict(review_id, sentence_idx, sentence_text, label):
  if 'quote' in label:
    labels = {
 "review_action": 'arg_structuring',
            "fine_review_action": "arg-structuring_quote",
    }
  else:
    labels = {
 "review_action": label_map[label],
            "fine_review_action": "none",
    }

  labels.update({
    "sentence_index": sentence_idx,
    "review_id": review_id,
    "text": sentence_text,
           "aspect": "none",
            "polarity": "none",
        })
  return labels


def main():
    args = parser.parse_args()


    for filename in glob.glob(f'{args.input_dir}/*.txt'):

      review_id, _, rating = filename.split(
          '/')[-1].rsplit(".", 1)[0].rsplit("_", 2)

      with open(filename, 'r') as f:
        sentence_dicts = []
        for i, line in enumerate(f):
          label, sentence = line.strip().split('\t', 1)
          sentence_dicts.append(build_dict(review_id, i, sentence, label))

      os.makedirs(args.output_dir, exist_ok=True)
      with open(f'{args.output_dir}/{review_id}.json', "w") as f:
          json.dump(
              {
                  "metadata": {
                      "review_id": review_id,
                      "rating": int(rating),
                  },
                  "review_sentences": sentence_dicts,
              },
              f,
          )

if __name__ == "__main__":
    main()
