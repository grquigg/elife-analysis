import argparse
import collections
import csv

import who_wins_lib

parser = argparse.ArgumentParser(
    description="Analyze Who Wins results"
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="path to overall data directory",
)

Results = collections.namedtuple("Results",
  "tp tn fp fn".split())

def splerp(labels, true_labels):
  tp = 0
  tn = 0
  fp = 0
  fn = 0

  for label, true_label in zip(labels, true_labels):
    if true_label: 
      if label:
        tp += 1
      else:
        fn += 1
    else:
      if label:
        fp += 1
      else:
        tn += 1

  return Results(tp=tp, tn=tn, fp=fp, fn=fn)


def metrics(results):
  total = sum(results)
  return {
    'accuracy': (results.tp + results.tn)/total,
    'precision': results.tp/(results.tp + results.fp),
    'recall': results.tp/(results.tp + results.fn),
  }


def main():

  args = parser.parse_args()
  config = who_wins_lib.read_config(args.config)

  for source in config.dev:
    with open(f'results/{config.config_name}_dev.csv', 'r') as f:
      reader = csv.DictReader(f)
      labels = []
      true_labels = []
      for row in reader:
        labels.append(int(row['label']))
        true_labels.append(int(row['true_label']))
    results = splerp(labels, true_labels)
    for k, v in metrics(results).items():
      print(k, v)

      

if __name__ == "__main__":
  main()

