import pandas as pd
import os
import glob
import csv
import stanza
import pprint
from tqdm import tqdm
from google.cloud import bigquery
from google.cloud import storage
import argparse


pp = pprint.PrettyPrinter(width=100, compact=True)

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-fp",
    "--file_path",
    type=str,
    help="full file path to save labels",
)
parser.add_argument(
    "-nr",
    "--n_reviews",
    type=str,
    help="n reviews to randomly sample",
)
parser.add_argument(
    "-ns",
    "--n_sents",
    type=str,
    help="n sentences to label by hand",
)
parser.add_argument(
    "-ft",
    "--first_time",
    type=str,
    help="first time?",
)


# Initialize google clients
BQ_CLIENT = bigquery.Client()
STORAGE_CLIENT = storage.Client()

# Initialize stanza pipeline
SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")

# DISAPERE Labels
ARGS = [
    "arg_EVALUATIVE",
    "arg_REQUEST",
    "arg_FACT",
    "arg_STRUCTURING",
    "arg_SOCIAL",
    "arg_OTHER",
]

ASPS = [
    "asp_MOTIVATION-IMPACT",
    "asp_ORIGINALITY",
    "asp_SOUNDNESS-CORRECTNESS",
    "asp_SUBSTANCE",
    "asp_REPLICABILITY",
    "asp_MEANINGFUL-COMPARISON",
    "asp_CLARITY",
    "asp_OTHER",
]

REQS = [
    "req_EDIT",
    "req_TYPO",
    "req_EXPERIMENT"
]

STRS = [
    "struc_SUMMARY",
    "struc_HEADING",
    "struc_QUOTE"
]

ALL = ARGS+ASPS+REQS+STRS 

PATH = "/home/jupyter/00_daniel/00_reviews/00_data/"


def summon_reviews(n_reviews):
    """
    Returns a pandas DF containing n sampled eLife reviews
    by summoning reviews from Google BQ.
    """

    REVIEW_QRY = """
    SELECT Manuscript_no_, review_id, rating_hat, Major_comments,
    FROM `gse-nero-dmcfarla-mimir`.eLife.eLife_Reviews_IDRating
    """
    print("Getting data from BQ...")

    # df = BQ_CLIENT.query(REVIEW_QRY)
    # print(type(df))
    # df = pd.DataFrame(df).dropna()
    df = BQ_CLIENT.query(REVIEW_QRY).result().to_dataframe()
    df = df.dropna()
    return df.sample(n_reviews, random_state=72)


def _make_identifier(review_id, index):
    return f"{review_id}|||{index}"


def get_sentences_df(df):
    """
    Tokenizes review sentence and
    Returns a df where row is sentence.
    """
    sentence_dicts = []
    for i, row in tqdm(df.iterrows()):
        review_id = row["review_id"]
        raw_text = row["Major_comments"]
        ms_id = row["Manuscript_no_"]
        for i, sentence in enumerate(SENTENCIZE_PIPELINE(raw_text).sentences):
            sentence_dct = {
                    "manuscript_no": ms_id,
                    "review_id": review_id,
                    "identifier": _make_identifier(review_id, i),
                    "text": sentence.text,
                }
            sentence_dct.update(dict.fromkeys([all.lower() for all in ALL], int(0)))
            sentence_dicts.append(sentence_dct)
    return pd.DataFrame.from_dict(sentence_dicts)


def label_sentences(sentences_df, n_sents, first_time, file_path):
    sentences_df = sentences_df.iloc[:n_sents]
    

    mode = "a"
    if first_time == "True": 
        mode = 'w' 

    with open(file_path, mode=mode) as f:
        writer = csv.DictWriter(f, sentences_df.columns)
        if mode == 'w':
            writer.writeheader()
        
        n_sentences = 0
        for _, sentence_dct in sentences_df.iterrows():
            sentence_dct = sentence_dct.to_dict()
            n_sentences += 1

            # Get identifying info
            mid = sentence_dct["manuscript_no"]
            rid = sentence_dct["review_id"]
            sid = sentence_dct["identifier"]

            # Print Sentence and its identifiers
            print()
            print("-" * 100)
            print(f"SENTENCE {n_sentences} OF {sentences_df.shape[0]} SENTENCES TO RATE")
            print(f"M_ID: {mid}\tR_ID: {rid}\tS_ID: {sid}")
            print("-" * 50)
            pp.pprint(f"{sentence_dct['text']}")
            print("-" * 100)



            print("\n\tSelect the action of this sentence:")
            for arg in ARGS:
                value = int(input(f"\t\t{arg}: "))
                sentence_dct[arg.lower()] = value
            

            if sentence_dct["arg_request"] == 1:

                # Get fine grained req
                print("\n\tSelect what this sentence requests:")
                for req in REQS:
                    value = int(input(f"\t\t{req}: "))
                    sentence_dct[req.lower()] = value

                # Get aspect when req
                print(
                    "\n\tSelect the aspect of the manuscript that is the subject of this sentence's request:"
                )
                for asp in ASPS:
                    value = int(input(f"\t\t{asp}: "))
                    sentence_dct[asp.lower()] = value
            

            elif sentence_dct["arg_structuring"] == 1:
                
                # Get fine grained structure
                print("\n\tSelect the kind of structuring of this sentence:")
                for struc in STRS:
                    value = int(input(f"\t\t{struc}: "))
                    sentence_dct[struc.lower()] = value
                        
            elif sentence_dct["arg_evaluative"] == 1:

                # Get aspect when eval
                print(
                    "\n\tSelect the aspect of the manuscript that this sentence evaluates:"
                )
                for asp in ASPS:
                    value = int(input(f"\t\t{asp}: "))
                    sentence_dct[asp.lower()] = value     


            writer.writerow(sentence_dct) 


def hello():
    print("\n"*3)
    print("+" * 33)
    print("START INTERACTIVE CODING SESSION!")
    print("+" * 33)


def goodbye():
    print()
    print("+" * 33)
    print("END INTERACTIVE CODING SESSION!")
    print("+" * 33)
    print("\n"*3)


def main(n_reviews, n_sents, first_time, file_path):

    # Get data
    sentences_df = summon_reviews(n_reviews)
    sentences_df = get_sentences_df(sentences_df)
    
    # Begin
    hello()

    # first time means new file
    if first_time == "True":
        label_sentences(sentences_df, n_sents, first_time, file_path)

    # nth time means append file, and make sure only unrated reviews are printed
    if first_time == "False":

        # open existing rated reviews file
        rater_df = pd.read_csv(file_path)
        already_reviewed = list(rater_df["identifier"])
        sentences_df = sentences_df[~ sentences_df["identifier"].isin(already_reviewed)]
        label_sentences(sentences_df, n_sents, first_time, file_path)
    
    # End
    goodbye()


if __name__ == "__main__":
    args = parser.parse_args()
    main(int(args.n_reviews),
         int(args.n_sents),
         args.first_time.capitalize(), 
         args.file_path)
