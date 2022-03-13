import collections
import joblib
import pickle
import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

import elife_lib

examples = collections.defaultdict(list)
labels = collections.defaultdict(list)

for split in "train dev test".split():

    with open(f"disapere/featurized_by_polarity_{split}.pkl", 'rb') as f:
        featurized_by_polarity = pickle.load(f)
    dependency_paths = []    
    for label, sentences in featurized_by_polarity.items():
        for sentence in tqdm.tqdm(sentences):
            examples[split].append(collections.Counter(elife_lib.extract_dep_paths(sentence)))
            labels[split].append(label)

pipe = Pipeline(
    [('vectorizer', DictVectorizer(sparse=False)),
     ('clf', LogisticRegression(random_state=0, max_iter=1500))])
pipe.fit(examples['train'], labels['train'])
print("Train accuracy", pipe.score(examples['train'], labels['train']))
print("Dev accuracy", pipe.score(examples['dev'], labels['dev']))
joblib.dump(pipe, "ckpt/polarity_classifier.joblib")