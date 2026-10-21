"""The classifiers: word counts, then Naive Bayes or a random forest."""
import os

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data import LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, 'models', 'fake_news_nb.joblib')


def build(kind='nb'):
    """An untrained pipeline. 'nb' is small and fast; 'rf' is more accurate and much larger."""
    if kind == 'nb':
        classifier = MultinomialNB()
    elif kind == 'rf':
        classifier = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    else:
        raise ValueError(f"Unknown model kind {kind!r}: use 'nb' or 'rf'")
    return Pipeline([('counts', CountVectorizer(stop_words='english')), ('classifier', classifier)])


def save(pipeline, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'sklearn': sklearn.__version__}, path, compress=3)


def load(path=DEFAULT_MODEL):
    """Load a model saved by save(). Only load files you trust: joblib files can run code."""
    bundle = joblib.load(path)
    if bundle['sklearn'] != sklearn.__version__:
        raise RuntimeError(f"{path} was saved with scikit-learn {bundle['sklearn']}, but {sklearn.__version__} is "
                           'installed. Install the version in requirements.txt, or retrain with train.py.')
    return bundle['pipeline']


def predict(pipeline, documents):
    """[(label, probability of that label)] for each document."""
    results = []
    for probs in pipeline.predict_proba(documents):
        index = int(probs.argmax())
        results.append((LABELS[int(pipeline.classes_[index])], float(probs[index])))
    return results
