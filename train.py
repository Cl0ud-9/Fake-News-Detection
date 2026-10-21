"""Train and evaluate a classifier on WELFake.

    python train.py --data WELFake_Dataset.csv               # Naive Bayes -> models/fake_news_nb.joblib
    python train.py --data WELFake_Dataset.csv --model rf    # random forest, slower and larger
"""
import argparse
import json
import os
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from fake_news import LABELS, build, load_welfake, save
from fake_news.model import ROOT

# A third of the articles are held out, as in the original notebook, stratified so both sides have the same mix
TEST_SIZE = 0.33
SPLIT_SEED = 53


def main():
    parser = argparse.ArgumentParser(description='Train a fake news classifier on WELFake.')
    parser.add_argument('--data', required=True, help='path to WELFake_Dataset.csv')
    parser.add_argument('--model', choices=['nb', 'rf'], default='nb')
    parser.add_argument('--out', help='where to save the model (default: models/fake_news_<model>.joblib)')
    args = parser.parse_args()

    data = load_welfake(args.data)
    train_docs, test_docs, train_labels, test_labels = train_test_split(
        data['doc'], data['label'], test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=data['label'])
    print(f'{len(data)} articles after removing duplicates: {len(train_docs)} to train, {len(test_docs)} held out')

    started = time.time()
    pipeline = build(args.model).fit(train_docs, train_labels)
    predicted = pipeline.predict(test_docs)
    names = [LABELS[c] for c in sorted(LABELS)]
    accuracy = accuracy_score(test_labels, predicted)
    print(f'Trained in {time.time() - started:.0f} s. Held-out accuracy: {accuracy:.1%}\n')
    print(classification_report(test_labels, predicted, target_names=names, digits=3))
    print('Confusion matrix (rows: true, columns: predicted):', names)
    print(confusion_matrix(test_labels, predicted))

    out = args.out or os.path.join(ROOT, 'models', f'fake_news_{args.model}.joblib')
    save(pipeline, out)
    report = classification_report(test_labels, predicted, target_names=names, output_dict=True)
    with open(os.path.splitext(out)[0] + '.metrics.json', 'w') as f:
        json.dump({'model': args.model, 'train_articles': len(train_docs), 'test_articles': len(test_docs),
                   'accuracy': report['accuracy'], 'per_class': {n: report[n] for n in names}}, f, indent=2)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
