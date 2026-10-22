"""Classify articles from the command line.

    python predict.py "Headline. Body of the article..."
    python predict.py --file article.txt
    python predict.py --explain "..."      also list the words that pushed the answer
"""
import argparse
import sys

from fake_news import explain, load, predict
from fake_news.model import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description='Say whether a news article looks real or fake.')
    parser.add_argument('text', nargs='*', help='the article (headline and body)')
    parser.add_argument('--file', help='read the article from a text file instead')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='a model saved by train.py')
    parser.add_argument('--explain', action='store_true', help='list the words that pushed the answer hardest')
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding='utf-8') as f:
            article = f.read()
    else:
        article = ' '.join(args.text)
    if not article.strip():
        parser.error('give the article as text or with --file')

    pipeline = load(args.model)
    label, probability = predict(pipeline, [article])[0]
    print(f'{label} (model confidence {probability:.1%})')
    if args.explain:
        reasons = explain(pipeline, article)
        if not reasons:
            print('(only the Naive Bayes model can say which words decided it)')
        for word, weight in reasons:
            print(f"  {word:<20} {'fake' if weight > 0 else 'real':<5} {abs(weight):6.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
