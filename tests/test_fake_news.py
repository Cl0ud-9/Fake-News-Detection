import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from fake_news import build, document, load, load_welfake, predict, save

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REAL = ['WASHINGTON (Reuters) - The Senate passed the budget bill on Tuesday, officials said.',
        'LONDON (Reuters) - The central bank held interest rates steady on Thursday, a spokesman said.',
        'BEIJING (Reuters) - Exports rose in March, government data showed on Monday.']
FAKE = ['SHOCKING! You will not believe what they are hiding from you, share before it is deleted!',
        'BREAKING: the elites are lying again, wake up people, the truth is finally out!',
        'EXPOSED: what the mainstream media refuses to tell you about the scandal!']


@pytest.fixture
def tiny_csv(tmp_path):
    rows = [(t.split(' - ')[0], t, 0) for t in REAL] + [(t.split('!')[0], t, 1) for t in FAKE]
    rows += [rows[0], ('', 'no title', 0), ('No body', None, 1)]
    path = tmp_path / 'welfake.csv'
    pd.DataFrame(rows, columns=['title', 'text', 'label']).to_csv(path)
    return path


def test_loading_drops_empty_rows_and_duplicates(tiny_csv):
    data = load_welfake(tiny_csv)
    assert len(data) == len(REAL) + len(FAKE)
    assert list(data.columns) == ['doc', 'label']
    assert data['doc'][0] == document(REAL[0].split(' - ')[0], REAL[0])


@pytest.mark.parametrize('kind', ['nb', 'rf'])
def test_a_trained_model_round_trips_and_names_its_labels(tiny_csv, tmp_path, kind):
    data = load_welfake(tiny_csv)
    pipeline = build(kind).fit(data['doc'], data['label'])
    save(pipeline, tmp_path / 'model.joblib')
    label, confidence = predict(load(tmp_path / 'model.joblib'), ['SHOCKING truth they are hiding, share now!'])[0]
    assert label == 'fake'
    assert 0.5 <= confidence <= 1


def test_unknown_model_kind_is_an_error():
    with pytest.raises(ValueError):
        build('svm')


def test_a_model_from_another_scikit_learn_version_is_refused(tmp_path):
    import joblib

    joblib.dump({'pipeline': None, 'sklearn': '0.0.1'}, tmp_path / 'old.joblib')
    with pytest.raises(RuntimeError, match='scikit-learn'):
        load(tmp_path / 'old.joblib')


def test_the_shipped_model_classifies_held_out_articles():
    with open(os.path.join(HERE, 'held_out_examples.json'), encoding='utf-8') as f:
        examples = json.load(f)
    model = load()
    for example in examples:
        assert predict(model, [example['text']])[0][0] == example['label']


def test_command_line():
    result = subprocess.run([sys.executable, 'predict.py', REAL[0]], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith(('real', 'fake'))
    assert subprocess.run([sys.executable, 'predict.py'], cwd=ROOT, capture_output=True).returncode != 0


def test_explain_names_the_words_that_decided_it(tiny_csv):
    from fake_news import explain

    data = load_welfake(tiny_csv)
    pipeline = build('nb').fit(data['doc'], data['label'])
    article = 'SHOCKING news from Reuters, share before it is deleted'
    reasons = explain(pipeline, article, top=5)
    words = dict(reasons)
    assert set(words) <= {'shocking', 'news', 'reuters', 'share', 'deleted'}
    assert words['shocking'] > 0 and words['reuters'] < 0
    assert [abs(w) for _, w in reasons] == sorted((abs(w) for _, w in reasons), reverse=True)


def test_explain_adds_up_to_the_model_log_odds(tiny_csv):
    import numpy as np

    from fake_news import explain

    data = load_welfake(tiny_csv)
    pipeline = build('nb').fit(data['doc'], data['label'])
    article = 'BREAKING the truth they hide, says Reuters'
    reasons = explain(pipeline, article, top=100)
    classifier = pipeline.named_steps['classifier']
    log_probs = pipeline.predict_log_proba([article])[0]
    fake, real = list(classifier.classes_).index(1), list(classifier.classes_).index(0)
    prior = classifier.class_log_prior_[fake] - classifier.class_log_prior_[real]
    assert np.isclose(prior + sum(w for _, w in reasons), log_probs[fake] - log_probs[real])


def test_explain_is_empty_for_models_that_cannot_say(tiny_csv):
    from fake_news import explain

    data = load_welfake(tiny_csv)
    pipeline = build('rf').fit(data['doc'], data['label'])
    assert explain(pipeline, 'SHOCKING truth') == []
    assert explain(build('nb').fit(data['doc'], data['label']), 'the and of') == []


def test_command_line_explains_on_request():
    result = subprocess.run([sys.executable, 'predict.py', '--explain', FAKE[0]], cwd=ROOT, capture_output=True,
                            text=True)
    assert result.returncode == 0
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith(('fake', 'real'))
    assert len(lines) > 1 and all(line.split()[1] in ('fake', 'real') for line in lines[1:])
