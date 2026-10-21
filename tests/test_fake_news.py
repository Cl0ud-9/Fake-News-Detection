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
