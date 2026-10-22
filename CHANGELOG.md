# Changelog

What changed in each version of Fake News Detection.

The format follows [Keep a Changelog](https://keepachangelog.com), and the versions follow
[Semantic Versioning](https://semver.org).

## [1.1.0] - 2026-10-22

### Added

- Explanations: the app tints each deciding word in the article, red for fake and green for real, and ranks the
  strongest on each side; `predict.py --explain` lists them. `fake_news.explain()` returns each word's exact share of
  the Naive Bayes decision, and a test checks that the shares add up to the model's log-odds.

## [1.0.1] - 2026-10-22

### Changed

- Python 3.14 is supported: the tests pass on it, and continuous integration now runs on Python 3.11 to 3.14.

## [1.0.0] - 2026-10-21

The first public version: the two 2024 projects, turned into a tested package with an app.

### Added

- `fake_news`, a small package: loading WELFake, the Naive Bayes and random forest pipelines, saving and loading.
- `train.py`, which trains either model and measures it on a third of the articles held out; `predict.py`, a command
  line; and `app.py`, a Streamlit app.
- The trained Naive Bayes model (89.1% held-out accuracy), and the random forest (92.0%) as a release download.
- Tests, continuous integration, and the security, contributing and conduct guides.

### Fixed

- The labels were backwards in places. WELFake's own description says 0 is fake, but its articles show 0 is real
  (21,266 of 21,282 Reuters articles are 0). The code now uses the data's meaning throughout.
- The notebook replaced its test set with one article before evaluating, so the evaluation failed and the random
  forest was never measured; its saved `model.pkl` was empty. `train.py` measures both on the held-out third.
- 8,203 articles in WELFake are repeats. They are now removed before splitting, so no article is both trained
  on and tested on.

## [0.1.0] - 2024-05-08

The college projects, kept in [notebooks/](notebooks): Naive Bayes and a random forest on WELFake, and an LSTM on the
Kaggle Fake News competition data (89% test accuracy).
