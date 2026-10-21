# Notebooks

The original 2024 notebooks, kept as they were run, outputs included.

| Notebook | Data | What it does |
|---|---|---|
| [welfake_naive_bayes_random_forest.ipynb](welfake_naive_bayes_random_forest.ipynb) | [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) | Explores the data with word clouds and length histograms, then trains Naive Bayes and a random forest on word counts. |
| [lstm_kaggle_fake_news.ipynb](lstm_kaggle_fake_news.ipynb) | [Kaggle Fake News](https://www.kaggle.com/c/fake-news) | Cleans and lemmatises the text with spaCy, then trains an LSTM on one-hot word indices. Reaches 89% accuracy on 4,572 test articles, against 99% on the training set. |

Two things to know about the WELFake notebook, both fixed in `train.py`:

- Cell 39 replaces the test set with a single article to try the model, so the evaluation after it fails and the random
  forest cells were never run. The saved `model.pkl` is empty for the same reason.
- `class_names = ['fake', 'real']` and the ratio cell treat label 0 as fake. In WELFake, 0 is real: see the
  [README](../README.md#the-data).

The WELFake notebook was written for Google Colab, and the LSTM notebook expects `train.csv` and `test.csv` from the
competition in the same folder. They need more than the app: seaborn, wordcloud, nltk, spaCy with
`en_core_web_sm`, and TensorFlow.
