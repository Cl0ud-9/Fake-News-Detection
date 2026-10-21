"""Fake news detection on WELFake with word counts and Naive Bayes or a random forest."""
from .data import LABELS, document, load_welfake
from .model import build, load, predict, save

__all__ = ['LABELS', 'document', 'load_welfake', 'build', 'load', 'predict', 'save']
__version__ = '1.0.1'
