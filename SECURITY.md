# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: open the [Security tab](https://github.com/Cl0ud-9/Fake-News-Detection/security)
and choose **Report a vulnerability**, or go straight to
[the advisory form](https://github.com/Cl0ud-9/Fake-News-Detection/security/advisories/new). Only the maintainer sees
what you send.

Please include what you found, how to reproduce it, and what you think an attacker could do with it. The maintainer
aims to answer within a week, and you will be credited in the fix unless you would rather not be.

## What counts

- A way for text typed into the app or given to `predict.py` to run code or read files.
- A way to make the app load a model file it was not pointed at.

Not in scope: problems in scikit-learn, joblib or Streamlit themselves (report those upstream), and model files you
chose to load yourself.

## Keeping yourself safe

Models are saved with joblib, which can run code when a file is loaded. Only load models from this repository, from
its GitHub releases, or ones you trained yourself with `train.py`. `FAKE_NEWS_MODEL` and `--model` point the app and
`predict.py` at another file: use them only with files you trust.

## Supported versions

Only the current `main` gets fixes.
