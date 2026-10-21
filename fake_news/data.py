"""Loading the WELFake dataset as (document, label) pairs."""
import pandas as pd

# WELFake's own description says 0 = fake and 1 = real, but the articles say the opposite: 21,266 of the 21,282
# articles bylined "(Reuters)" are labelled 0, and all 1,272 from the conspiracy site 21st Century Wire are labelled 1.
LABELS = {0: 'real', 1: 'fake'}


def document(title, text):
    """The model reads the headline and the body together."""
    return f'{title} {text}'.strip()


def load_welfake(path):
    """Articles with both a title and a body, duplicates removed, as a DataFrame with 'doc' and 'label'."""
    df = pd.read_csv(path, usecols=['title', 'text', 'label'])
    df = df.dropna(subset=['title', 'text'])
    df = df[(df['title'].str.strip() != '') & (df['text'].str.strip() != '')]
    df = df.assign(doc=[document(t, x) for t, x in zip(df['title'], df['text'])])
    # 8,203 articles are repeats; keeping both would put copies on both sides of the split
    return df.drop_duplicates('doc')[['doc', 'label']].reset_index(drop=True)
