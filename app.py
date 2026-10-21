"""Paste an article, see whether it reads like real or fake news. Run with `streamlit run app.py`."""
import os

import streamlit as st

from fake_news import load, predict
from fake_news.model import DEFAULT_MODEL

EXAMPLES = {
    'A wire-service report': ('Appeals court to weigh rehearing of Trump travel ban (Reuters) - A U.S. federal appeals '
                              'court on Friday asked the Justice Department and the state of Washington to submit '
                              'briefs on whether a larger panel of judges should reconsider its ruling.'),
    'A clickbait post': ('BREAKING: Hillary caught in SHOCKING scandal, mainstream media REFUSES to report it! '
                         'Share this before it gets deleted!'),
}


@st.cache_resource
def model():
    return load(os.environ.get('FAKE_NEWS_MODEL', DEFAULT_MODEL))


st.set_page_config(page_title='Fake News Detection', page_icon='📰')
st.title('📰 Fake News Detection')
st.caption('Naive Bayes on word counts, trained on about 42,000 articles from the WELFake dataset. '
           'It judges writing style and vocabulary, not whether the facts are true.')

choice = st.selectbox('Start from an example, or paste your own', ['Paste my own'] + list(EXAMPLES))
article = st.text_area('Headline and article', value=EXAMPLES.get(choice, ''), height=220)

if st.button('Check', type='primary', disabled=not article.strip()):
    label, confidence = predict(model(), [article])[0]
    if label == 'fake':
        st.error(f'Reads like **fake** news (model confidence {confidence:.1%})')
    else:
        st.success(f'Reads like **real** news (model confidence {confidence:.1%})')
    st.caption('Naive Bayes is usually overconfident: a high number does not mean the answer is certain.')
