"""Paste an article, see whether it reads like real or fake news, and which words decided it.

Run with `streamlit run app.py`.
"""
import html
import os
import re

import streamlit as st

from fake_news import explain, load, predict
from fake_news.model import DEFAULT_MODEL

EXAMPLES = {
    'A wire-service report': ('Appeals court to weigh rehearing of Trump travel ban (Reuters) - A U.S. federal appeals '
                              'court on Friday asked the Justice Department and the state of Washington to submit '
                              'briefs on whether a larger panel of judges should reconsider its ruling.'),
    'A clickbait post': ('BREAKING: Hillary caught in SHOCKING scandal, mainstream media REFUSES to report it! '
                         'Share this before it gets deleted!'),
}
# The words CountVectorizer counts: two or more letters or digits
TOKEN = re.compile(r'(?u)\b\w\w+\b')
FAKE_COLOUR, REAL_COLOUR = '214, 48, 49', '39, 174, 96'


@st.cache_resource
def model():
    return load(os.environ.get('FAKE_NEWS_MODEL', DEFAULT_MODEL))


def highlighted(article, weights):
    """The article as HTML, each deciding word tinted by how hard it pushed (red: fake, green: real)."""
    strongest = max((abs(w) for w in weights.values()), default=1) or 1
    out, last = [], 0
    for match in TOKEN.finditer(article):
        out.append(html.escape(article[last:match.start()]))
        word = match.group()
        weight = weights.get(word.lower())
        if weight is None:
            out.append(html.escape(word))
        else:
            colour = FAKE_COLOUR if weight > 0 else REAL_COLOUR
            alpha = 0.18 + 0.6 * abs(weight) / strongest
            out.append(f'<mark style="background: rgba({colour}, {alpha:.2f}); color: inherit; border-radius: 4px; '
                       f'padding: 0 2px">{html.escape(word)}</mark>')
        last = match.end()
    out.append(html.escape(article[last:]))
    return ('<div style="line-height: 1.7; padding: 1rem; border: 1px solid rgba(128,128,128,.3); '
            'border-radius: 8px">' + ''.join(out) + '</div>')


def bars(title, words, strongest):
    st.markdown(f'**{title}**')
    for word, weight in words[:6]:
        st.progress(min(weight / strongest, 1.0), text=f'{word}  ·  {weight:.2f}')
    if not words:
        st.caption('Nothing in this article.')


st.set_page_config(page_title='Fake News Detection', page_icon='📰')
st.title('📰 Fake News Detection')
st.caption('Naive Bayes on word counts, trained on about 42,000 articles from the WELFake dataset. '
           'It judges writing style and vocabulary, not whether the facts are true.')

choice = st.selectbox('Start from an example, or paste your own', ['Paste my own'] + list(EXAMPLES))
article = st.text_area('Headline and article', value=EXAMPLES.get(choice, ''), height=220)

if st.button('Check', type='primary', disabled=not article.strip()):
    pipeline = model()
    label, confidence = predict(pipeline, [article])[0]
    if label == 'fake':
        st.error(f'Reads like **fake** news (model confidence {confidence:.1%})')
    else:
        st.success(f'Reads like **real** news (model confidence {confidence:.1%})')
    st.caption('Naive Bayes is usually overconfident: a high number does not mean the answer is certain.')

    reasons = explain(pipeline, article, top=12)
    if reasons:
        st.subheader('Why')
        st.markdown(highlighted(article, dict(reasons)), unsafe_allow_html=True)
        strongest = max(abs(weight) for _, weight in reasons)
        left, right = st.columns(2)
        with left:
            bars('Pushed towards fake', [(w, x) for w, x in reasons if x > 0], strongest)
        with right:
            bars('Pushed towards real', [(w, -x) for w, x in reasons if x < 0], strongest)
        st.caption('Each weight is how far the word moved the decision, in log-odds: the number of times it appears '
                   'times how much more often fake (or real) articles in the training data used it. Common English '
                   'words are not counted.')
