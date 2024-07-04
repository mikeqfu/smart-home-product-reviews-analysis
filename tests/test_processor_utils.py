import datetime

import pytest
from pyhelpers.store import load_pickle

from src.processor.utils import *


def test_parse_review_date():
    rslt1 = parse_review_date('December 6, 2020')
    assert rslt1 == [datetime.datetime(2020, 12, 6, 0, 0), 'United States']
    rslt2 = parse_review_date('Reviewed in the United Kingdom on August 20, 2020')
    assert rslt2 == [datetime.datetime(2020, 8, 20, 0, 0), 'United Kingdom']


def test_regulate_people_found_helpful():
    assert regulate_people_found_helpful('One') == 1
    assert regulate_people_found_helpful('') == 0


def test_correct_identified_typos():
    # noinspection SpellCheckingInspection
    original_review_text = 'I would like to replce it.'
    rslt1 = correct_identified_typos(original_review_text)
    assert rslt1 == ['I', 'would', 'like', 'to', 'replace', 'it', '.']
    rslt2 = correct_identified_typos(original_review_text, split=False)
    assert rslt2 == 'I would like to replace it.'


def test_get_vader_sentiment_score():
    example_review_text = load_pickle("tests\\data\\example_review_text.pkl")

    x1 = example_review_text.values[0]
    rslt1 = get_vader_sentiment_score(x1)
    assert set(rslt1.keys()) == {'vs_neg_score', 'vs_neu_score', 'vs_pos_score', 'vs_compound_score'}

    x2 = lemmatize_text(x1)
    rslt2 = get_vader_sentiment_score(x2)
    assert set(rslt2.keys()) == {'vs_neg_score', 'vs_neu_score', 'vs_pos_score', 'vs_compound_score'}

    x3 = remove_stopwords(x1)
    rslt3 = get_vader_sentiment_score(x3)
    assert set(rslt3.keys()) == {'vs_neg_score', 'vs_neu_score', 'vs_pos_score', 'vs_compound_score'}

    rslt4 = get_vader_sentiment_score(example_review_text)
    assert isinstance(rslt4, pd.DataFrame)


def test_determine_sentiment_on_rating():
    assert determine_sentiment_on_rating(rating=5) == 'positive'
    assert determine_sentiment_on_rating(rating=4) == 'positive'
    assert determine_sentiment_on_rating(rating=3) == 'neutral'
    assert determine_sentiment_on_rating(rating=2) == 'negative'
    assert determine_sentiment_on_rating(rating=1) == 'negative'


def test_determine_sentiment_on_vs_score():
    example_review_text = load_pickle("tests\\data\\example_review_text.pkl")
    example_vs_scores = get_vader_sentiment_score(example_review_text)
    assert isinstance(example_vs_scores, pd.DataFrame)
    rslt = example_vs_scores['vs_compound_score'].map(determine_sentiment_on_vs_score)
    assert rslt.values[0] == 'positive'


if __name__ == '__main__':
    pytest.main()
