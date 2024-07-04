"""Test the module :py:mod:`~src.utils`."""

import pytest


def test_normalise_text():
    from src.utils import normalise_text

    # noinspection SpellCheckingInspection
    rslt = normalise_text('Ítem last a whole 20 minutes')
    assert rslt == 'Item last a whole 20 minutes'

    rslt = normalise_text('2000 ft. ²')
    assert rslt == '2000 ft 2'


def test_correct_typo():
    from src.utils import correct_typo

    # noinspection SpellCheckingInspection
    rslt = correct_typo('We shoud replce letter A with frut apple')
    assert rslt == 'we should replace letter a with fruit apple'


def test_identify_language():
    from src.utils import identify_language

    rslt = identify_language('This is about Amazon product reviews.')
    assert rslt == 'English'

    # noinspection SpellCheckingInspection
    rslt = identify_language('Se trata de las reseñas de productos de Amazon.')
    assert rslt == 'Spanish'

    rslt = identify_language('+-*/')  # None
    assert rslt == 'Unknown'


def test_is_english_word():
    from src.utils import is_english_word

    assert is_english_word(x='apple')
    assert is_english_word(x='apples')
    assert not is_english_word(x='xyz')


def test_is_english():
    from src.utils import is_english

    assert is_english(x='This is about Amazon product reviews.')

    # noinspection SpellCheckingInspection
    assert not is_english(x='ESe trata de las reseñas de productos de Amazon.')


def test_remove_stopwords():
    from src.utils import remove_stopwords

    rslt = remove_stopwords('This is an apple.')
    assert rslt == 'apple.'

    rslt = remove_stopwords('There were some apples.')
    assert rslt == 'apples.'

    rslt = remove_stopwords("I'm going to school.")
    assert rslt == "I'm going school."


def test_remove_single_letters():
    from src.utils import remove_single_letters
    from pyhelpers.text import remove_punctuation

    rslt = remove_single_letters('There is a bug.')
    assert rslt == 'There is bug.'

    rslt = remove_single_letters('It is a b c.')
    assert rslt == 'It is c.'

    rslt = remove_single_letters(remove_punctuation('It is a b c.'))
    assert rslt == 'It is'


def test_remove_digits():
    from src.utils import remove_digits

    rslt = remove_digits('There are 2 bugs.')
    assert rslt == 'There are bugs.'

    rslt = remove_digits("Hello world! 666")
    assert rslt == 'Hello world!'

    rslt = remove_digits("Hello world! 666!")
    assert rslt == 'Hello world! !'


def test_lemmatize_text():
    from src.utils import lemmatize_text

    rslt = lemmatize_text('This is an apple.')
    assert rslt == 'apple'

    rslt = lemmatize_text('There were some apples.')
    assert rslt == 'be apple'

    rslt = lemmatize_text("I'm going to school.")
    assert rslt == 'go school'


if __name__ == '__main__':
    pytest.main()
