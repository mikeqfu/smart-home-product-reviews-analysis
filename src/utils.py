"""
The module provides helper classes/functions for facilitating the implementation of other modules.
"""

import copy
import gc
import glob
import json
import os
import pickle
import pkgutil
import re
import string

import contractions
import numpy as np
import pandas as pd
import pycld2
import unicodedata
from pyhelpers.dbms import PostgreSQL
from pyhelpers.store import _check_loading_path, _check_saving_path, load_data, save_data
from pyhelpers.text import remove_punctuation

from src._cache import EN_SPELL_CHECKER, EN_STOPWORDS, EN_WORDS, LANG_DETECTOR, MOSES_TOKENIZER, \
    SPACY_EN_NLP, US_EN_CHECKER, WORDNET_LEMMATIZER


class CustomerReviewsAnalysis(PostgreSQL):
    """
    Provide a basic `PostgreSQL <https://www.postgresql.org/>`_ instance
    for managing data of the project.

    This class inherits from the class `pyhelpers.dbms.PostgreSQL
    <https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dbms.PostgreSQL.html>`_.
    """

    def __init__(self, host=None, port=None, username=None, password=None,
                 database_name='UoB_CustomerReviewsAnalysis', **kwargs):
        """
        :param host: The database host; defaults to ``None``.
        :type host: str | None
        :param port: The database port; defaults to ``None``.
        :type port: int | None
        :param username: The database username; defaults to ``None``.
        :type username: str | None
        :param password: The database password; defaults to ``None``.
        :type password: str | int | None
        :param database_name: The name of the database; defaults to ``STFC_DAFNI_ClimaTracks``.
        :type database_name: str
        :param kwargs: [Optional] parameters of the class `pyhelpers.dbms.PostgreSQL`_.

        .. _`pyhelpers.dbms.PostgreSQL`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/pyhelpers.dbms.PostgreSQL.html

        **Examples**::

            >>> from src.utils import CustomerReviewsAnalysis
            >>> db_instance = CustomerReviewsAnalysis()
            Password (postgres@localhost:5432): ***
            Connecting postgres:***@localhost:5432/postgres ... Successfully.
            >>> db_instance.database_name
            'UoB_CustomerReviewsAnalysis'
            >>> # Remote server
            >>> db_instance = CustomerReviewsAnalysis()
            >>> db_instance.database_name
            'UoB_CustomerReviewsAnalysis'
        """

        credentials = {
            'host': host,
            'port': port,
            'username': username,
            'password': password,
            'database_name': database_name,
        }

        if host not in {'localhost', '127.0.0.1'}:
            try:  # Load credentials from the .credentials file
                credentials = json.loads(pkgutil.get_data(__name__, "data/.credentials").decode())
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        kwargs.update(credentials)
        super().__init__(**kwargs)


def _normalise_text(x):
    normalised_text = ''

    for x_ in x:
        if 120458 <= ord(x_) <= 120483:
            normalised_text += chr(ord(x_) - 120361)
        elif 120432 <= ord(x_) <= 120457:
            normalised_text += chr(ord(x_) - 120367)
        else:
            normalised_text += x_

    return normalised_text


def normalise_text(x):
    """
    Normalise textual data.

    :param x: textual data
    :type x: str
    :return: normalised text
    :rtype: str

    **Examples**::

        >>> from src.utils import normalise_text

        >>> # noinspection SpellCheckingInspection
        >>> normalise_text('Ítem last a whole 20 minutes')
        'Item last a whole 20 minutes'

        >>> normalise_text('2000 ft. ²')
        '2000 ft 2'
    """

    x1 = unicodedata.normalize('NFKD', x)
    x2 = re.sub(r'[^\x00-\x7f]', '', x1).strip()

    y = remove_punctuation(re.sub(r'[^\w\s]', ' ', x2), rm_whitespace=True)
    # y = ' '.join(x2.split())

    if y.lower() in {'na', 'n a'}:
        y = ''

    return y


def correct_typo(x):
    """
    Correct misspelled words and/or typos.

    :param x: textual data
    :type x: str
    :return: text with misspelled words being corrected
    :rtype: str

    **Examples**::

        >>> from src.utils import correct_typo

        >>> # noinspection SpellCheckingInspection
        >>> correct_typo('We shoud replce letter A with frut apple')
        'we should replace letter a with fruit apple'
    """

    x_ = [EN_SPELL_CHECKER.correction(w).lower() for w in re.compile(r'\w+').findall(x)]

    y = ' '.join(x_)

    return y


def identify_language(x):
    """
    Identify language of textual data.

    :param x: textual data
    :type x: str
    :return: full name of a language
    :rtype: str or None

    **Examples**::

        >>> from src.utils import identify_language

        >>> identify_language('This is about Amazon product reviews.')
        'English'

        >>> # noinspection SpellCheckingInspection
        >>> identify_language('Se trata de las reseñas de productos de Amazon.')
        'Spanish'

        >>> identify_language('+-*/')  # None
        'Unknown'
    """

    y = normalise_text(x)
    if y == '':
        language = 'Unknown'

    else:
        is_reliable, _, details = pycld2.detect(remove_punctuation(x, rm_whitespace=True))
        lang = details[0][0]  # lang, lang_code, percent = details[0][:3]

        # if is_reliable:  # and percent >= 90:
        if lang == 'INTERLINGUE':
            language = 'English'
        else:
            language = lang.title()

        # else:
        #     try:  # Use googletrans
        #         GOOGLE_TRANSLATOR.client.headers.update(fake_requests_headers())
        #         detected = GOOGLE_TRANSLATOR.detect(x)
        #
        #         # noinspection PyProtectedMember
        #         if detected._response.is_error:
        #             language = FASTTEXT_LANGUAGE_DETECTOR.identify_language(x)
        #         else:
        #             lang = detected.lang
        #             language = LANGUAGE_CODES[lang[0] if isinstance(lang, list) else lang].title()
        #
        #     except Exception:
        #         language = None

    return language


def is_english_word(x):
    """
    Check whether a given word is English.

    :param x: textual data of a word
    :type x: str
    :return: whether ``x`` is an English word
    :rtype: bool

    **Examples**::

        >>> from src.utils import is_english_word

        >>> is_english_word(x='apple')
        True

        >>> is_english_word(x='apples')
        True

        >>> is_english_word(x='xyz')
        False
    """

    if x.lower() in EN_WORDS or US_EN_CHECKER.check(x):
        return True
    else:
        return False


def _is_english(x):
    y = remove_punctuation(re.sub(r'[^\w\s]', ' ', x), rm_whitespace=True)

    if all(is_english_word(z) for z in y.split()):
        return True
    else:
        return False


def is_english(x):
    """
    Check whether a given piece of textual data is written in English.

    :param x: textual data
    :type x: str
    :return: whether ``x`` is written in English
    :rtype: bool

    **Examples**::

        >>> from src.utils import is_english

        >>> is_english(x='This is about Amazon product reviews.')
        True

        >>> # noinspection SpellCheckingInspection
        >>> is_english(x='ESe trata de las reseñas de productos de Amazon.')
        False
    """

    y1, y2 = normalise_text(x), remove_punctuation(re.sub(r'[^\w\s]', ' ', x), rm_whitespace=True)
    if y1 == '':
        is_en = False

    else:
        is_reliable, _, details = pycld2.detect(y2)
        lang = details[0][0]

        if is_reliable:
            if lang == 'ENGLISH':
                is_en = True
            elif lang == 'INTERLINGUE':
                if _is_english(x):
                    is_en = True
                else:
                    is_en = 'tbc'  # i.e. To be confirmed
            else:
                is_en = False

        else:
            language = LANG_DETECTOR.identify_language(y2)
            if language == 'English' and all(is_english_word(z) for z in y2.split()):
                is_en = True

            else:
                is_en = None  # i.e. To be confirmed

    return is_en


def remove_stopwords(x):
    """
    Remove stop words from textual data.

    :param x: textual data
    :type x: str
    :return: text without stopwords
    :rtype: str

    **Examples**::

        >>> from src.utils import remove_stopwords

        >>> remove_stopwords('This is an apple.')
        'apple.'

        >>> remove_stopwords('There were some apples.')
        'apples.'

        >>> remove_stopwords("I'm going to school.")
        "I'm going school."
    """

    # x_ = nltk.tokenize.word_tokenize(x)

    y = ' '.join(filter(lambda word: word.lower() not in EN_STOPWORDS, x.split()))

    return y


def remove_single_letters(x):
    """
    Remove all single letters from textual data.

    :param x: textual data
    :type x: str
    :return: text without single letters
    :rtype: str

    **Examples**::

        >>> from src.utils import remove_single_letters
        >>> from pyhelpers.text import remove_punctuation

        >>> remove_single_letters('There is a bug.')
        'There is bug.'

        >>> remove_single_letters('It is a b c.')
        'It is c.'

        >>> remove_single_letters(remove_punctuation('It is a b c.'))
        'It is'
    """

    # y = ' '.join([w for w in x.split() if len(w) > 1])
    y = ' '.join(filter(lambda w: len(w) > 1, x.split()))

    return y


def remove_digits(x):
    """
    Remove digits from textual data.

    :param x: textual data
    :type x: str
    :return: text without digits
    :rtype: str

    **Examples**::

        >>> from src.utils import remove_digits

        >>> remove_digits('There are 2 bugs.')
        'There are bugs.'

        >>> remove_digits("Hello world! 666")
        'Hello world!'

        >>> remove_digits("Hello world! 666!")
        'Hello world! !'
    """

    x.translate({ord(k): None for k in string.digits})

    y = ' '.join(x.translate(str.maketrans('', '', string.digits)).split())

    return y


def _lemmatize_text(x, pos=None, detokenized=True):
    """
    Lemmatize textual data.

    :param x: textual data
    :type x: str
    :param pos: ``pos`` parameter of the method `nltk.stem.WordNetLemmatizer.lemmatize()`_
    :type pos: str or None or list
    :param detokenized: whether to detokenize the texts, defaults to ``True``
    :type detokenized: bool
    :return: text without tense
    :rtype: str

    .. _`nltk.stem.WordNetLemmatizer.lemmatize()`:
        https://www.nltk.org/api/nltk.stem.wordnet.html#nltk.stem.wordnet.WordNetLemmatizer.lemmatize

    **Examples**::

        >>> from src.utils import lemmatize_text

        >>> _lemmatize_text('This is an apple.')
        'This be an apple'

        >>> _lemmatize_text('There were some apples.')
        'There be some apple'

        >>> _lemmatize_text("I'm going to school.")
        'I be go to school'
    """

    if pos is None:
        pos_ = ['n', 'v', 'a', 'r', 's']
    else:
        pos_ = [pos] if isinstance(pos, str) else pos

    tokens = []
    for word in MOSES_TOKENIZER.tokenize(contractions.fix(x.lower())):
        for p in pos_:
            word = WORDNET_LEMMATIZER.lemmatize(word, pos=p)
        tokens.append(word)

    if detokenized:
        y = remove_punctuation(" ".join(tokens))
    else:
        y = tokens

    return y


def lemmatize_text(x, allowed_postags=None, detokenized=True):
    """
    Lemmatize textual data.

    See also https://spacy.io/api/annotation.

    :param x: textual data
    :type x: str
    :param allowed_postags: allowed postags, defaults to ``None``
    :type allowed_postags: list or None
    :param detokenized: whether to detokenize the texts, defaults to ``True``
    :type detokenized: bool
    :return: lemmatized textual data
    :rtype: str

    **Examples**::

        >>> from src.utils import lemmatize_text

        >>> lemmatize_text('This is an apple.')
        'apple'

        >>> lemmatize_text('There were some apples.')
        'be apple'

        >>> lemmatize_text("I'm going to school.")
        'go school'
    """

    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']

    text_ = [token.lemma_ for token in SPACY_EN_NLP(x) if token.pos_ in allowed_postags]

    if detokenized:
        # text_ = " ".join(filter(lambda word: word.lower() not in EN_STOPWORDS, text_))
        text_ = " ".join(text_)

    return text_


def split_large_pickle(path_to_pkl, dest_dir=None, number_of_chunks=5, verbose=True):
    """
    Split a very large pickle file into smaller chunks and save them to separate files.

    :param path_to_pkl: pathname of a pickle file
    :type path_to_pkl: str
    :param dest_dir: pathname of a directory where the pickle partitions are to be saved,
        defaults to ``None``
    :type dest_dir: str or None
    :param number_of_chunks: number of chunks/partitions, defaults to ``5``
    :type number_of_chunks: int
    :param verbose: whether to print relevant information in console, defaults to ``False``
    :type verbose: bool or int
    :return: pathnames of the partitions
    :rtype: list
    """

    pathname_, ext = os.path.splitext(path_to_pkl)

    if dest_dir is None:
        dest_dir_ = pathname_
    else:
        dest_dir_ = copy.copy(dest_dir)
    os.makedirs(dest_dir_, exist_ok=True)

    file_size = os.path.getsize(path_to_pkl)
    write_size = file_size // number_of_chunks

    raw_pkl = open(path_to_pkl, 'rb')
    part_num = 0
    pathnames = []

    if verbose:
        print(f"Splitting \"{os.path.relpath(pathname_)}\" into {number_of_chunks} parts ... ")

    while True:
        chunk = raw_pkl.readline(write_size)  # Read a portion of the input file

        if not chunk:
            break

        part_num_ = str(part_num).zfill(len(str(number_of_chunks)))
        if verbose:
            print(f"\t\"part{part_num_}\"", end=" ... ")

        pathname = os.path.join(dest_dir_, f"part{part_num_}{ext}")

        with open(pathname, 'wb') as part_output:
            pickle.dump(chunk, part_output, protocol=pickle.HIGHEST_PROTOCOL)

        pathnames.append(pathname)

        del chunk
        gc.collect()

        if verbose:
            print("Done.")

        part_num += 1

    raw_pkl.close()

    if verbose:
        print("Completed.")

    return pathnames


def join_pickle_parts(part_pkl_pathnames):
    """
    Join partitions of a pickle file.

    :param part_pkl_pathnames: pathnames of partitions
    :type part_pkl_pathnames: list
    :return: data
    :rtype: typing.Any
    """

    dest_file = os.path.commonpath(part_pkl_pathnames) + "_new.pkl"

    output_file = open(dest_file, 'wb')

    for part_pkl_file in part_pkl_pathnames:

        # Open the part
        input_file = open(part_pkl_file, 'rb')

        while True:
            # Read all bytes of the part
            part_pkl_bytes = input_file.read()

            # Break out of loop if we are at end of file
            if not part_pkl_bytes:
                break

            pickle.dump(part_pkl_bytes, output_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Close the input file
        input_file.close()

    # Close the output file
    output_file.close()


def save_partitioned_df(data, path_to_file, number_of_chunks=5, verbose=False, **kwargs):
    """
    Split (a very large) dataframe into smaller partitions and save the partitions to separate files.

    :param data: dataframe
    :type data: pandas.DataFrame
    :param path_to_file: pathname of a file,
        or pathname of a directory where partitioned data files are to be saved
    :type path_to_file: str or os.PathLike[str]
    :param number_of_chunks: number of chunks/partitions, defaults to ``5``
    :type number_of_chunks: int
    :param verbose: whether to print relevant information in console, defaults to ``False``
    :type verbose: bool or int

    .. seealso::

        - Examples of the method
          :meth:`LatentDirichletAllocation.make_evaluation_summary()
          <src.modeller.LatentDirichletAllocation.make_evaluation_summary>`.
    """

    pathname_, ext = os.path.splitext(path_to_file)

    _check_saving_path(path_to_file, verbose=verbose, print_end=" (partitioned) ... ")

    try:
        df_parts = np.array_split(ary=data, indices_or_sections=number_of_chunks)

        if ext in {"pickle", "pkl"}:
            kwargs.update({'protocol': pickle.HIGHEST_PROTOCOL})

        for i, df_part in zip(range(1, number_of_chunks + 1), df_parts):
            path_to_file_ = os.path.join(pathname_, f"part{str(i).zfill(2)}.pkl")
            save_data(df_part, path_to_file=path_to_file_, **kwargs)

        if verbose:
            print("Done.")

    except Exception as e:
        print(f"Failed. {e}.")


def load_partitioned_df(path_to_file, verbose=False, **kwargs):
    """
    Load partitions of data and concatenate them into one dataframe.

    :param path_to_file: pathname of a directory where partitioned data files are saved,
        or pathname of a target file
    :type path_to_file: str
    :param verbose: whether to print relevant information in console, defaults to ``False``
    :type verbose: bool or int
    :return: dataframe concatenated from partitions
    :rtype: pandas.DataFrame

    .. seealso::

        - Examples of the methods
          :meth:`LatentDirichletAllocation.make_evaluation_summary()
          <src.modeller.LatentDirichletAllocation.make_evaluation_summary>` and
          :meth:`LatentDirichletAllocation.fetch_evaluation_summary()
          <src.modeller.LatentDirichletAllocation.fetch_evaluation_summary>`.
    """

    pathname_, ext = os.path.splitext(path_to_file)

    part_pathnames = glob.glob(os.path.join(pathname_, f"part*{ext}"))

    if len(part_pathnames) > 0:
        _check_loading_path(path_to_file, verbose=verbose, print_end=" (partitioned) ... ")

        try:
            if ext in {"pickle", "pkl"}:
                kwargs.update({'protocol': pickle.HIGHEST_PROTOCOL})

            data = pd.concat([load_data(p, **kwargs) for p in part_pathnames], ignore_index=True)

            gc.collect()

            if verbose:
                print("Done.")

            return data

        except Exception as e:
            print(f"Failed. {e}.")

    else:
        if verbose:
            if os.path.isdir(pathname_):
                print(f"No data is available at the specified path \"{os.path.relpath(pathname_)}\".")
            else:
                print("Check if `path_to_file` is correctly specified.")
