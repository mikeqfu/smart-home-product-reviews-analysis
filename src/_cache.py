"""
Cached functions and constants.
"""

import os
import pathlib
import re
import string

import enchant
import nltk.corpus
import nltk.stem
import sacremoses
import spacy
import spellchecker
from pyhelpers.store import load_json, load_pickle, save_pickle

# WordNet English words
EN_WORDS_WORDNET = [x.lower().replace('_', ' ') for x in nltk.corpus.wordnet.words(lang='eng')]

# English words
EN_WORDS = set(nltk.corpus.words.words() + EN_WORDS_WORDNET)

# English (US) spelling checker (pyenchant)
US_EN_CHECKER = enchant.Dict(tag='en_US')

# English spelling checker (pyspellchecker)
EN_SPELL_CHECKER = spellchecker.SpellChecker()

# Language codes
# noinspection SpellCheckingInspection
LANG_CODES = {
    'af': 'Afrikaans',
    'sq': 'Albanian',
    'am': 'Amharic',
    'ar': 'Arabic',
    'hy': 'Armenian',
    'az': 'Azerbaijani',
    'eu': 'Basque',
    'be': 'Belarusian',
    'bn': 'Bengali',
    'bs': 'Bosnian',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'ceb': 'Cebuano',
    'ny': 'Chichewa',
    'zh-cn': 'Chinese (Simplified)',
    'co': 'Corsican',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'eo': 'Esperanto',
    'et': 'Estonian',
    'tl': 'Filipino',
    'fi': 'Finnish',
    'fr': 'French',
    'fy': 'Frisian',
    'gl': 'Galician',
    'ka': 'Georgian',
    'de': 'German',
    'el': 'Greek',
    'gu': 'Gujarati',
    'ht': 'Haitian Creole',
    'ha': 'Hausa',
    'haw': 'Hawaiian',
    'iw': 'Hebrew',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hmn': 'Hmong',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'ig': 'Igbo',
    'id': 'Indonesian',
    'ga': 'Irish',
    'it': 'Italian',
    'ja': 'Japanese',
    'jw': 'Javanese',
    'kn': 'Kannada',
    'kk': 'Kazakh',
    'km': 'Khmer',
    'ko': 'Korean',
    'ku': 'Kurdish (Kurmanji)',
    'ky': 'Kyrgyz',
    'lo': 'Lao',
    'la': 'Latin',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'lb': 'Luxembourgish',
    'mk': 'Macedonian',
    'mg': 'Malagasy',
    'ms': 'Malay',
    'ml': 'Malayalam',
    'mt': 'Maltese',
    'mi': 'Maori',
    'mr': 'Marathi',
    'mn': 'Mongolian',
    'my': 'Myanmar (Burmese)',
    'ne': 'Nepali',
    'no': 'Norwegian',
    'or': 'Odia',
    'ps': 'Pashto',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'pa': 'Punjabi',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sm': 'Samoan',
    'gd': 'Scots Gaelic',
    'sr': 'Serbian',
    'st': 'Sesotho',
    'sn': 'Shona',
    'sd': 'Sindhi',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'so': 'Somali',
    'es': 'Spanish',
    'su': 'Sundanese',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'tg': 'Tajik',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'ug': 'Uyghur',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'cy': 'Welsh',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'zu': 'Zulu',
}


class EnDetector:

    def __init__(self):
        self.data_pathname = pathlib.Path("data\\misc\\lid.176.bin")

        import fasttext

        try:
            self.model = fasttext.FastText._FastText(model_path=str(self.data_pathname))
        except ValueError:
            self.model = fasttext.FastText._FastText(model_path="..\\..\\" + str(self.data_pathname))

    def identify_language(self, x):
        prediction = self.model.predict(x, k=1)
        language_code = prediction[0][0].replace('__label__', '')

        try:
            language = LANG_CODES[language_code]
        except KeyError:
            language = 'unknown'

        return language

    def update_trained_model_file(self, verbose=False):
        from pyhelpers.ops import download_file_from_url

        url_ = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/'
        download_file_from_url(
            url=url_ + self.data_pathname.name, path_to_file=self.data_pathname, verbose=verbose)

        ftz_filename = self.data_pathname.name.replace(".bin", ".ftz")
        ftz_url = url_ + ftz_filename
        download_file_from_url(
            url=ftz_url, path_to_file=self.data_pathname.with_name(ftz_filename), verbose=verbose)


LANG_DETECTOR = EnDetector()


def get_english_stopwords(subjoin_spacy=False, subjoin_sklearn=False, update=False):
    # "src\\data\\eng_stopwords.pkl"
    path_to_pickle = pathlib.Path(__file__).parent / "data" / "eng_stopwords.pkl"

    if os.path.exists(path_to_pickle) and not update:
        eng_stopwords = load_pickle(path_to_pickle)

    else:
        eng_stopwords = set(nltk.corpus.stopwords.words('english')).difference(
            {"empty", "enough", "full", "various"})

        if subjoin_spacy:
            spacy_en = spacy.load('en_core_web_sm')
            eng_stopwords = eng_stopwords.union(spacy_en.Defaults.stop_words)

        if subjoin_sklearn:
            from sklearn.feature_extraction import text
            eng_stopwords = eng_stopwords.union(text.ENGLISH_STOP_WORDS)

        custom_stopwords_pathname = pathlib.Path(__file__).parent / "data" / "adhoc_stopwords.json"
        eng_stopwords.update(set(load_json(custom_stopwords_pathname).keys()))

        if update:
            save_pickle(eng_stopwords, path_to_pickle)

    return eng_stopwords


# English stopwords
EN_STOPWORDS = get_english_stopwords(subjoin_spacy=False, subjoin_sklearn=False)

# Hex codes to characters
HEX_TO_CHAR = {
    "\xa0": "",
    "\x80": "€",
    "\x84": "\"",
    "\x85": "...",
    "\x91": "'",
    "\x92": "'",
    "\x93": "\"",
    "\x94": "\"",
    "\x95": "_",
    "\x96": " – ",
    "\x97": " – ",
    "¡": "!",
    "½": "1/2"
}

# Replacement for contractions (Source: https://stackoverflow.com/questions/43018030/)
# noinspection SpellCheckingInspection
CONTRACTIONS = {
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}


def get_typo_corrections(regex=False):
    # Known typos
    # pathname = "src\\data\\typo_corrections.json"
    typo_corrections = load_json(pathlib.Path(__file__).parent / "data" / "typo_corrections.json")

    if regex:
        typo_corrections = {
            re.compile(r' ?%s[ %s]' % (k, string.punctuation), re.IGNORECASE): f' {v} '
            for k, v in typo_corrections.items()}
        typo_corrections.update({re.compile(r' ?hm(m)\1+', re.IGNORECASE): ''})
        typo_corrections.update({re.compile(r' ?bo(o)\1+', re.IGNORECASE): ''})

    return typo_corrections


# Typo corrections
TYPO_CORRECTIONS = get_typo_corrections()

# An instance of the class
# `nltk.stem.WordNetLemmatizer <https://www.nltk.org/_modules/nltk/stem/wordnet.html>`_
# that is used for lemmatizing a word (namely, converts a word to its base form).
WORDNET_LEMMATIZER = nltk.stem.WordNetLemmatizer()

# SpaCy English NLP model
SPACY_EN_NLP = spacy.load('en_core_web_sm')

# sacremoses
MOSES_TOKENIZER = sacremoses.MosesTokenizer(lang='en')
MOSES_DETOKENIZER = sacremoses.MosesDetokenizer(lang='en')
