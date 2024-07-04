"""
A module that implements `latent Dirichlet allocation (LDA)
<https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_ for topic modelling.
"""

import collections
import copy
import datetime
import functools
import gc
import itertools
import os
import pickle
import shutil
import string
import time

import gensim
import gensim.corpora
import gensim.models
import gensim.utils
import matplotlib.ticker
import nltk.corpus
import numpy as np
import pandas as pd
import seaborn as sns
from pyhelpers._cache import _print_failure_msg
from pyhelpers.dirs import cd
from pyhelpers.ops import confirmed
from pyhelpers.settings import mpl_preferences
from pyhelpers.store import _check_saving_path, load_data, save_data, save_spreadsheets

from src._cache import MOSES_TOKENIZER
from src.modeller._base import _Base
from src.utils import save_partitioned_df


class LatentDirichletAllocation(_Base):
    """
    A class for instantiating LDA (Latent Dirichlet Allocation) model for the review texts.
    """

    #: Name of the model.
    NAME: str = 'Latent Dirichlet Allocation (LDA)'

    def __init__(self, product_category, product_type, sentiment_on='dual_scale',
                 review_column_name='review_text', random_state=0, **kwargs):
        """
        :param product_category: Product category.
        :type product_category: str
        :param product_type: product type, valid values include ``{'Robotic', 'Traditional'}``
        :type product_type: str
        :param sentiment_on: column name for the metric on which sentiment is determined,
            defaults to ``'dual_scale'``
        :type sentiment_on: str
        :param review_column_name: column name of the review texts;
            when ``review_column_name=None`` (default), it defaults to ``'review_text'``
        :type review_column_name: str or None
        :param random_state: random seed number
        :type random_state: int or None
        :param kwargs: [optional] parameters of the class :class:`~src.modeller._Base`

        :ivar list min_counts: A list of ``min_count`` for model evaluation.
        :ivar list thresholds: A list of ``threshold`` for model evaluation.
        :ivar numpy.ndarray corpus_proportions: An array of corpus proportions for model evaluation.
        :ivar range pos_topic_numbers:
            A range of topic numbers for model evaluation on positive reviews.
        :ivar list pos_alphas: A list of ``alpha`` for model evaluation on positive reviews.
        :ivar list pos_etas: A list of ``eta`` for model evaluation on positive reviews.
        :ivar range neg_topic_numbers:
            A range of topic numbers for model evaluation on negative reviews.
        :ivar list neg_alphas: A list of ``alpha`` for model evaluation on negative reviews.
        :ivar list neg_etas: A list of ``eta`` for model evaluation on negative reviews.

        :ivar str or None sentiment: Label of sentiment.
        :ivar list pos_tokenized_docs: Tokenized documents of positive reviews.
        :ivar list neg_tokenized_docs: Tokenized documents of negative reviews.
        :ivar list neu_tokenized_docs: Tokenized documents of neutral reviews.
        :ivar dict tokenized_docs: Data of tokenized documents.

        :ivar pandas.DataFrame pos_eval_summary:
            A summary of model evaluation results for positive reviews.
        :ivar pandas.DataFrame neg_eval_summary:
            A summary of model evaluation results for negative reviews.
        :ivar pandas.DataFrame neu_eval_summary:
            A summary of model evaluation results for neutral reviews.
        :ivar dict eval_summary: All summaries of model evaluation results.
        :ivar pandas.DataFrame or None eval_summary_:
            The summary of model evaluation for the given ``sentiment``.

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> lda_robovac = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> lda_robovac.VALID_SENTIMENT_LABELS
            {'negative', 'neutral', 'positive'}
            >>> lda_robovac.reviews.preprocd_data.shape
            (77775, 18)
            >>> lda_tradvac = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> lda_tradvac.VALID_SENTIMENT_LABELS
            {'negative', 'neutral', 'positive'}
            >>> lda_tradvac.reviews.preprocd_data.shape
            (110978, 18)
            >>> lda_smtherms = LatentDirichletAllocation('thermostats', product_type='smart')
            >>> lda_smtherms.VALID_SENTIMENT_LABELS
            {'negative', 'neutral', 'positive'}
            >>> lda_smtherms.reviews.preprocd_data.shape
            (26285, 18)
        """

        super().__init__(
            product_category=product_category, product_type=product_type, sentiment_on=sentiment_on,
            review_column_name=review_column_name, random_state=random_state, **kwargs)

        self.min_counts = [1, 5]  # range(1, 6)
        self.thresholds = [10e-5, 1]  # range(10e-5, 1, 10e-1)
        self.corpus_proportions = np.arange(0.75, 1.0, 0.05)

        self.pos_topic_numbers = range(3, 16)  # Number of topics
        self.pos_alphas = ['symmetric', 'asymmetric', 'auto']  # Document-topic density
        self.pos_etas = [0.01, 0.5, 1.0, 'symmetric', 'auto']  # Word-topic density

        self.neg_topic_numbers = range(2, 11)
        self.neg_alphas = ['symmetric', 'asymmetric', 'auto']
        self.neg_etas = [0.01, 0.5, 1.0, 'symmetric', 'auto']

        self.sentiment = None

        self.pos_tokenized_docs = None
        self.neg_tokenized_docs = None
        self.neu_tokenized_docs = None
        self.tokenized_docs = {
            'positive': self.pos_tokenized_docs,
            'negative': self.neg_tokenized_docs,
            'neutral': self.neu_tokenized_docs}

        self.pos_eval_summary, self.neg_eval_summary, self.neu_eval_summary = map(
            self.fetch_evaluation_summary, ['positive', 'negative', 'neutral'])
        self.eval_summary = {
            'positive': self.pos_eval_summary,
            'negative': self.neg_eval_summary,
            'neutral': self.neu_eval_summary}
        self.eval_summary_ = None

    def cd_models(self, *args, **kwargs):
        # noinspection PyShadowingNames
        """
        Change to the directory where the models and their relevant files are saved.

        :return: pathname of the directory for storing models
        :rtype: str

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> import os
            >>> lda = LatentDirichletAllocation('vacuum', 'robotic', load_preprocd_data=False)
            >>> os.path.relpath(lda.cd_models())
            'data\\amazon_reviews\\vacuum_cleaners\\robotic\\models\\lda'
            >>> lda = LatentDirichletAllocation('vacuum', 'traditional', load_preprocd_data=False)
            >>> os.path.relpath(lda.cd_models())
            'data\\amazon_reviews\\vacuum_cleaners\\traditional\\models\\lda'
            >>> lda = LatentDirichletAllocation('therms', 'smart', load_preprocd_data=False)
            >>> os.path.relpath(lda.cd_models())
            'data\\amazon_reviews\\thermostats\\smart\\models\\lda'
        """

        # pathname = cdd(f"robot_vacuum_cleaners\\amazon_reviews\\models\\lda", *args, **kwargs)
        pathname = super().cd_models("lda", *args, **kwargs)

        return pathname

    @classmethod
    def get_tokens(cls, doc, bespoke_stopwords=None):
        # noinspection PyShadowingNames
        """
        Get tokens of a given document.

        :param doc: any document
        :type doc: str
        :param bespoke_stopwords: a set of bespoke stopwords, defaults to ``None``
        :type bespoke_stopwords: set or list or tuple or None
        :return: tokens of the given ``doc``
        :rtype: list

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> example_doc = lda.data['review_text'][0]
            >>> example_doc_tokens = lda.get_tokens(example_doc, bespoke_stopwords=None)
            >>> isinstance(example_doc_tokens, list)
            True
            >>> # Traditional vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> example_doc = lda.data['review_text'][0]
            >>> example_doc_tokens = lda.get_tokens(example_doc, bespoke_stopwords=None)
            >>> isinstance(example_doc_tokens, list)
            True
        """

        # noinspection SpellCheckingInspection
        x = doc.replace('dollar dollar dollar', 'dollar').replace('dollar dollar', 'dollar').replace(
            'zoozee home', '').replace('light n easy', '').replace('pure clean', '')

        y = MOSES_TOKENIZER.tokenize(x)

        b_stopwords = set() if bespoke_stopwords is None else bespoke_stopwords
        if len(b_stopwords) > 0:
            y = [token for token in y if token not in b_stopwords]

        return y

    @classmethod
    def specify_adhoc_stopwords(cls):
        # noinspection PyShadowingNames
        """
        Create a set of ad-hoc stopwords.

        :return: Ad-hoc stopwords.
        :rtype: set

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', 'robotic', load_preprocd_data=False)
            >>> rslt = lda.specify_adhoc_stopwords()
            >>> isinstance(rslt, set)
            True
            >>> # Traditional vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', 'traditional', load_preprocd_data=False)
            >>> rslt = lda.specify_adhoc_stopwords()
            >>> isinstance(rslt, set)
            True
        """

        adhoc_stopwords = set(string.ascii_lowercase)

        # noinspection SpellCheckingInspection
        names_ = {
            'asimov', 'agador', 'andrae', 'baskin', 'dooney', 'gobbe', 'kiki', 'lautaro', 'mcclean',
            'murph', 'tanja', 'tilley', 'viktor', 'zac', 'zeke', 'zorg', 'zue', 'zoila', 'ralphie',
        }
        english_names = set(x.lower() for x in nltk.corpus.names.words()).union(names_)
        adhoc_stopwords.update(english_names)

        # noinspection SpellCheckingInspection
        brands_and_prod = {
            '2nice', 'famree', 'airrobo', 'amarey', 'amdieu', 'amrobt', 'aquabot', 'bagotte',
            'bissell', 'bobsweep', 'braava', 'chunkui', 'clobot', 'coayu', 'coredy', 'deenkee',
            'dopinmin', 'dreametech', 'dser', 'ecovacs', 'enther', 'eufy', 'eureka', 'experobot',
            'ffsign', 'fouramz', 'g00vi', 'generic', 'gokoco', 'gooovi', 'goovi', 'goovii',
            'gttvo robot', 'honiture', 'ihome', 'ilife', 'imartine', 'imass', 'irobot', 'joybros',
            'kenmore', 'laresar', 'lefant', 'lhhting', 'lrkq', 'vac', 'sew', 'narwal', 'noisz',
            'okp', 'orrhomi', 'ot', 'qomotop', 'paylesshere', 'proscenic', 'realme', 'robit',
            'roborock', 'roubow', 'samsung', 'satily', 'serenelife', 'shark', 'shellbot', 'sqfzll',
            'sysperl', 'syvio', 'tab', 'tesvor', 'thamtu', 'tikom', 'trifo', 'twotoo', 'uoni',
            'victony', 'victonyus', 'will smith', 'yeedi', 'yuntuo', 'zoozee', 'ozmo', 'robovac',
            'scooba', 'rvc', 'roomba', 'scooby', 'roomella', 'iroomba', 'botvac', 'rhoomba',
            'goove', 'rhomba',
        }
        adhoc_stopwords.update(brands_and_prod)

        # noinspection SpellCheckingInspection
        adhoc_stopwords_ = {
            'absolutely', 'also', 'always', 'amazon', 'apparent', 'avae', 'bbi', 'be', 'bim',
            'bopi', 'certainly', 'definitely', 'especially', 'even', 'eventually', 'first', 'floor',
            'fre', 'frivvy', 'get', 'go', 'however', 'https', 'i', 'it', 'l', 'little', 'lot', 'm',
            'make', 'much', 'n', 'need', 'nood', 'often', 'one', 'pacman', 'particularly', 'ppx',
            're', 'real', 'really', 'ref', 'robo', 'robot', 's', 'since', 'something', 'te',
            'thing', 'think', 'title', 'today', 'too', 'utf', 'vacuum', 'work', 'would', 'z',
            'robovaccum', 'seem', 'come', 'run', 'use', 'give', 'take', 'product', 'purchase',
            'buy', 'put', 'instead', 'good', 'well', 'reason'
        }

        return adhoc_stopwords.union(adhoc_stopwords_)

    def get_tokenized_docs(self, docs, sentiment, bespoke_stopwords=None):
        # noinspection PyShadowingNames
        """
        Get tokenized documents.

        :param docs: any documents
        :type docs: typing.Iterable
        :param sentiment: label of sentiment;
            options are :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`
        :type sentiment: str
        :param bespoke_stopwords: a set of bespoke stopwords
        :type bespoke_stopwords: set
        :return: tokenized documents
        :rtype: list

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> example_docs = lda.data['review_text']
            >>> pos_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='positive')
            >>> isinstance(pos_tokenized_docs, list)
            True
            >>> neg_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='negative')
            >>> isinstance(neg_tokenized_docs, list)
            True
            >>> # Traditional vacuum cleaners
            >>> lda = LatentDirichletAllocation('traditional')
            >>> example_docs = lda.data['review_text']
            >>> pos_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='positive')
            >>> isinstance(pos_tokenized_docs, list)
            True
            >>> neg_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='negative')
            >>> isinstance(neg_tokenized_docs, list)
            True
        """

        adhoc_stopwords = self.specify_adhoc_stopwords()

        if sentiment.lower() in {'pos', 'positive'}:  # bespoke stopwords for positive reviews
            sentiment_ = 'positive'
            # noinspection SpellCheckingInspection
            temp = {
                'amaze', 'amazing', 'awesome', 'better', 'bravo', 'excellent', 'fantastic',
                'farfegnugen', 'fine', 'glad', 'great', 'happy', 'impressed', 'jeeve', 'like',
                'love', 'nice', 'okay', 'perfect', 'surprisingly', 'yeete', 'godsend',
            }

        elif sentiment.lower() in {'neg', 'negative'}:  # bespoke stopwords for negative reviews
            sentiment_ = 'negative'
            # noinspection SpellCheckingInspection
            temp = {
                'bad', 'awful', 'issue', 'problem', 'frustration', 'frustrating',
                'frustrated', 'anger', 'angry', 'fear', 'sad', 'upset', 'shocked', 'annoyed',
                'disgusted', 'disgusting', 'disappointed', 'disappointing', 'disappoint',
                'miserable', 'terrible', 'horrible', 'disatified', 'disatifion', 'useless'
            }

        else:
            sentiment_ = 'neutral'
            temp = set()

        adhoc_stopwords.update(temp)

        if bespoke_stopwords is not None:
            adhoc_stopwords.update(set(bespoke_stopwords))

        tokenized_docs = [
            self.get_tokens(doc, bespoke_stopwords=set(adhoc_stopwords))
            for _, doc in enumerate(docs)]

        self.tokenized_docs[sentiment_] = tokenized_docs
        setattr(self, f'{sentiment_[:3]}_tokenized_docs', tokenized_docs)

        return tokenized_docs

    @staticmethod
    def _make_bi_grams(docs, bi_gram_model):
        return [bi_gram_model[doc] for doc in docs]

    @staticmethod
    def _make_tri_grams(docs, bi_gram_model, tri_gram_model):
        return [tri_gram_model[bi_gram_model[doc]] for doc in docs]

    def make_corpus(self, tokenized_docs, ngram=2, min_count=1, threshold=10e-5, scoring='npmi'):
        # noinspection PyShadowingNames
        """
        Make a corpus.

        :param tokenized_docs: tokenized documents
        :type tokenized_docs: list
        :param ngram: number of grams
        :type ngram: int
        :param min_count: ``min_count`` of the class `gensim.models.phrases.Phrases()`_,
            defaults to ``1``
        :type min_count: int
        :param threshold: ``threshold`` of the class `gensim.models.phrases.Phrases()`_,
            defaults to ``10e-5``
        :type threshold: float
        :param scoring: ``scoring`` of the class `gensim.models.phrases.Phrases()`_,
            defaults to ``'npmi'``
        :type scoring: str
        :return: corpus (i.e. term-document frequency, see `gensim.corpora.Dictionary.doc2bow()`_),
            id-word mapping dictionary (see `gensim.corpora.Dictionary()`_), and
            lemmatized review texts
        :rtype: tuple

        .. _`gensim.models.phrases.Phrases()`:
            https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
        .. _`gensim.corpora.Dictionary.doc2bow()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#
            gensim.corpora.dictionary.Dictionary.doc2bow
        .. _`gensim.corpora.Dictionary()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> example_docs = lda.data['review_text']
            >>> pos_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='positive')
            >>> # Considering bi-grams
            >>> pos_reviews_corpus = lda.make_corpus(pos_tokenized_docs, ngram=2)
            >>> isinstance(pos_reviews_corpus, tuple)
            True
            >>> # Considering tri-grams
            >>> pos_reviews_corpus = lda.make_corpus(pos_tokenized_docs, ngram=3)
            >>> isinstance(pos_reviews_corpus, tuple)
            True
            >>> # Traditional vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> example_docs = lda.data['review_text']
            >>> pos_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='positive')
            >>> # Considering bi-grams
            >>> pos_reviews_corpus = lda.make_corpus(pos_tokenized_docs, ngram=2)
            >>> isinstance(pos_reviews_corpus, tuple)
            True
            >>> # Considering tri-grams
            >>> pos_reviews_corpus = lda.make_corpus(pos_tokenized_docs, ngram=3)
            >>> isinstance(pos_reviews_corpus, tuple)
            True
        """

        texts = copy.copy(tokenized_docs)

        if ngram not in {None, 1}:
            # The higher the min_count/threshold, the less bi-grams there will be
            phrases = gensim.models.Phrases(
                sentences=tokenized_docs, min_count=min_count, threshold=threshold,
                connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS, scoring='npmi',
            )

            # Build a 2-gram model
            bi_gram_model = gensim.models.phrases.Phraser(phrases)

            texts = self._make_bi_grams(tokenized_docs, bi_gram_model)

            # pd.Series(texts)[pd.Series(texts).map(lambda x: any(y.count('_') == 1 for y in x))]

            if ngram == 3:  # Build a 3-gram model
                phrases_ = gensim.models.Phrases(
                    phrases[tokenized_docs], min_count=10, threshold=threshold,
                    connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS, scoring=scoring,
                )
                tri_gram_model = gensim.models.phrases.Phraser(phrases_)

                texts = self._make_tri_grams(tokenized_docs, bi_gram_model, tri_gram_model)

        # Create a dictionary for id-word mapping
        id2word = gensim.corpora.Dictionary(texts)
        # Term-document frequency
        corpus = [id2word.doc2bow(dat) for dat in texts]

        return corpus, id2word, texts

    def get_coherence_score(self, corpus, id2word, texts, num_topics, alpha, eta, **kwargs):
        # noinspection PyShadowingNames
        """
        Get the coherence score for an LDA model.

        :param corpus: corpus (i.e. term-document frequency, see `gensim.corpora.Dictionary.doc2bow()`_)
        :type corpus: list
        :param id2word: id-word mapping dictionary (see `gensim.corpora.Dictionary()`_)
        :type id2word: gensim.corpora.Dictionary
        :param texts: lemmatized review texts
        :type texts: list
        :param num_topics: number of topics, see ``num_topics`` of `gensim.models.LdaMulticore()`_
        :type num_topics: int
        :param alpha: ``alpha`` of `gensim.models.LdaMulticore()`_
        :type alpha: float or numpy.ndarray or list
        :param eta: ``eta`` of `gensim.models.LdaMulticore()`_
        :type eta: float or numpy.ndarray or list
        :param kwargs: [optional] parameters of `gensim.models.LdaMulticore()`_
        :return: coherence score of the LDA model given the specified parameters
        :rtype: float

        .. _`gensim.corpora.Dictionary.doc2bow()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#
            gensim.corpora.dictionary.Dictionary.doc2bow
        .. _`gensim.corpora.Dictionary()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#
            gensim.corpora.dictionary.Dictionary
        .. _`gensim.models.LdaMulticore()`:
            https://radimrehurek.com/gensim/models/ldamulticore.html#
            gensim.models.ldamulticore.LdaMulticore

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> # Traditional vacuum cleaners
            >>> # lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> example_docs = lda.data['review_text']
            >>> pos_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='positive')
            >>> # Consider bi-grams
            >>> pos_corpus, pos_id2word, pos_texts = lda.make_corpus(pos_tokenized_docs, ngram=2)
            >>> pos_coherence_score = lda.get_coherence_score(
            ...     pos_corpus, pos_id2word, pos_texts, num_topics=3, alpha=1, eta=1)
            >>> isinstance(pos_coherence_score, float)
            True
        """

        lda_pos = gensim.models.LdaMulticore(
            corpus=corpus, num_topics=num_topics, id2word=id2word, random_state=self.random_state,
            passes=10, per_word_topics=True, alpha=alpha, eta=eta,
            **kwargs)

        coh_model = gensim.models.coherencemodel.CoherenceModel(
            model=lda_pos, texts=texts, dictionary=id2word, coherence='c_v',
        )

        coherence_score = coh_model.get_coherence()

        return coherence_score

    def _train_model(self, corpus, id2word, num_topics, alpha, eta, **kwargs):
        lda_model_args = {
            'corpus': corpus,
            'num_topics': num_topics,
            'id2word': id2word,
            'passes': 10,
            'alpha': alpha,
            'eta': eta,
            'random_state': self.random_state,
            'per_word_topics': True,
        }
        kwargs.update(lda_model_args)

        if alpha == 'auto':
            kwargs.update({'distributed': False})
            lda_model = gensim.models.LdaModel(**kwargs)
        else:
            lda_model = gensim.models.LdaMulticore(**kwargs)

        return lda_model

    def train_model(self, corpus, id2word, texts, num_topics, alpha='asymmetric', eta='symmetric',
                    **kwargs):
        # noinspection PyShadowingNames
        """
        Train an LDA model.

        :param corpus: corpus (i.e. term-document frequency,
            see `gensim.corpora.Dictionary.doc2bow()`_)
        :type corpus: list
        :param id2word: id-word mapping dictionary (see `gensim.corpora.Dictionary()`_)
        :type id2word: gensim.corpora.Dictionary or list
        :param texts: lemmatized review texts
        :type texts: list
        :param num_topics: number of topics, see ``num_topics`` of `gensim.models.LdaMulticore()`_
        :type num_topics: int
        :param alpha: ``alpha`` of `gensim.models.LdaMulticore()`_, defaults to ``'asymmetric'``
        :type alpha: str or float or numpy.ndarray or list
        :param eta: ``eta`` of `gensim.models.LdaMulticore()`_, defaults to ``'symmetric'``
        :type eta: str or float or numpy.ndarray or list or None
        :param kwargs: [optional] parameters of `gensim.models.LdaMulticore()`_ or
            `gensim.models.LdaModel()`_
        :return: a collection of results, including an LDA model,
            a coherence model and coherence score
        :rtype: dict

        .. _`gensim.corpora.Dictionary.doc2bow()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#
            gensim.corpora.dictionary.Dictionary.doc2bow
        .. _`gensim.corpora.Dictionary()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#
            gensim.corpora.dictionary.Dictionary
        .. _`gensim.models.LdaModel()`:
            https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel
        .. _`gensim.models.LdaMulticore()`:
            https://radimrehurek.com/gensim/models/ldamulticore.html#
            gensim.models.ldamulticore.LdaMulticore

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> # Traditional vacuum cleaners
            >>> # lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> example_docs = lda.data['review_text']
            >>> neg_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='negative')
            >>> # Consider bi-grams
            >>> neg_corpus, neg_id2word, neg_texts = lda.make_corpus(neg_tokenized_docs, ngram=2)
            >>> # Consider three topics
            >>> neg_results = lda.train_model(neg_corpus, neg_id2word, neg_texts, num_topics=3)
            >>> isinstance(neg_results, dict)
            True
            >>> len(neg_results) == 3
            True
        """

        lda_model = self._train_model(
            corpus=corpus, id2word=id2word, num_topics=num_topics, alpha=alpha, eta=eta, **kwargs)

        coherence_model = gensim.models.coherencemodel.CoherenceModel(
            model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')

        coherence_score = coherence_model.get_coherence()

        results_ = [
            ('lda_model', lda_model),
            ('coherence_model', coherence_model),
            ('coherence_score', coherence_score),
        ]
        results = collections.OrderedDict(results_)

        return results

    def train_models(self, corpus, id2word, texts, num_topics_min=2, num_topics_max=6,
                     alpha='asymmetric', eta='symmetric', verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Train a number of LDA models.

        :param corpus: corpus
            (i.e. term-document frequency, see `gensim.corpora.Dictionary.doc2bow()`_)
        :type corpus: list
        :param id2word: id-word mapping dictionary (see `gensim.corpora.Dictionary()`_)
        :type id2word: gensim.corpora.Dictionary
        :param texts: lemmatized review texts
        :type texts: list
        :param num_topics_min: number of topics ranging from, defaults to ``2``
        :type num_topics_min: int
        :param num_topics_max: number of topics up to, defaults to ``6``
        :type num_topics_max: int
        :param alpha: ``alpha`` of `gensim.models.LdaMulticore()`_, defaults to ``'auto'``
        :type alpha: float or numpy.ndarray or list
        :param eta: ``eta`` of `gensim.models.LdaMulticore()`_, defaults to ``'asymmetric'``
        :type eta: float or numpy.ndarray or list
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :param kwargs: [optional] parameters of `gensim.models.LdaMulticore()`_ or
            `gensim.models.LdaModel()`_
        :return: a collection of results, including an LDA model, a coherence model and coherence score,
            for each given number of topics
        :rtype: dict

        .. _`gensim.corpora.Dictionary.doc2bow()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#
            gensim.corpora.dictionary.Dictionary.doc2bow
        .. _`gensim.corpora.Dictionary()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary
        .. _`gensim.models.LdaModel()`:
            https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel
        .. _`gensim.models.LdaMulticore()`:
            https://radimrehurek.com/gensim/models/ldamulticore.html#
            gensim.models.ldamulticore.LdaMulticore

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> # Traditional vacuum cleaners
            >>> # lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> # Smart thermostats
            >>> # lda = LatentDirichletAllocation('thermostats', product_type='smart')
            >>> example_docs = lda.data['review_text']
            >>> pos_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='positive')
            >>> # Consider tri-grams
            >>> pos_corpus, pos_id2word, pos_texts = lda.make_corpus(pos_tokenized_docs, ngram=3)
            >>> pos_results = lda.train_models(
            ...     pos_corpus, pos_id2word, pos_texts, num_topics_max=3, verbose=True)
            Coherence scores:
                2 topics: 0.6053
                3 topics: 0.5918
            >>> isinstance(pos_results, dict)
            True
        """

        results = collections.OrderedDict()

        for num_topics in range(num_topics_min, num_topics_max + 1):
            rslt = self.train_model(
                corpus=corpus, id2word=id2word, texts=texts,
                num_topics=num_topics, alpha=alpha, eta=eta,
                **kwargs)

            results[num_topics] = rslt

            if verbose:
                if num_topics == num_topics_min:
                    print("Coherence scores:")
                print(f"\t{num_topics} topics: {round(rslt['coherence_score'], 4)}")

        return results

    # == Evaluate models ===========================================================================

    @classmethod
    def prep_eval_corpuses(cls, corpus, proportions=None):
        # noinspection PyShadowingNames
        """
        Get a number of corpuses by specified proportions for model evaluation.

        :param corpus: corpus (i.e. term-document frequency, see `gensim.corpora.Dictionary.doc2bow()`_)
        :type corpus: gensim.utils.ClippedCorpus or list
        :param proportions: proportions, defaults to ``None``
        :type proportions: typing.Iterable or None
        :return: a list of corpuses and their respective proportions
        :rtype: typing.Tuple[list, list]

        .. _`gensim.corpora.Dictionary.doc2bow()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#
            gensim.corpora.dictionary.Dictionary.doc2bow

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> # Traditional vacuum cleaners
            >>> # lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> example_docs = lda.data['review_text']
            >>> neg_tokenized_docs = lda.get_tokenized_docs(example_docs, sentiment='negative')
            >>> # Consider bi-grams
            >>> neg_corpus, neg_id2word, neg_texts = lda.make_corpus(neg_tokenized_docs, ngram=2)
            >>> neg_corpus_eval_lists, neg_corpus_eval_props = lda.prep_eval_corpuses(neg_corpus)
            >>> isinstance(neg_corpus_eval_lists, list)
            True
            >>> len(neg_corpus_eval_lists)
            1
            >>> len(neg_corpus_eval_lists[0])
            77775
            >>> neg_corpus_eval_props
            ['100%']
            >>> neg_corpus_eval_lists, neg_corpus_eval_props = lda.prep_eval_corpuses(
            ...     corpus=neg_corpus, proportions=[0.8])
            >>> isinstance(neg_corpus_eval_lists, list)
            True
            >>> len(neg_corpus_eval_lists)
            2
            >>> list(map(len, neg_corpus_eval_lists))
            [62220, 77775]
            >>> neg_corpus_eval_props
            ['80%', '100%']
        """

        if proportions is None:
            # proportions_ = [1.0]  # np.arange(0.75, 1.0, 0.05)
            corpus_eval_lists = [corpus]
            corpus_eval_props = ['100%']

        else:
            # E.g. proportions = np.arange(0.75, 1.0, 0.05)
            max_docs = [int(len(corpus) * x) for x in proportions]
            corpus_eval_lists = [gensim.utils.ClippedCorpus(corpus, max_docs=x) for x in max_docs]

            corpus_eval_props = [f'{round(x * 100)}%' for x in proportions]
            if '100%' not in corpus_eval_props:
                corpus_eval_props.append('100%')
                corpus_eval_lists.append(corpus)

        return corpus_eval_lists, corpus_eval_props

    def _save_eval_summary(self, save_summary, sentiment, eval_summary, verbose, partitioned):
        if save_summary:
            rslt_pkl_filename = "eval_results.pkl" if save_summary is True else save_summary
            rslt_pkl_pathname = self.cd_models(sentiment, rslt_pkl_filename)

            save_args = {'data': eval_summary, 'path_to_file': rslt_pkl_pathname, 'verbose': verbose}
            if partitioned:
                save_partitioned_df(number_of_chunks=10, **save_args)
                gc.collect()
            else:
                save_data(protocol=pickle.HIGHEST_PROTOCOL, **save_args)

    def _eval_models_1(self, sentiment, corpus, id2word, texts, corpus_proportions, topic_numbers,
                       alphas, etas, sort_results=True, save_results=True, verbose=False):
        """
        Evaluate LDA models, given a set of specified parameters.

        One approach towards finding the best number of topics is using the coherence score metric.
        The coherence score essentially shows how similar the words from each topic are
        in terms of semantic value, with a higher score corresponding to higher similarity.

        :param sentiment: label of sentiment;
            options include :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`
        :type sentiment: str
        :param corpus: corpus (i.e. term-document frequency, see `gensim.corpora.Dictionary.doc2bow()`_)
        :type corpus: list
        :param id2word: id-word mapping dictionary (see `gensim.corpora.Dictionary()`_)
        :type id2word: gensim.corpora.Dictionary
        :param texts: lemmatized review texts
        :type texts: list
        :param topic_numbers: a list of numbers of topics
        :type topic_numbers: list
        :param alphas: a list of ``alpha`` (see ``alpha`` of `gensim.models.LdaMulticore()`_)
        :type alphas: list
        :param etas: a list of ``eta`` (see ``eta`` of `gensim.models.LdaMulticore()`_)
        :type etas: list
        :param corpus_proportions: proportions of the corpus for modelling, defaults to ``None``
        :type corpus_proportions: typing.Iterable or None
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :return: model evaluation results
        :rtype: pandas.DataFrame

        .. _`gensim.corpora.Dictionary.doc2bow()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#
            gensim.corpora.dictionary.Dictionary.doc2bow
        .. _`gensim.corpora.Dictionary()`:
            https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary
        .. _`gensim.models.LdaMulticore()`:
            https://radimrehurek.com/gensim/models/ldamulticore.html#
            gensim.models.ldamulticore.LdaMulticore

        **Examples**::

            from src.modeller import LatentDirichletAllocation
            import copy

            lda = LatentDirichletAllocation('vacuum', product_type='robotic')

            dat = copy.copy(lda.data)
            sen_col = copy.copy(lda.sentiment_column_name)
            rev_col = copy.copy(lda.review_column_name)

            sentiment = 'negative'

            docs = dat[dat[sen_col] == sentiment][rev_col]
            tokenized_docs = lda.get_tokenized_docs(docs, sentiment=sentiment)

            corpus, id2word, texts = lda.make_corpus(
                tokenized_docs=tokenized_docs, ngram=3, min_count=5, threshold=10e-5)

            corpus_proportions = [0.8]
            topic_numbers = [3, 4]
            alphas = ['asymmetric']  # document-topic density; ['symmetric', 'auto']
            etas = ['symmetric']  # word-topic density

            neg_lda_eval_results = lda._eval_models_1(
                sentiment=sentiment,
                corpus=corpus, id2word=id2word, texts=texts, corpus_proportions=corpus_proportions,
                topic_numbers=topic_numbers, alphas=alphas, etas=etas,
                verbose=True)
        """

        corpus_lists, corpus_props = self.prep_eval_corpuses(corpus, proportions=corpus_proportions)
        params_set = list(itertools.product(range(len(corpus_props)), topic_numbers, alphas, etas))

        total = len(params_set)
        counter = 1

        if verbose:
            print(f"Evaluating {total} LDA models for \"{sentiment}\" reviews ... ")

        results_ = collections.defaultdict(list)
        start_time = time.time()
        for i, num_topics, alpha, eta in params_set:
            if verbose:
                a_, e_ = map(lambda x: f"'{x}'" if isinstance(x, str) else f'{x}', [alpha, eta])
                print(f"\t{counter}/{total}: {'{'}"
                      f"corpus_proportion={corpus_props[i]}, "
                      f"num_topics={num_topics}, "
                      f"alpha={a_}, "
                      f"eta={e_}"
                      f"{'}'}", end=" ... ")

            try:
                modelling_results = self.train_model(
                    corpus=corpus_lists[i], id2word=id2word, texts=texts,
                    num_topics=num_topics, alpha=alpha, eta=eta)

                results_['num_topics'].append(num_topics)
                results_['alpha'].append(alpha)
                results_['eta'].append(eta)

                # lda_model, coh_model, coh_score = lda_results.values()
                _, _, coh_score = modelling_results.values()
                # results_['lda_model'].append(lda_model)
                # results_['coherence_model'].append(coh_model)
                results_['coherence_score'].append(coh_score)

                if verbose:
                    end_time = time.time()
                    elapsed_time = str(
                        datetime.timedelta(seconds=round(end_time - start_time, 2)))[:-7]
                    print(f"Done. (Elapsed time: {elapsed_time})")

                del modelling_results  # lda_model, coh_model, coh_score
                gc.collect()

            except Exception as e:
                if verbose:
                    end_time = time.time()
                    elapsed_time = str(datetime.timedelta(seconds=round(end_time - start_time, 2)))[:-7]
                    print(f"Failed. {e} (Elapsed time: {elapsed_time})")
                else:
                    _print_failure_msg(e, msg="Failed.")

            counter += 1

        results = pd.DataFrame(results_)
        if sort_results:
            results.sort_values('coherence_score', ascending=False, ignore_index=True, inplace=True)

        self._save_eval_summary(save_results, sentiment, results, verbose, partitioned=False)

        return results

    def _eval_models_2(self, sentiment, tokenized_docs, ngram, min_counts, thresholds,
                       corpus_proportions, topic_numbers, alphas, etas, sort_results=True,
                       save_results=True, verbose=False):
        """
        from src.modeller import LatentDirichletAllocation
        import copy

        lda = LatentDirichletAllocation('vacuum', product_type='robotic')

        dat = copy.copy(lda.data)
        sentiment = 'negative'
        docs = dat[dat[lda.sentiment_column_name] == sentiment][lda.review_column_name]

        tokenized_docs = lda.get_tokenized_docs(docs=docs, sentiment=sentiment)

        ngram = 3
        min_counts = [1, 5]  # min_counts = np.arange(1, 6)
        thresholds = [10e-5]  # thresholds = [10e-5, 0.5, 1]

        topic_numbers = range(3, 4)
        alphas = ['asymmetric']  # ['symmetric', 'asymmetric']
        etas = ['symmetric']
        corpus_proportions = [0.8]

        neg_lda_eval_results = lda._eval_models_2(
            sentiment=sentiment, tokenized_docs=tokenized_docs,
            ngram=ngram, min_counts=min_counts, thresholds=thresholds,
            topic_numbers=topic_numbers,
            alphas=alphas,  # document-topic density
            etas=etas,  # word-topic density
            corpus_proportions=corpus_proportions,
            verbose=True)
        """

        corpus_params = list(itertools.product(min_counts, thresholds))

        corpus_lists, id2word_list, texts_list = [], [], []
        for min_count, threshold in corpus_params:
            corpus, id2word, texts = self.make_corpus(
                tokenized_docs=tokenized_docs, ngram=ngram, min_count=min_count, threshold=threshold,
                scoring='npmi')

            corpus_temp, props_temp = self.prep_eval_corpuses(corpus, proportions=corpus_proportions)
            corpus_lists.append([[min_count, threshold, x, y] for x, y in zip(props_temp, corpus_temp)])

            id2word_list.append([id2word] * len(props_temp))
            texts_list.append([texts] * len(props_temp))

        corpus_lists, id2word_list, texts_list = map(
            lambda x: list(itertools.chain.from_iterable(x)), [corpus_lists, id2word_list, texts_list])

        eval_params = list(itertools.product(range(len(corpus_lists)), topic_numbers, alphas, etas))

        total = len(eval_params)
        counter = 1

        if verbose:
            print(f"Evaluating {total} LDA models for \"{sentiment}\" reviews ... ")

        results_ = collections.defaultdict(list)

        start_time = time.time()

        for i, num_topics, alpha, eta in eval_params:
            min_count, threshold, prop, corpus = corpus_lists[i]
            try:
                if verbose:
                    a_, e_ = map(lambda x: f"'{x}'" if isinstance(x, str) else f"{x}", [alpha, eta])
                    print(f"\t{counter}/{total}: {'{'}"
                          f"min_count={min_count}, "
                          f"threshold={threshold}, "
                          f"corpus_prop='{prop}', "
                          f"num_topics={num_topics}, "
                          f"alpha={a_}, "
                          f"eta={e_}"
                          f"{'}'}", end=" ... ")

                results_['min_count'].append(min_count)
                results_['threshold'].append(threshold)
                results_['corpus_proportion'].append(prop)
                results_['num_topics'].append(num_topics)
                results_['alpha'].append(alpha)
                results_['eta'].append(eta)

                lda_results = self.train_model(
                    corpus=corpus, id2word=id2word_list[i], texts=texts_list[i],
                    num_topics=num_topics, alpha=alpha, eta=eta, random_state=self.random_state)

                # lda_model, coh_model, coh_score = lda_results.values()
                _, _, coh_score = lda_results.values()
                # results_['lda_model'].append(lda_model)
                # results_['coherence_model'].append(coh_model)
                results_['coherence_score'].append(coh_score)

                if verbose:
                    end_time = time.time()
                    elapsed_time = str(datetime.timedelta(seconds=round(end_time - start_time, 2)))
                    print(f"Done. (Elapsed time: {elapsed_time[:-7]})")

            except Exception as e:
                if verbose:
                    end_time = time.time()
                    elapsed_time = str(datetime.timedelta(seconds=round(end_time - start_time, 2)))
                    print(f"Failed. {e} (Elapsed time: {elapsed_time[:-7]})")

            counter += 1

        results = pd.DataFrame(results_)
        if sort_results:
            results.sort_values(
                by='coherence_score', ascending=False, ignore_index=True, inplace=True)

        self._save_eval_summary(save_results, sentiment, results, verbose, partitioned=False)

        return results

    def _make_eval_corpus(self, sentiment, tokenized_docs, corpus_prop, min_count, threshold,
                          ngram=3):
        prop = int(corpus_prop.strip('%'))
        temp_pkl_filename = f"mc{min_count}-thr{threshold}-prop{prop}".replace(".", "_") + ".pkl"
        temp_pkl_pathname = self.cd_models(sentiment, "_temp", temp_pkl_filename)

        if os.path.isfile(temp_pkl_pathname):
            corpus, id2word, texts = load_data(temp_pkl_pathname)

        else:
            corpus, id2word, texts = self.make_corpus(
                tokenized_docs=tokenized_docs, ngram=ngram, min_count=min_count,
                threshold=threshold, scoring='npmi')
            corpus = gensim.utils.ClippedCorpus(
                corpus=corpus, max_docs=int(id2word.num_docs * prop / 100))

            save_data([corpus, id2word, texts], temp_pkl_pathname)

        return corpus, prop, id2word, texts

    def _evaluate_models(self, sentiment, ngram, min_counts, thresholds, corpus_proportions,
                         topic_numbers, alphas, etas, sort_results=True, save_results=True,
                         partitioned=False, conf_reqd=True, verbose=True):
        """
        Evaluate LDA models.

        :param sentiment: label of sentiment;
            options include :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`
        :type sentiment: str
        :param topic_numbers: a list of numbers of topics
        :type topic_numbers: list or range
        :param alphas: a list of ``alpha`` (see ``alpha`` of `gensim.models.LdaMulticore()`_)
        :type alphas: list or range
        :param etas: a list of ``eta`` (see ``eta`` of `gensim.models.LdaMulticore()`_)
        :type etas: list or range
        :param corpus_proportions: proportions of the corpus for modelling, defaults to ``None``
        :type corpus_proportions: typing.Iterable or None
        :param ngram: number of grams, defaults to ``3``
        :type ngram: int
        :param conf_reqd: whether to ask for confirmation to proceed, defaults to ``True``
        :type conf_reqd: bool
        :param save_results: whether to save the evaluation results as a pickle file,
            defaults to ``True``
        :type save_results: bool or int or str
        :return: model evaluation results
            (and, optionally, the topics with the highest coherence score for each topic)
        :rtype: pandas.DataFrame or tuple

        .. _`gensim.models.LdaMulticore()`:
            https://radimrehurek.com/gensim/models/ldamulticore.html#
            gensim.models.ldamulticore.LdaMulticore
        .. _`gensim.models.phrases.Phrases()`:
            https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases

        from src.modeller import LatentDirichletAllocation

        lda = LatentDirichletAllocation('vacuum', product_type='robotic')

        sentiment = 'negative'
        ngram = 3
        min_counts = [1, 5]  # np.arange(1, 6)
        thresholds = [10e-5]  # [10e-5, 0.5, 1]
        corpus_proportions = [0.8]
        topic_numbers = [3, 4]
        alphas = ['asymmetric']  # ['symmetric', 'asymmetric']
        etas = ['symmetric']

        neg_eval_results = lda._evaluate_models(
            sentiment,
            ngram, min_counts, thresholds, corpus_proportions,
            topic_numbers, alphas, etas
        )
        """

        self._assert_sentiment(sentiment)

        prod_name = self.reviews.PRODUCT_NAME.lower()
        cfm_msg = f'Evaluating LDA models for "{self.sentiment}" reviews of {prod_name}\n?'

        if confirmed(cfm_msg, confirmation_required=conf_reqd):
            docs = self.data[
                self.data[self.sentiment_column_name] == self.sentiment][self.review_column_name]
            tokenized_docs = self.get_tokenized_docs(docs=docs, sentiment=self.sentiment)

            if corpus_proportions is None:
                corpus_props = ['100%']
            else:
                # E.g. proportions = np.arange(0.75, 1.0, 0.05)
                corpus_props = [f'{round(x * 100)}%' for x in corpus_proportions]
                if '100%' not in corpus_props:
                    corpus_props.append('100%')
            corpus_gen_params = list(itertools.product(min_counts, thresholds, corpus_props))

            params_set = list(itertools.product(corpus_gen_params, topic_numbers, alphas, etas))

            total = len(params_set)
            counter = 1

            if verbose:
                if conf_reqd:
                    print(f"Evaluation starts ... ")
                else:
                    print(f"Evaluating {total} LDA models for \"{self.sentiment}\" reviews ... ")

            eval_summary_ = collections.defaultdict(list)

            start_time = time.time()

            for min_count, threshold, corpus_prop in corpus_gen_params:
                corpus, prop, id2word, texts = self._make_eval_corpus(
                    sentiment=sentiment, tokenized_docs=tokenized_docs, corpus_prop=corpus_prop,
                    min_count=min_count, threshold=threshold, ngram=ngram)

                for num_topics, alpha, eta in itertools.product(topic_numbers, alphas, etas):
                    if verbose:
                        alpha_, eta_ = map(
                            lambda x: f"'{x}'" if isinstance(x, str) else f"{x}", [alpha, eta])
                        print(f"\t{counter}/{total}: {'{'}"
                              f"min_count={min_count}, "
                              f"threshold={threshold}, "
                              f"corpus_prop='{corpus_prop}', "
                              f"num_topics={num_topics}, "
                              f"alpha={alpha_}, "
                              f"eta={eta_}"
                              f"{'}'}", end=" ... ")

                    eval_summary_['min_count'].append(min_count)
                    eval_summary_['threshold'].append(threshold)
                    eval_summary_['corpus_proportion'].append(corpus_prop)
                    eval_summary_['num_topics'].append(num_topics)
                    eval_summary_['alpha'].append(alpha)
                    eval_summary_['eta'].append(eta)

                    modelling_results = self.train_model(
                        corpus=corpus, id2word=id2word, texts=texts,
                        num_topics=num_topics, alpha=alpha, eta=eta, random_state=self.random_state)

                    # lda_model, coh_model, coh_score = lda_results.values()
                    _, _, coh_score = modelling_results.values()
                    # results_['lda_model'].append(lda_model)
                    # results_['coherence_model'].append(coh_model)
                    eval_summary_['coherence_score'].append(coh_score)

                    del modelling_results
                    gc.collect()

                    if verbose:
                        end_time = time.time()
                        elapsed_time_ = round(end_time - start_time, 2)
                        elapsed_time = str(datetime.timedelta(seconds=elapsed_time_))
                        print(f"Done. (Elapsed time: {elapsed_time[:-7]})")

                    counter += 1

            eval_summary = pd.DataFrame(eval_summary_)
            if sort_results:
                eval_summary.sort_values(
                    by='coherence_score', ascending=False, ignore_index=True, inplace=True)

            self._save_eval_summary(
                save_summary=save_results, sentiment=sentiment, eval_summary=eval_summary,
                verbose=verbose, partitioned=partitioned)

            return eval_summary

    def evaluate_models(self, verbose=True):
        # noinspection PyShadowingNames
        """
        Evaluate LDA models for each group of reviews (e.g. positive reviews and negative reviews).

        :param verbose: whether to print relevant information in console, defaults to ``True``
        :type verbose: bool | int

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> lda.evaluate_models()  # (This may take a huge amount of time.)
            >>> # Traditional vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> lda.evaluate_models()  # (This may take a huge amount of time.)
            >>> # Smart thermostats
            >>> lda = LatentDirichletAllocation('therms', product_type='smart')
            >>> lda.min_counts = range(1, 6)
            >>> lda.thresholds = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
            >>> lda.corpus_proportions = [1.0]
            >>> lda.pos_topic_numbers = range(2, 6)
            >>> lda.neg_topic_numbers = range(2, 6)
            >>> lda.evaluate_models()  # (This may take a huge amount of time.)
        """

        product_name = self.reviews.PRODUCT_NAME.lower()

        if confirmed(f'To evaluate LDA models for the reviews of "{product_name}"\n?'):

            # -- 1. Positive reviews ---------------------------------------------------------------
            try:
                _ = self._evaluate_models(
                    sentiment='positive',
                    ngram=3, min_counts=self.min_counts, thresholds=self.thresholds,
                    corpus_proportions=self.corpus_proportions,
                    topic_numbers=self.pos_topic_numbers,
                    alphas=self.pos_alphas, etas=self.pos_etas, conf_reqd=False, verbose=verbose)
            except Exception as e:
                print(e)
                pass

            gc.collect()

            # -- 2. Negative reviews ---------------------------------------------------------------
            try:
                _ = self._evaluate_models(
                    sentiment='negative',
                    ngram=3, min_counts=self.min_counts, thresholds=self.thresholds,
                    corpus_proportions=self.corpus_proportions,
                    topic_numbers=self.neg_topic_numbers,
                    alphas=self.neg_alphas, etas=self.neg_etas, conf_reqd=False, verbose=verbose)
            except Exception as e:
                print(e)
                pass

            gc.collect()

    # == Summarise evaluation results ==============================================================

    def fetch_evaluation_summary(self, sentiment, verbose=False):
        # noinspection PyShadowingNames
        """
        Fetch the summary of the LDA model evaluation results.

        :param sentiment: label of sentiment;
            options include :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`
        :type sentiment: str
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :return: summary of the LDA model evaluation results for the given ``sentiment``
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> import pandas as pd
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> pos_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='positive')
            >>> isinstance(pos_lda_eval_summary, pd.DataFrame)
            True
            >>> neg_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='negative')
            >>> isinstance(neg_lda_eval_summary, pd.DataFrame)
            True
            >>> neu_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='neutral')
            >>> neu_lda_eval_summary is None
            True
            >>> # Traditional vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> pos_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='positive')
            >>> isinstance(pos_lda_eval_summary, pd.DataFrame)
            True
            >>> neg_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='negative')
            >>> isinstance(neg_lda_eval_summary, pd.DataFrame)
            True
            >>> neu_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='neutral')
            >>> neu_lda_eval_summary is None
            True
        """

        self._assert_sentiment(sentiment)

        rslt_pkl_pathname = self.cd_models(sentiment, "eval_results.pkl")

        if os.path.isfile(rslt_pkl_pathname):
            eval_summary = load_data(rslt_pkl_pathname, verbose=verbose)

        else:
            if verbose:
                print(f'"{os.path.relpath(rslt_pkl_pathname)}" does not exist.')
            eval_summary = None

        return eval_summary

    @staticmethod
    def _colour_order(dat):
        temp_list = list(dat.unique())
        temp_list.sort(key=lambda x: (str(type(x)), x))
        return temp_list

    def view_evaluation_summary(self, sentiment, partially=None, save_as=None, verbose=False,
                                **kwargs):
        # noinspection PyShadowingNames
        """
        Visualise the results of the evaluation summary.

        :param sentiment: Label of sentiment;
            options include :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`.
        :type sentiment: str
        :param partially: View the evaluation summary based a selected set of hyperparameters
            (particularly when the numbers of some hyperparameters are large);
            defaults to ``None``.
        :type partially: None | dict
        :param save_as: Extension of figure filename, or whether to save the figure;
            defaults to ``None``.
        :type save_as: str | bool | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :param kwargs: [Optional] additional parameters of `pyhelpers.store.save_figure`_.

        .. _`pyhelpers.store.save_figure`:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> from pyhelpers.settings import mpl_preferences
            >>> mpl_preferences(backend='TkAgg')
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> # lda.view_evaluation_summary(sentiment='positive', save_as=".svg", verbose=True)
            >>> lda.view_evaluation_summary(sentiment='positive')

        .. figure:: ../_images/robotic_vacuum_cleaners/positive/eval_summary.*
            :name: robotic_vacuum_cleaners_positive_eval_summary
            :align: center
            :width: 100%

            LDA modeling trials for positive reviews on robotic vacuum cleaners.

        .. code-block:: python

            >>> # lda.view_evaluation_summary(sentiment='negative', save_as=".svg", verbose=True)
            >>> lda.view_evaluation_summary(sentiment='negative')

        .. figure:: ../_images/robotic_vacuum_cleaners/negative/eval_summary.*
            :name: robotic_vacuum_cleaners_negative_eval_summary
            :align: center
            :width: 100%

            LDA modeling trials for negative reviews on robotic vacuum cleaners.

        .. code-block:: python

            >>> # Traditional vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> # lda.view_evaluation_summary(sentiment='positive', save_as=".svg", verbose=True)
            >>> lda.view_evaluation_summary(sentiment='positive')

        .. figure:: ../_images/traditional_vacuum_cleaners/positive/eval_summary.*
            :name: traditional_vacuum_cleaners_positive_eval_summary
            :align: center
            :width: 100%

            LDA modeling trials for positive reviews on traditional vacuum cleaners.

        .. code-block:: python

            >>> # lda.view_evaluation_summary(sentiment='negative', save_as=".svg", verbose=True)
            >>> lda.view_evaluation_summary(sentiment='negative')

        .. figure:: ../_images/traditional_vacuum_cleaners/negative/eval_summary.*
            :name: traditional_vacuum_cleaners_negative_eval_summary
            :align: center
            :width: 100%

            LDA modeling trials for negative reviews on traditional vacuum cleaners.

        .. code-block:: python

            >>> # Smart thermostats
            >>> lda = LatentDirichletAllocation('therms', product_type='smart')
            >>> partially = {'min_count': (1, 5), 'threshold': (0.0001, 1)}
            >>> # lda.view_evaluation_summary(
            ... #     sentiment='positive', partially=partially, save_as=".svg", verbose=True)
            >>> lda.view_evaluation_summary(sentiment='positive', partially=partially)

        .. figure:: ../_images/smart_thermostats/positive/eval_summary_partial.*
            :name: smart_thermostats_positive_eval_summary_partial
            :align: center
            :width: 100%

            LDA modeling trials for positive reviews on smart thermostats.

        .. code-block:: python

            >>> # lda.view_evaluation_summary(
            ... #     sentiment='negative', partially=partially, save_as=".svg", verbose=True)
            >>> lda.view_evaluation_summary(sentiment='negative', partially=partially)

        .. figure:: ../_images/smart_thermostats/negative/eval_summary_partial.*
            :name: smart_thermostats_negative_eval_summary_partial
            :align: center
            :width: 100%

            LDA modeling trials for negative reviews on smart thermostats.
        """

        self._assert_sentiment(sentiment)

        data = self.eval_summary[self.sentiment].query('`alpha` != "auto" & `eta` != "auto" ')

        if partially:
            data = data.query(' & '.join([f'`{k}` in {v}' for k, v in partially.items()]))

        sns.set_theme(style='ticks')
        sns.set(font_scale=1.4)
        mpl_preferences(font_size=16)

        g = sns.relplot(
            data=data, x='num_topics', y='coherence_score',
            col='min_count',  # col_wrap=1,
            row='threshold',
            hue='eta', hue_order=self._colour_order(data['eta']),
            style='alpha', style_order=self._colour_order(data['alpha']),
            # height=2, aspect=2,
            palette='Set2',
            facet_kws={'sharey': False, 'sharex': True},
            kind='line',
        )

        x_min = data['num_topics'].min()
        x_max = data['num_topics'].max()
        g.set(xlim=(x_min, x_max))

        for ax in g.axes.flatten():  # ax = g.axes.flatten()[1]
            t = '(' + ax.get_title().replace(' = ', '=') + ')'
            ax.set_title('')
            ax.set_title(t, loc='right')
            # ax.set_title('(' + ax.get_title().replace(' = ', '=') + ')')
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
            # ax.xaxis.grid(True)
            # ax.yaxis.grid(True)

        for l_txt in g._legend.texts:
            txt = l_txt.get_text()
            if txt in {'eta', 'alpha'}:
                l_txt.set_text('\n' + txt + ': ')
                l_txt.set_weight('bold')
                l_txt.set_position((-50, -5))

        g.set_axis_labels(
            x_var='Number of topics', y_var='Coherence score', fontdict=dict(weight='bold'))

        g.tight_layout()

        if save_as:
            for save_as_ in {save_as, ".svg", ".pdf"}:
                suffix = "_partial" if partially is not None else ""
                sum_filename = f"eval_summary{suffix}{save_as_}"
                path_to_fig = self.cd_models(self.sentiment, sum_filename)

                # g.savefig(path_to_fig, bbox_inches='tight')
                save_data(g, path_to_fig, verbose=verbose, bbox_inches='tight', **kwargs)

                prod_name = self.reviews.PRODUCT_NAME.lower().replace(' ', '_')
                _file_path = f"{prod_name}\\{self.sentiment}\\{sum_filename}"
                docs_file_path = cd(f"docs\\source\\_images\\{_file_path}", mkdir=True)
                shutil.copyfile(path_to_fig, docs_file_path)

    # == Analyse the modelling results =============================================================

    @classmethod
    def _parse_proportion(cls, corpus_proportion):
        if isinstance(corpus_proportion, str):
            if '%' in corpus_proportion:
                proportion = float(corpus_proportion.strip('%')) / 100
            else:
                proportion = float(corpus_proportion)
        else:
            proportion = corpus_proportion

        if proportion > 1:
            proportion = proportion / 100

        assert 0 < proportion <= 1

        return proportion

    def _get_topics(self, corpus, id2word, texts, num_topics, alpha, eta, n_top_tokens=50,
                    **kwargs):
        lda_model = self._train_model(
            corpus=corpus, id2word=id2word, num_topics=num_topics, alpha=alpha, eta=eta, **kwargs)

        model_top_topics = lda_model.top_topics(
            corpus=corpus, texts=texts, dictionary=id2word, coherence='c_v', topn=n_top_tokens)

        summary_dat = [
            pd.DataFrame(
                x[0], columns=[f'topic_{topic_id}_prob', f'topic_{topic_id}_word_or_phrase'])
            for topic_id, x in enumerate(model_top_topics)]

        top_topics = pd.concat(summary_dat, axis=1)

        return top_topics

    def get_topics(self, sentiment, i, n_top_tokens=50, export_to_file=True, verbose=True, **kwargs):
        # noinspection PyShadowingNames
        """
        Get the top ``n_top_tokens`` words/phrases for LDA models.

        :param sentiment: label of sentiment;
            options include :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`
        :type sentiment: str
        :param i: an index or a list of indices of the dataframe of model evaluation summary
        :type i: int | list
        :param n_top_tokens: number of words/phrases in each of the resulting topics,
            defaults to ``50``; see ``topn`` of `gensim.models.LdaMulticore.top_topics()`_
        :type n_top_tokens: int
        :param export_to_file: whether to save the results to a spreadsheet file;
            defaults to ``True``.
        :type export_to_file: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :param kwargs: [Optional] additional parameters for the function
            `pyhelpers.store.save_spreadsheets()`_
        :return: the top ``n_top_tokens`` words/phrases for each of the specified LDA models
        :rtype: collections.OrderedDict

        .. _`gensim.models.LdaMulticore.top_topics()`:
            https://radimrehurek.com/gensim/models/ldamulticore.html#
            gensim.models.ldamulticore.LdaMulticore.top_topics
        .. _`pyhelpers.store.save_spreadsheets()`:
            https://pyhelpers.readthedocs.io/en/latest/_generated/
            pyhelpers.store.save_spreadsheets.html

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> import collections
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> pos_top_topics_data = lda.get_topics(
            ...     sentiment='positive', i=76, n_top_tokens=50, export=False)
            >>> isinstance(pos_top_topics_data, collections.OrderedDict)
            True
            >>> pos_top_topics_data[76].shape
            (50, 6)
            >>> neg_top_topics_data = lda.get_topics(
            ...     sentiment='negative', i=[98, 123], n_top_tokens=50, export=False)
            >>> isinstance(neg_top_topics_data, collections.OrderedDict)
            True
            >>> list(neg_top_topics_data.keys())
            [98, 123]
        """

        self._assert_sentiment(sentiment)

        tokenized_docs = self.get_tokenized_docs(self.data[self.review_column_name], sentiment)

        eval_summary = self.fetch_evaluation_summary(sentiment=sentiment)
        params_columns = [
            'min_count', 'threshold', 'num_topics', 'alpha', 'eta', 'corpus_proportion']
        j = [i] if isinstance(i, int) else i

        top_topics_data = collections.OrderedDict()
        for idx, dat in eval_summary.loc[j].iterrows():
            min_count, threshold, num_topics, alpha, eta, corpus_prop = dat[params_columns]

            corpus, prop, id2word, texts = self._make_eval_corpus(
                sentiment=sentiment, tokenized_docs=tokenized_docs, corpus_prop=corpus_prop,
                min_count=min_count, threshold=threshold, ngram=3)

            top_topics = self._get_topics(
                corpus=corpus, id2word=id2word, texts=texts, num_topics=num_topics, alpha=alpha,
                eta=eta, n_top_tokens=n_top_tokens)

            top_topics_data.update({idx: top_topics})

            del top_topics
            gc.collect()

        if export_to_file:
            path_to_top_topics = self.cd_models(sentiment, "topics_for_top_10_models.xlsx")
            save_spreadsheets(
                list(top_topics_data.values()), path_to_file=path_to_top_topics,
                sheet_names=[f'model_{idx}' for idx in top_topics_data.keys()], verbose=verbose,
                **kwargs)

        return top_topics_data

    @staticmethod
    def get_common_words(topics_data):
        # noinspection PyShadowingNames
        """
        Get common words from a number of topics estimated by an LDA model.

        :param topics_data: data of a number of topics
        :type topics_data: pandas.DataFrame
        :return: a set of common words
        :rtype: set

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Robotic vacuum cleaners
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> pos_top_topics_50tokens = lda.get_topics('positive', i=76, n_top_tokens=50)
            >>> lda.get_common_words(pos_top_topics_50tokens)
            {'area',
             'clean',
             'cleaning',
             'long',
             'look',
             'room',
             'set',
             'time',
             'try',
             'want'}
        """

        topic_words_ = topics_data[[x for x in topics_data if x.endswith('_word_or_phrase')]]

        topic_words = [topic_words_[x].to_list() for x in topic_words_]

        common_words = functools.reduce(lambda a, b: a & b, map(set, topic_words))

        return common_words

    def _retrain_model(self, **kwargs):
        lda_model_results = self.train_model(**kwargs)
        lda_model_ = lda_model_results['lda_model']

        return lda_model_

    def _prep_materials(self, sentiment):
        self._assert_sentiment(sentiment)

        if self.sentiment == 'positive':
            if self.pos_tokenized_docs is None:
                data_ = self.data[self.data[self.sentiment_column_name] == self.sentiment]
                docs = data_[self.review_column_name]
                self.pos_tokenized_docs = self.get_tokenized_docs(docs=docs, sentiment=self.sentiment)
            self.tokenized_docs[self.sentiment] = self.pos_tokenized_docs

        elif self.sentiment == 'negative':
            if self.neg_tokenized_docs is None:
                data_ = self.data[self.data[self.sentiment_column_name] == self.sentiment]
                docs = data_[self.review_column_name]
                self.neg_tokenized_docs = self.get_tokenized_docs(docs=docs, sentiment=self.sentiment)
            self.tokenized_docs[self.sentiment] = self.neg_tokenized_docs

        if self.eval_summary[self.sentiment] is None:
            self.eval_summary[self.sentiment] = self.fetch_evaluation_summary(sentiment=sentiment)

    def _get_vis_data(self, sentiment, tokenized_docs, corpus_prop, min_count, threshold,
                      num_topics, alpha, eta, **kwargs):

        import pyLDAvis.gensim_models

        corpus, _, id2word, _ = self._make_eval_corpus(
            sentiment=sentiment, tokenized_docs=tokenized_docs, corpus_prop=corpus_prop,
            min_count=min_count, threshold=threshold, ngram=3)

        train_lda_args = {
            'corpus': corpus,
            'id2word': id2word,
            'num_topics': num_topics,
            'alpha': alpha,
            'eta': eta}
        lda_model = self._train_model(**train_lda_args)

        # with warnings.catch_warnings():
        #     warnings.simplefilter(action='ignore', category=DeprecationWarning)
        prepare_args = {
            'topic_model': lda_model,
            'corpus': corpus,
            'dictionary': id2word,
            'sort_topics': False,
        }
        kwargs.update(prepare_args)

        lda_vis = pyLDAvis.gensim_models.prepare(**kwargs)

        if 'complex' in lda_vis.topic_coordinates['x'].dtype.name:
            kwargs.update({'mds': 'mmds'})
            lda_vis = pyLDAvis.gensim_models.prepare(**kwargs)

        del corpus, id2word
        gc.collect()

        return lda_vis, lda_model

    def _export_vis_data_to_html(self, ignore_auto_alpha, vis_data, sentiment, i, update, verbose):
        import pyLDAvis

        self._assert_sentiment(sentiment)

        if isinstance(i, int):
            j = self.eval_summary_.index[i]
            k = str(j).zfill(3)
            html_filename = f"prepared_{k}.html"
            html_pathname = self.cd_models(self.sentiment, "vis", html_filename, mkdir=True)

            if isinstance(vis_data, dict):
                vis_data_ = vis_data[f'LDA_{k}']
            else:  # isinstance(vis_data, tuple):
                vis_data_ = vis_data

            if not os.path.isfile(html_pathname) or update:
                _check_saving_path(html_pathname, verbose=verbose, ret_info=False)

                try:
                    pyLDAvis.save_html(data=vis_data_, fileobj=html_pathname)
                    if verbose:
                        print("Done.")
                except Exception as e:
                    _print_failure_msg(e)

                save_data(vis_data_, html_pathname.replace(".html", ".pkl"), verbose=verbose)

                prod_name = self.reviews.PRODUCT_NAME.lower().replace(' ', '_')
                _file_path = f"lda_vis\\{prod_name}\\{self.sentiment}\\{html_filename}"
                docs_file_path = cd(f"docs\\source\\_static\\{_file_path}", mkdir=True)
                shutil.copyfile(html_pathname, docs_file_path)

            else:
                if verbose:
                    print(f"\"{os.path.relpath(html_pathname)}\" already exists.")

        else:
            assert isinstance(vis_data, dict)

            for dat, idx in zip(vis_data.values(), i):
                self._export_vis_data_to_html(
                    ignore_auto_alpha=ignore_auto_alpha, vis_data=dat, sentiment=sentiment,
                    i=idx, update=update, verbose=verbose)

    def get_vis_data(self, sentiment, i=None, ignore_auto_alpha=False, export_to_html=False,
                     update=False, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Get visualisation data for LDA models.

        :param sentiment: label of sentiment;
            options include :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`
        :type sentiment: str
        :param i: row index or indices of the model evaluation summary, defaults to ``None``
        :type i: int or typing.Iterable or None
        :param ignore_auto_alpha: whether to ignore the situation when ``alpha='auto'``
        :type ignore_auto_alpha: bool
        :param export_to_html: whether to save the model visualisation data to an HTML file,
            defaults to ``False``
        :type export_to_html: bool
        :param update: whether to replace the existing HTML file with an updated one,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :param kwargs: [optional] parameters of `pyLDAvis.gensim_models.prepare()`_
        :return: prepared data for visualising the LDA model
        :rtype: pyLDAvis.PreparedData

        .. _`pyLDAvis.gensim_models.prepare()`:
            https://pyldavis.readthedocs.io/en/latest/modules/API.html#pyLDAvis.prepare

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> product_category = 'vacuum'
            >>> product_type = 'robotic'

        **Positive reviews**::

            >>> lda = LatentDirichletAllocation(product_category, product_type)
            >>> # lda = LatentDirichletAllocation(product_category, product_type='traditional')
            >>> # pos_lda_vis_data_1 = lda.get_vis_data(
            ... #     sentiment='positive', i=0, export_to_html=True, verbose=True)
            >>> # pos_lda_vis_data_2 = lda.get_vis_data(
            ... #     sentiment='positive', i=range(50), verbose=True)
            >>> pos_lda_vis_data_3 = lda.get_vis_data(
            ...     sentiment='positive', i=range(10), export_to_html=True, verbose=True,
            ...     ignore_auto_alpha=True)

        **Negative reviews**::

            >>> lda = LatentDirichletAllocation(product_category, product_type)
            >>> # lda = LatentDirichletAllocation(product_category, product_type='traditional')
            >>> # neg_lda_vis_data_1 = lda.get_vis_data(
            ... #     sentiment='negative', i=0, export_to_html=True, verbose=True)
            >>> # neg_lda_vis_data_2 = lda.get_vis_data(
            ... #     sentiment='negative', i=range(50), verbose=True)
            >>> neg_lda_vis_data_3 = lda.get_vis_data(
            ...     sentiment='negative', i=range(10), export_to_html=True, verbose=True,
            ...     ignore_auto_alpha=True)
        """

        self._prep_materials(sentiment=sentiment)

        if i is None:
            indices_ = range(10)
        elif isinstance(i, int):
            indices_ = [i]
        else:
            indices_ = list(i)

        if ignore_auto_alpha:
            self.eval_summary_ = self.eval_summary[self.sentiment].query(f'alpha != "auto"')
            # self.eval_summary_.index = range(len(self.eval_summary_))
        else:
            self.eval_summary_ = self.eval_summary[self.sentiment].copy()

        lda_vis_ = collections.OrderedDict()

        for idx in indices_:
            j = self.eval_summary_.index[idx]
            k = str(j).zfill(3)
            pkl_pathname = self.cd_models(self.sentiment, "vis", f"prepared_{k}.pkl", mkdir=True)

            if os.path.isfile(pkl_pathname) and not update:
                lda_vis = load_data(pkl_pathname, verbose=verbose)

            else:
                # Get (hyper-)parameters and the corresponding trained model
                min_count, threshold, corpus_prop, num_topics, alpha, eta, _ = \
                    self.eval_summary_.loc[j]

                lda_vis, _ = self._get_vis_data(
                    sentiment=self.sentiment, tokenized_docs=self.tokenized_docs,
                    corpus_prop=corpus_prop, min_count=min_count, threshold=threshold,
                    num_topics=num_topics, alpha=alpha, eta=eta, **kwargs)

            if export_to_html:
                self._export_vis_data_to_html(
                    ignore_auto_alpha=ignore_auto_alpha, vis_data=lda_vis, sentiment=self.sentiment,
                    i=idx, update=update, verbose=verbose)

            lda_vis_.update({f'LDA_{k}': lda_vis})

            del lda_vis
            gc.collect()

        return lda_vis_

    @staticmethod
    def _get_top_terms_of_topics(lda_model, lda_vis, num_terms, lambda_):
        top_terms_dict = collections.OrderedDict()

        for i in range(1, lda_model.num_topics + 1):
            k = 'Topic' + str(i)

            # Get data of the i-th topic
            topic_i = lda_vis.topic_info[lda_vis.topic_info['Category'] == k].copy()

            # Calculate relevance based on 'λ' (i.e. `lambda_`)
            topic_i['relevance'] = topic_i['loglift'] * (1 - lambda_) + topic_i['logprob'] * lambda_

            # Get the top `num_terms` terms of the topic
            temp = topic_i.sort_values('relevance', ascending=False, ignore_index=True)
            top_terms_dict[k] = temp.loc[:num_terms, 'Term'].values

        top_terms_of_topics = pd.DataFrame(top_terms_dict)

        return top_terms_of_topics

    def get_top_terms_of_topics(self, sentiment, i=None, ignore_auto_alpha=False, num_terms=15,
                                lambda_=0.0, vis_data_to_html=False, update=False, verbose=False,
                                **kwargs):
        # noinspection PyShadowingNames
        """
        Get the top ``num_terms`` terms for each topic.

        :param sentiment: label of sentiment;
            options include :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`
        :type sentiment: str
        :param i: row index or indices of the model evaluation summary, defaults to ``None``
        :type i: int or typing.Iterable or None
        :param ignore_auto_alpha: whether to ignore the situation when ``alpha='auto'``
        :type ignore_auto_alpha: bool
        :param num_terms: number of terms to be considered
        :type num_terms: int
        :param lambda_: lambda value for the LDA model
        :type lambda_: float or int
        :param vis_data_to_html: whether to save the model visualisation data to an HTML file,
            defaults to ``False``
        :type vis_data_to_html: bool
        :param update: whether to replace the existing HTML file with an updated one,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :param kwargs: [optional] parameters of `pyLDAvis.gensim_models.prepare()`_
        :return: the top ``num_terms`` terms for each topic
        :rtype: collections.OrderedDict

        .. _`pyLDAvis.gensim_models.prepare()`:
            https://pyldavis.readthedocs.io/en/latest/modules/API.html#pyLDAvis.prepare

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> top_terms = lda.get_top_terms_of_topics(sentiment='positive', i=76)
            >>> top_terms
            OrderedDict([('LDA_076',
                                       Topic1           Topic2          Topic3
                          0         large_dog        keep_zone    washable_pad
                          1         twice_day  firmware_update  authentic_part
                          2   hair_everywhere           define        chemical
                          3               mom       clean_zone    reusable_pad
                          4          dog_pick         firmware  disposable_pad
                          5    clean_everyday        cleanbase        damp_wet
                          6        life_saver  software_update    sweeping_pad
                          7         wish_soon            cloud          bottle
                          8         life_easy            width       streaking
                          9         lifesaver         homebase          bravva
                          10           obsess       map_create    dry_sweeping
                          11      amazed_pick             beam      change_pad
                          12        sweep_day           reboot    cleaning_pad
                          13            hairy             ugly          capful
                          14       clean_hair            remap          dilute
                          15         everyday        avoidance        reusable)])
        """
        # sentiment = 'positive'
        # num_terms = 15
        # lambda_ = 0.0

        self._prep_materials(sentiment=sentiment)

        if i is None:
            indices_ = range(10)
        elif isinstance(i, int):
            indices_ = [i]
        else:
            indices_ = list(i)

        if ignore_auto_alpha:
            self.eval_summary_ = self.eval_summary[self.sentiment].query(f'alpha != "auto"')
            # self.eval_summary_.index = range(len(self.eval_summary_))
        else:
            self.eval_summary_ = self.eval_summary[self.sentiment].copy()

        top_terms_of_topics = collections.OrderedDict()

        for idx in indices_:
            j = self.eval_summary_.index[idx]
            k = str(j).zfill(3)
            # Get (hyper-)parameters and the corresponding trained model
            min_count, threshold, corpus_prop, num_topics, alpha, eta, _ = self.eval_summary_.loc[j]

            lda_vis, lda_model = self._get_vis_data(
                sentiment=self.sentiment, tokenized_docs=self.tokenized_docs,
                corpus_prop=corpus_prop, min_count=min_count, threshold=threshold,
                num_topics=num_topics, alpha=alpha, eta=eta, **kwargs)

            if vis_data_to_html:
                self._export_vis_data_to_html(
                    ignore_auto_alpha=ignore_auto_alpha, vis_data=lda_vis, sentiment=self.sentiment,
                    i=idx, update=update, verbose=verbose)

            top_terms_of_topics_ = self._get_top_terms_of_topics(
                lda_model=lda_model, lda_vis=lda_vis, num_terms=num_terms, lambda_=lambda_)

            top_terms_of_topics[f'LDA_{k}'] = top_terms_of_topics_

            del lda_vis, lda_model
            gc.collect()

        return top_terms_of_topics

    def find_original_reviews(self, sentiment, i=None, num_terms=15, lambda_=0.0,
                              export_to_file=False, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Find original review texts containing terms that are most relevant to each topic,
        for the top 10 models given their coherence scores.

        :param sentiment: label of sentiment;
            options include :attr:`~src.modeller.LatentDirichletAllocation.VALID_SENTIMENT_LABELS`
        :type sentiment: str
        :param i: row index or indices of the model evaluation summary, defaults to ``None``
        :type i: int or typing.Iterable or None
        :param num_terms: number of terms to be considered
        :type num_terms: int
        :param lambda_: lambda value for the LDA model
        :type lambda_: float or int
        :param export_to_file: whether to save the results to a spreadsheet file, defaults to ``True``
        :type export_to_file: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :param kwargs: [optional] parameters of the method
            :meth:`~src.modeller.LatentDirichletAllocation.get_top_terms_of_topics`
        :return: topic-specific original review texts for the top 10 models given their coherence scores
        :rtype: collections.OrderedDict

        **Examples**::

            >>> from src.modeller import LatentDirichletAllocation
            >>> # Positive reviews:
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> # lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> pos_reviews = lda.find_original_reviews(
            ...     sentiment='positive', i=range(10), ignore_auto_alpha=True, verbose=True)
            >>> # Negative reviews:
            >>> lda = LatentDirichletAllocation('vacuum', product_type='robotic')
            >>> # lda = LatentDirichletAllocation('vacuum', product_type='traditional')
            >>> neg_reviews = lda.find_original_reviews(
            ...     sentiment='negative', i=range(10), ignore_auto_alpha=True, verbose=True)
        """

        match_columns = ['ASIN', 'Brand', 'ProductTitle', 'ParentID', 'ReviewerName', 'ReviewTitle']

        kwargs.update({'verbose': verbose})
        top_terms_of_topics = self.get_top_terms_of_topics(
            sentiment=sentiment, i=i, num_terms=num_terms, lambda_=lambda_, **kwargs)

        self.reviews.load_prep_data()
        prep_data = self.reviews.prep_data[match_columns + ['ReviewText']]
        prep_data.set_index(match_columns, inplace=True)

        self.reviews.load_preprocd_data()
        preprocd_data = self.reviews.preprocd_data.query(
            f'sentiment_on_vs_score == "{sentiment}"')[match_columns + ['review_text']]

        original_reviews = collections.OrderedDict()

        for k, dat in top_terms_of_topics.items():
            k_ = k.split('_')[-1]
            data = dat.applymap(lambda x: x.replace('_', ' '))

            orig_rev = collections.OrderedDict()
            for col in data.columns:
                selection = [x for x in data[col] if x]
                pattern = '|'.join(r"\b{}\b".format(x) for x in selection)  # '|'.join(selection)
                temp_ = preprocd_data[preprocd_data['review_text'].str.contains(pattern)]
                temp = temp_.join(prep_data, on=match_columns)[['ReviewText']]
                orig_rev[col] = temp

            if export_to_file:
                temp_pathname = self.cd_models(sentiment, "vis", f"prepared_{k_}_raw_reviews.xlsx")
                save_spreadsheets(
                    list(orig_rev.values()), path_to_file=temp_pathname,
                    sheet_names=list(orig_rev.keys()), index=False, mode='a',
                    if_sheet_exists='replace', verbose=verbose)

            original_reviews[k] = orig_rev

        return original_reviews


if __name__ == '__main__':
    from src.modeller import LatentDirichletAllocation
    from pyhelpers.settings import pd_preferences

    pd_preferences()

    lda = LatentDirichletAllocation('therms', product_type='smart')

    pos_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='positive')

    neg_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='negative')

    if confirmed("To proceed?"):

        # pos_lda_vis_data_1 = lda.get_vis_data(
        #     sentiment='positive', i=0, export_to_html=True, verbose=True)
        # pos_lda_vis_data_2 = lda.get_vis_data(
        #     sentiment='positive', i=range(50), export_to_html=True, verbose=True)
        pos_lda_vis_data_3 = lda.get_vis_data(
            sentiment='positive', i=range(10), export_to_html=True, verbose=True,
            ignore_auto_alpha=True)
        pos_reviews = lda.find_original_reviews(
            sentiment='positive', i=range(10), ignore_auto_alpha=True, export_to_file=True,
            verbose=True)

        # neg_lda_vis_data_1 = lda.get_vis_data(
        #     sentiment='negative', i=0, export_to_html=True, verbose=True)
        # neg_lda_vis_data_2 = lda.get_vis_data(
        #     sentiment='negative', i=range(50), export_to_html=True, verbose=True)
        neg_lda_vis_data_3 = lda.get_vis_data(
            sentiment='negative', i=range(10), export_to_html=True, verbose=True,
            ignore_auto_alpha=True)
        neg_reviews = lda.find_original_reviews(
            sentiment='negative', i=range(10), ignore_auto_alpha=True, export_to_file=True,
            verbose=True)
