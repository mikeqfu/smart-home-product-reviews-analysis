"""
Prepare data sets for conducting fuzzy-set qualitative comparative analysis (fsQCA).
"""


import collections

import numpy as np
import pandas as pd
from pyhelpers.store import save_data


# noinspection PyShadowingNames
def get_doc_topic_probs(lda, sentiment, best_lda_index, incl_original_reviews=True,
                        further_screen=True):
    """
    Get topic probabilities for each review.

    :param lda: An instance of the `~src.modeller.lda.LatentDirichletAllocation` class.
    :type lda: LatentDirichletAllocation
    :param sentiment: Sentiment label.
    :type sentiment: str
    :param best_lda_index:
    :param incl_original_reviews: Whether to include original review texts; defaults to ``True``.
    :type incl_original_reviews: bool
    :param further_screen: Whether to further screen the data of documents for fsQCA;
        defaults to ``True``. This process aims to ensure that vader scores
        (calculated based on the original review texts) also reflect the true sentiment of
        the corresponding cleansed tokens in the returned data set.
    :type further_screen: bool
    :return: Data of topic probabilities for each review.

    **Examples**::

        >>> from src.analyser import get_doc_topic_probs
        >>> from src.modeller import LatentDirichletAllocation
        >>> lda = LatentDirichletAllocation(product_category='vacuum', product_type='robotic')
        >>> doc_topic_probs = get_doc_topic_probs(lda, sentiment='positive', best_lda_index=76)
        >>> doc_topic_probs.head()
            topic_1   topic_2  ...  vs_compound_score new_vs_compound_score
        0  0.968450  0.019314  ...             0.9781                0.7430
        1  0.918801  0.048385  ...             0.9591                0.4019
        2  0.922607  0.045102  ...             0.3971                0.5719
        3  0.961893  0.022935  ...             0.9754                0.4939
        4  0.971785  0.017273  ...             0.7787                0.6249
        [5 rows x 7 columns]
        >>> doc_topic_probs = get_doc_topic_probs(lda, sentiment='negative', best_lda_index=0)
        >>> doc_topic_probs.head()
           topic_1   topic_2  ...  vs_compound_score new_vs_compound_score
        0      0.0  0.356828  ...            -0.4359               -0.5719
        1      0.0  0.116166  ...            -0.4012               -0.3182
        2      0.0  0.978022  ...            -0.4779               -0.5719
        3      0.0  0.981127  ...            -0.7944               -0.2960
        4      0.0  0.983219  ...            -0.9284               -0.5574
        [5 rows x 7 columns]
        >>> lda = LatentDirichletAllocation(product_category='vacuum', product_type='traditional')
        >>> doc_topic_probs = get_doc_topic_probs(lda, sentiment='positive', best_lda_index=9)
        >>> doc_topic_probs.head()
            topic_1   topic_2  ...  vs_compound_score new_vs_compound_score
        0  0.713384  0.283967  ...             0.7389                0.5859
        1  0.396577  0.601326  ...             0.9002                0.7264
        2  0.529634  0.468892  ...             0.8805                0.5106
        3  0.544901  0.451072  ...             0.6908                0.6908
        4  0.450820  0.546059  ...             0.6597                0.0772
        [5 rows x 7 columns]
        >>> doc_topic_probs = get_doc_topic_probs(lda, sentiment='negative', best_lda_index=49)
        >>> doc_topic_probs.head()
            topic_1   topic_2  ...  vs_compound_score new_vs_compound_score
        0  0.271696  0.081394  ...            -0.1923               -0.7269
        1  0.025579  0.635595  ...            -0.6590               -0.4927
        2  0.017298  0.529355  ...            -0.8078               -0.5574
        3  0.260156  0.219738  ...            -0.8118               -0.5994
        4  0.214129  0.248306  ...            -0.8436               -0.4536
        [5 rows x 7 columns]
        >>> lda = LatentDirichletAllocation(product_category='thermostats', product_type='smart')
        >>> doc_topic_probs = get_doc_topic_probs(lda, sentiment='positive', best_lda_index=19)
        >>> doc_topic_probs.head()
            topic_1   topic_2  ...  vs_compound_score new_vs_compound_score
        0  0.206133  0.000000  ...             0.9902                0.7269
        1  0.949537  0.029732  ...             0.8849                0.3612
        2  0.960210  0.023300  ...             0.8915                0.0516
        3  0.052526  0.928535  ...             0.9692                0.4019
        4  0.974451  0.015015  ...             0.7351                0.5106
        [5 rows x 7 columns]
        >>> doc_topic_probs = get_doc_topic_probs(lda, sentiment='negative', best_lda_index=9)
        >>> doc_topic_probs.head()
            topic_1   topic_2  ...  vs_compound_score new_vs_compound_score
        0  0.919857  0.039846  ...            -0.6863               -0.4767
        1  0.000000  0.000000  ...            -0.7975               -0.5423
        2  0.967758  0.015943  ...            -0.9205               -0.3612
        3  0.922046  0.039310  ...            -0.1280               -0.1280
        4  0.096724  0.531999  ...            -0.4310               -0.3612
        [5 rows x 7 columns]
    """

    lda._assert_sentiment(sentiment)

    tokenized_docs = lda.get_tokenized_docs(lda.data[lda.review_column_name], sentiment)

    eval_summary = lda.fetch_evaluation_summary(sentiment=sentiment)

    params_columns = ['min_count', 'threshold', 'num_topics', 'alpha', 'eta', 'corpus_proportion']

    min_count, threshold, num_topics, alpha, eta, corpus_prop = (
        eval_summary.loc[best_lda_index, params_columns].values)

    corpus, prop, id2word, texts = lda._make_eval_corpus(
        sentiment=sentiment, tokenized_docs=tokenized_docs, corpus_prop=corpus_prop,
        min_count=min_count, threshold=threshold, ngram=3)

    lda_model = lda._train_model(
        corpus=corpus, id2word=id2word, num_topics=num_topics, alpha=alpha, eta=eta)

    # Get the topic probabilities for each document
    doc_topic_probs_dat = [{k: v for k, v in lda_model.get_document_topics(bow)} for bow in corpus]

    # Convert the list of dictionaries into a pandas DataFrame
    doc_topic_probs_0 = pd.DataFrame(doc_topic_probs_dat).sort_index(axis=1).fillna(0)

    doc_topic_probs_0.columns = [f'topic_{i + 1}' for i in doc_topic_probs_0.columns]

    original_texts = lda.retrieve_original_text(corpus=corpus, id2word=id2word, texts=texts)
    if not incl_original_reviews:
        original_texts.drop(columns=['ReviewText'], inplace=True)

    if further_screen:
        if sentiment == 'positive':
            original_texts = original_texts[original_texts['new_vs_compound_score'] >= 0.05]
        elif sentiment == 'negative':
            original_texts = original_texts[original_texts['new_vs_compound_score'] <= -0.05]

    doc_topic_probs_1 = doc_topic_probs_0.loc[original_texts.index]

    doc_topic_probs = pd.concat([doc_topic_probs_1, original_texts], axis=1).reset_index(drop=True)

    return doc_topic_probs


def calibrate_fuzzy_membership(prob, full_non_membership, crossover, full_membership):
    # noinspection PyShadowingNames
    """
    Calibrates the probability into fuzzy-set membership score.

    :param prob: The topic probability to calibrate (between 0 and 1)
    :type prob: int | float
    :param full_non_membership: The threshold for full non-membership (calibrated to 0)
    :type full_non_membership: int | float
    :param crossover: The crossover point where membership is 0.5
    :type crossover: int | float
    :param full_membership: The threshold for full membership (calibrated to 1)
    :type full_membership: int | float
    :return: Fuzzy-set membership score (between 0 and 1)
    :rtype: float

    **Examples**::

        >>> from src.analyser import calibrate_fuzzy_membership, get_doc_topic_probs
        >>> from src.modeller import LatentDirichletAllocation
        >>> import numpy as np
        >>> lda = LatentDirichletAllocation(product_category='vacuum', product_type='robotic')
        >>> doc_topic_probs = get_doc_topic_probs(lda, sentiment='positive', best_lda_index=76)
        >>> data = doc_topic_probs.copy()
        >>> # topic_col_names = [x for x in data.columns if x.startswith('topic_')]
        >>> topic_col_names = ['topic_1', 'topic_2', 'topic_3']
        >>> for col in topic_col_names + ['rating', 'vs_compound_score', 'new_vs_compound_score']:
        ...     f_n_membership, crossover_point, f_membership = map(
        ...         lambda x: np.percentile(data[col].values, x), [5, 50, 95])
        ...     data[col + '_fuzzy'] = data[col].map(
        ...         lambda prob: calibrate_fuzzy_membership(
        ...             prob, f_n_membership, crossover_point, f_membership)).values
        >>> data.columns.tolist()
        ['topic_1',
         'topic_2',
         'topic_3',
         'ReviewText',
         'rating',
         'vs_compound_score',
         'new_vs_compound_score',
         'topic_1_fuzzy',
         'topic_2_fuzzy',
         'topic_3_fuzzy',
         'rating_fuzzy',
         'vs_compound_score_fuzzy',
         'new_vs_compound_score_fuzzy']
    """

    if prob <= full_non_membership:
        return 0
    elif prob >= full_membership:
        return 1
    else:
        # Logistic transformation for fuzzy membership calibration
        return 1 / (1 + np.exp(- (prob - crossover) * 10 / (full_membership - full_non_membership)))


# noinspection PyShadowingNames
def prep_fsqca_data(doc_topic_probs, lda, sentiment, save_res=True):
    """
    Prepare data sets for fsQCA.

    :param doc_topic_probs: Data of topic probabilities for each review.
    :type doc_topic_probs: pandas.DataFrame
    :param lda: An instance of the `~src.modeller.lda.LatentDirichletAllocation` class.
    :type lda: LatentDirichletAllocation
    :param sentiment: Sentiment label.
    :type sentiment: str
    :param save_res: Whether to save the prepared data sets as CSV files; defaults to ``True``.
    :type save_res: bool
    :return: Data sets ready for fsQCA.
    :rtype: collections.OrderedDict

    **Examples**::

        >>> from src.analyser import prep_fsqca_data, get_doc_topic_probs
        >>> from src.modeller import LatentDirichletAllocation
        >>> lda = LatentDirichletAllocation(product_category='vacuum', product_type='robotic')
        >>> sentiment = 'positive'
        >>> best_lda_index = 76
        >>> doc_topic_probs = get_doc_topic_probs(lda, sentiment, best_lda_index)
        >>> doc_topic_probs.head()
            topic_1   topic_2  ...  vs_compound_score new_vs_compound_score
        0  0.968435  0.019337  ...             0.9781                0.7430
        1  0.918798  0.048387  ...             0.9591                0.4019
        2  0.922610  0.045099  ...             0.3971                0.5719
        3  0.961892  0.022936  ...             0.9754                0.4939
        4  0.971793  0.017264  ...             0.7787                0.6249
        [5 rows x 7 columns]
        >>> fsqca_prep_data = prep_fsqca_data(doc_topic_probs, lda, sentiment, save_res=False)
        >>> list(fsqca_prep_data.keys())
        ['robotic-positive-doc_topic_probs_5_10_25',
         'robotic-positive-doc_topic_probs_5_25_50',
         'robotic-positive-doc_topic_probs_5_50_95']
    """

    percentile_lists = [
        [5, 10, 25],
        [5, 25, 50],
        [5, 50, 95],
    ]

    fsqca_prep_data = collections.OrderedDict()

    for p_list in percentile_lists:
        # p1, p2, p3 = p_list

        data = doc_topic_probs.copy()

        topic_col_names = [x for x in data.columns if x.startswith('topic_')]

        for col in topic_col_names + ['rating', 'vs_compound_score', 'new_vs_compound_score']:
            f_n_membership, crossover_point, f_membership = map(
                lambda x: np.percentile(data[col].values, x), p_list)

            data[col + '_fuzzy'] = data[col].map(
                lambda prob: calibrate_fuzzy_membership(
                    prob, f_n_membership, crossover_point, f_membership)).values

        filename_ = f"{lda.product_type.lower()}-{sentiment}-doc_topic_probs"
        key = filename_ + "_" + "_".join([str(p) for p in p_list])

        fsqca_prep_data[key] = data

        if save_res:
            save_data(data, lda.cd_models(sentiment, key + ".csv"), verbose=True)

    return fsqca_prep_data


# if __name__ == '__main__':
#     from pyhelpers.store import load_data
#     from src.modeller import LatentDirichletAllocation
#
#     case_params_set = load_data("src/data/case_params_set.json")
#
#     for case, case_params in case_params_set.items():
#         product_category = case_params['product_category']
#         product_type = case_params['product_type']
#         random_state = case_params['random_state']
#         sentiment = case_params['sentiment']
#         best_lda_index = case_params['best_lda_index']
#
#         lda = LatentDirichletAllocation(
#             product_category=product_category, product_type=product_type)
#
#         doc_topic_probs = get_doc_topic_probs(
#             lda, sentiment=sentiment, best_lda_index=best_lda_index, incl_original_reviews=False,
#             further_screen=True)
#
#         fsqca_data = prep_fsqca_data(
#             doc_topic_probs=doc_topic_probs, lda=lda, sentiment=sentiment, save_res=False)
