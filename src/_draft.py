"""
Draft script.
"""

from pyhelpers.settings import pd_preferences

from src.modeller import LatentDirichletAllocation

pd_preferences()

# ==================================================================================================

lda = LatentDirichletAllocation(product_type='robotic', load_preprocd_data=False)

neg_sum = lda.fetch_evaluation_summary(sentiment='negative')

import matplotlib.pyplot as plt
from pyhelpers.settings import mpl_preferences

mpl_preferences()

fig = plt.figure()
ax = fig.add_subplot()

y_col = 'coherence_score'
x_col = 'num_topics'
group_by = ['eval_corpus', 'eta']  # 'alpha',

# 'trigram_min_count' == 1, 'trigram_threshold' == 0.0001
dat = neg_sum.query('trigram_min_count == 1 and trigram_threshold == 0.0001')
dat = dat.groupby(group_by)[[x_col, y_col]].apply(dict)

import seaborn as sns

pp = sns.pairplot(dat[group_by], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)


# == Update "eval_summary_10.pkl" =======================================================================

import gensim
import warnings
import pyLDAvis.gensim_models
from pyhelpers.store import save_data

sentiment = 'negative'  # 'positive'
indices = None
indices_ = range(10)

lda = LatentDirichletAllocation(product_type='robotic')

eval_result_summary = lda.fetch_evaluation_summary(sentiment=sentiment)
# eval_result_summary = lda.fetch_evaluation_summary(sentiment=sentiment, incl_model=True)

data_ = lda.data[lda.data[lda.sentiment_column_name] == sentiment]
docs = data_[lda.review_column_name]
tokenized_docs = lda.get_tokenized_docs(docs=docs, sentiment=sentiment)

err_idx = []
for i in indices_:
    # Get (hyper-)parameters and the corresponding trained model
    corpus_proportion, num_topics, alpha, eta, lda_model, _, min_count, threshold, = \
        eval_result_summary.loc[i, :]

    corpus, id2word, texts = lda.make_corpus(
        tokenized_docs=tokenized_docs, ngram=3, min_count=min_count, threshold=threshold)

    if corpus_proportion != '100%':
        proportion = lda._parse_proportion(corpus_proportion)
        max_docs = int(len(corpus) * proportion)
        corpus = gensim.utils.ClippedCorpus(corpus=corpus, max_docs=max_docs)

    lda_model_ = lda_model

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=DeprecationWarning)

        try:
            lda_vis = pyLDAvis.gensim_models.prepare(
                topic_model=lda_model_, corpus=corpus, dictionary=id2word, sort_topics=False)

        except pyLDAvis._prepare.ValidationError:
            err_idx.append(i)

            train_lda_args = {
                'corpus': corpus, 'id2word': id2word, 'texts': texts,
                'num_topics': num_topics, 'alpha': alpha, 'eta': eta}
            lda_model_ = lda._retrain_model(**train_lda_args)

            lda_vis = pyLDAvis.gensim_models.prepare(
                topic_model=lda_model_, corpus=corpus, dictionary=id2word, sort_topics=False)

            # Update the invalid model
            eval_result_summary.loc[i, 'lda_model'] = lda_model_


path_to_summary_10 = lda.cd_models(sentiment, "eval_summary_10.pkl")
save_data(eval_result_summary, path_to_summary_10, verbose=True)
