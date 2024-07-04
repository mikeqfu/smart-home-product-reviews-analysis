"""Test the module :py:mod:`~src.modeller`."""

import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from src.modeller import LatentDirichletAllocation, LogisticRegressionModel


class TestLogisticRegressionModel:

    def test_logistic_regression(self):
        lgr = LogisticRegressionModel(product_category='vacuum', product_type='Robotic')

        lgr.logistic_regression()

        assert isinstance(lgr.word_counter, csr_matrix)
        assert isinstance(lgr.logit, LogisticRegression)
        assert isinstance(lgr.score, float)
        assert isinstance(lgr.coefficients, list)
        assert isinstance(lgr.odds_ratios, list)
        assert isinstance(lgr.summary, pd.DataFrame)


class TestLatentDirichletAllocation:
    lda = LatentDirichletAllocation(product_category='vacuum', product_type='robotic')
    example_docs = lda.data[lda.review_column_name]

    def test_get_tokens(self):
        example_doc_tokens = self.lda.get_tokens(self.example_docs[0], bespoke_stopwords=None)
        assert isinstance(example_doc_tokens, list)
        assert all(isinstance(x, str) for x in example_doc_tokens)

    def test_specify_adhoc_stopwords(self):
        rslt = self.lda.specify_adhoc_stopwords()
        assert isinstance(rslt, set)
        assert len(rslt) >= 5000

    @pytest.mark.parametrize('sentiment', ['positive', 'negative'])
    def test_get_tokenized_docs(self, sentiment):
        tokenized_docs = self.lda.get_tokenized_docs(self.example_docs, sentiment=sentiment)
        assert tokenized_docs == getattr(self.lda, f'{sentiment[:3]}_tokenized_docs')
        assert len(tokenized_docs) == len(self.example_docs)
        assert all([isinstance(x, list) for x in tokenized_docs])


if __name__ == '__main__':
    pytest.main()
