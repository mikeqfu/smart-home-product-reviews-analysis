"""
Run tasks.
"""

if __name__ == '__main__':
    from pyhelpers.settings import pd_preferences

    pd_preferences()

    # from src.modeller import ToyExamples
    # from pyhelpers.settings import pd_preferences
    # pd_preferences()
    # toy = ToyExamples()
    # toy.logistic_regression(random_state=0)
    # print('Mean accuracy: %.2f%%' % (toy.score * 100))

if __name__ == '__main__':
    from src.modeller import LatentDirichletAllocation

    # lda = LatentDirichletAllocation(product_category='vacuum', product_type='robotic')
    # lda.evaluate_models()

    lda = LatentDirichletAllocation('vacuum', product_type='robotic', load_preprocd_data=False)
    pos_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='positive')
    print("\n", pos_lda_eval_summary)
    neg_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='negative')
    print("\n", neg_lda_eval_summary)

    # lda = LatentDirichletAllocation(product_category='vacuum', product_type='traditional')
    # lda.evaluate_models()

    lda = LatentDirichletAllocation('vacuum', product_type='traditional', load_preprocd_data=False)
    pos_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='positive')
    print("\n", pos_lda_eval_summary)
    neg_lda_eval_summary = lda.fetch_evaluation_summary(sentiment='negative')
    print("\n", neg_lda_eval_summary)
