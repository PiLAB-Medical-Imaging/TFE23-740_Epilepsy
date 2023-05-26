from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

class SequentialFeatureSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    
    def __init__(self, estimator):
        pass

    def fit(self, X, y=None):
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_


