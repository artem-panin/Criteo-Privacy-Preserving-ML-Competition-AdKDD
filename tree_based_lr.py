from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import numpy as np


# class for the tree-based/logistic regression pipeline
class TreeBasedLR:
    
    # initialization
    def __init__(self, forest_params, lr_params, forest_model):
        
        # storing parameters
        self.forest_params = forest_params
        self.lr_params = lr_params 
        self.forest_model = forest_model
        
    # method for fitting the model
    def fit(self, X, y, sample_weight=None):
        
        # dict for finding the models
        forest_model_dict = {'lgbm': LGBMClassifier}
        
        # configuring the models
        self.lr = LogisticRegression(**self.lr_params)
        self.forest = forest_model_dict[self.forest_model](**self.forest_params)
        self.classes_ = np.unique(y)
        
        # first, we fit our tree-based model on the dataset
        self.forest.fit(X, y)
        
        # then, we apply the model to the data in order to get the leave indexes
        leaves = self.forest.predict(X, pred_leaf=True)
        
        # then, we one-hot encode the leave indexes so we can use them in the logistic regression
        self.encoder = OneHotEncoder()
        leaves_encoded = self.encoder.fit_transform(leaves)
        
        # and fit it to the encoded leaves
        self.lr.fit(leaves_encoded, y)
        
    # method for predicting probabilities
    def predict_proba(self, X):
        
        # then, we apply the model to the data in order to get the leave indexes
        leaves = self.forest.predict(X, pred_leaf=True)
        
        # then, we one-hot encode the leave indexes so we can use them in the logistic regression
        leaves_encoded = self.encoder.transform(leaves)
        
        # and fit it to the encoded leaves
        y_hat = self.lr.predict_proba(leaves_encoded)
        
        # retuning probabilities
        return y_hat
    
    # get_params, needed for sklearn estimators
    def get_params(self, deep=True):
        return {'forest_params': self.forest_params,
                'lr_params': self.lr_params,
                'forest_model': self.forest_model}