"""
Independent functions necessary for user classification to be performed.
Author:
    Sabareesh Mamidipaka
"""

# import necessary functions
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import re
import numpy as np

# Function takes each row of the Ad topic and returns 1, -1 and 0 correponding to presence of 
# influential positive, negative and neutral words.
def if_influential_words(x : str, yes_words : list, no_words : list):
    """
    x: Ad topic line
    yes_words: List of positive influential words based on the training data
    no_words: List of negative influential words based on the training data
    
    returns: 1, -1 or 0
    """
    if any([keyword.casefold() in x for keyword in yes_words]):
        return 1
    elif any([keyword.casefold() in x for keyword in no_words]):
        return -1
    return 0

# returns the parameter grid depending on the model. Can be specified by the user to override these values.
def get_param_map(model: str):
    """
    model: The name of the model given as a string.
    
    returns: parameter grid corresponding to the model.
    """
    param_grid_map = {'log_re': {"C": [0.01, 0.5, 1, 5, 10]}, \
                  'dec_tree': {'max_depth': [10, 20, 40],
                              'min_samples_split': [2, 4, 8],
                              'min_samples_leaf': [1, 3, 6]}, \
                  'rf': {'n_estimators': [250, 500, 1000],
                          'max_features': [3, 4, 6],
                          'max_depth': [10, 20, 40],
                          'min_samples_split': [2, 4, 8],
                          'min_samples_leaf': [1, 3, 6]}}
    return param_grid_map[model]
    
def get_model_map(model : str):
    """
    model: The name of the model given as a string.
    
    returns: Corresponding sklearn model
    """
    model_map = {'log_re': LogisticRegression(random_state=42, penalty="l2", solver='lbfgs'),\
                 'dec_tree': DecisionTreeClassifier(),\
                 'rf': RandomForestClassifier()}
    return model_map[model]

class Data_Scaler(object):
    """
    Performs min max scaling of a dataframe for the given columns.
    """
    
    def __init__(self, data_frame : pd.DataFrame, cols : list):
        """
        data_frame: The data frame w.r.t to which the scaling needs to perform. (training data)
        cols: List of column names to be scaled (most probably all the numerical variables apart from target)
        """
        self.data = data_frame[cols]
        self.min_data = np.min(self.data)
        self.max_data = np.max(self.data)
        self.cols = cols
    
    # Scales down the values in the dataframe.
    def transform_data(self, data_frame_to_scale : pd.DataFrame):
        """
        data_frame_to_scale: The dataframe that needs to be transformed.
        
        returns: Scaled dataframe in place.
        """
        data_frame_to_scale[self.cols] = (data_frame_to_scale[self.cols] - self.min_data)/(self.max_data-self.min_data)
        return data_frame_to_scale
    
    # inverse scales the values in the dataframe
    def inverse_transform_data(self, data_frame_to_inv):
        """
        data_frame_to_inv: The dataframe to perform inverse transform if needed.
        
        returns: Inverse scaled dataframe in place.
        """
        data_frame_to_inv[self.cols] = data_frame_to_inv[self.cols]*(self.max_data-self.min_data)+self.min_data
        return data_frame_to_inv
