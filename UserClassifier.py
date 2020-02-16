"""
Can send in updated data directly into this model.
Performs processing, fit and prediction for the Ad classifier dataset.
Author:
    Sabareesh Mamidipaka
"""

__all__ = ['UserClassifier']

from comp_func import *
from nltk.corpus import stopwords
from itertools import islice

class UserClassifier():
    
    def __init__(self, param_grid = None, model = 'log_re'):
        """
        param_grid: Parameter grid to perform validation. Given in the form of dict.
                    Default is None in which case the standard parameter grid is used for validation.
        model: The model name as a string. Can be one of 'log_re', 'dec_tree' and 'rf'.
               Default value is 'log_re' corresponding to logistic regression. 
        """
        if param_grid == None:
            self.param_grid = get_param_map(model)
        else:
            self.param_grid = param_grid
        self.model = get_model_map(model)
        
    def __repr__(self):
        return "I'm a classifier model to predict if a user will click on the data or not. \
                I'm a Logistic regression model unless you specified otherwise. \
                Imputaion in training set is done by using mean for numerical values and mode for\
                categorical variables to avoid any leakage of data."
    
    def pre_process(self, data : pd.DataFrame):
        """
        data: dataframe to perform cleaning on.
        
        Drops null values, duplicated rows and rows which maybe corrupted.
        """
        data.dropna(axis=0, inplace=True)
        data = data.drop_duplicates()
        data = data[(data['Age'] >= 18) & (data['Age'] < 100)]
        data = data[(data['Daily Internet Usage'] > data['Daily Time Spent on Site'])]
        
    def feature_engineering(self, data : pd.DataFrame):
        """
        data: dataframe to perform feature engineering.
        Adds extracted information as column and drops unnecessary columns.
        """
        data['influential_words'] = data['Ad Topic Line'].apply(lambda x: \
                                                                if_influential_words(x, self.most_used_yes, self.most_used_no))
        data.drop(["Ad Topic Line", "Timestamp", "City"], axis=1, inplace= True)
               
    def scaling(self):
        """
        performs the scaling of data inplace.
        """
        numerical_vars = ["Daily Time Spent on Site", "Area Income", "Daily Internet Usage", "Age"]
        self.scaler = Data_Scaler(self.data, numerical_vars) 
        self.data[numerical_vars] = self.scaler.transform_data(self.data[numerical_vars])
    
    def dummies(self, data : pd.DataFrame):
        """
        data: data containing variables to perform one hot encoding
        
        returns: one hot encoded data of all categorical variables
        """
        return pd.get_dummies(data)
        
    def find_influential_words(self, data : pd.DataFrame):
        """
        data: find most used words in the dataset after removing stopwords affecting positively and negatively.
        """
        cachedStopWords = stopwords.words("english")

        result  = [word for word in data[data[self.target]==1]['Ad Topic Line']]
        yes_words = ' '.join(result)

        result  = [word for word in data[data[self.target]==0]['Ad Topic Line']]
        no_words = ' '.join(result)

        # WordCloud has been selected to get the most influential words because of their ability
        # to sort and group words efficiently.
        yes_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, \
                             stopwords=cachedStopWords).generate(yes_words)        
        no_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, \
                             stopwords=cachedStopWords).generate(no_words)

        self.most_used_no = set(islice(no_cloud.words_, 10))
        self.most_used_yes = set(islice(yes_cloud.words_, 10))
    
    def process_train(self, data : pd.DataFrame):
        """
        processing step for training data.
        """
        self.pre_process(data)
        self.find_influential_words(data)
        self.feature_engineering(data)
        self.scaling()
        return self.dummies(data)
    
    def process_test(self, data : pd.DataFrame):
        """
        processing step for test data.
        """
        
        #Fill the missing values with mean and mode for numeric and categorical variables.
        data = data.fillna(self.train_mean).fillna(self.train_mode)
        self.feature_engineering(data)
        data = self.scaler.transform_data(data)
        return self.dummies(data)
    
    def add_missing_dummy_columns(self, data : pd.DataFrame):
        """
        data: test data after performing 
        """
        missing_cols = set(self.columns) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
    
    def fix_columns(self, data : pd.DataFrame):
        """
        data: rearrange the columns of the dataframe to align with the original training set.
        """
        self.add_missing_dummy_columns(data)
        data = data[self.columns]
        return data
        
    def validation(self):
        """
        Perform cross validation and returns the best model.
        """
        grid_obj = GridSearchCV(self.model, param_grid=self.param_grid, cv=5)
        grid_fit = grid_obj.fit(self.data.drop([self.target], axis=1), self.data[self.target])
        return grid_fit.best_estimator_

    def fit(self, data : pd.DataFrame, target : str, validation = True):
        """
        data : training dataframe.
        target: target column of the dataframe.
        """
        
        # Make sure the original dataframe is not modified.
        self.data = data.copy()
        self.target = target
        
        # Fit on the training data
        self.data = self.process_train(self.data)
        self.model.fit(self.data.drop([target], axis=1), self.data[target])
        
        # Perform validation
        if validation:
            self.model = self.validation()
                
        # Storing the statistics of training data (without the target column) in case we need imputation on the test set.
        self.train_mean = self.data.drop([target], axis=1).mean()
        self.train_mode = self.data.drop([target], axis=1).mode().iloc[0]
        self.columns = self.data.drop([target], axis=1).columns

        # Predictions on the training data
        self.fitted_values = self.predict(data.drop([target], axis=1))  
        
    def predict(self, data : pd.DataFrame, threshold=0.4):
        """
        data : test data to predict.
        
        returns: predictions.
        """
        
        # Processing test data
        data = self.process_test(data.copy())
        data = self.fix_columns(data)
        
        # Predictions depending on the threshold.
        test_pred_prob = self.model.predict_proba(data)[:,1]
        return np.where(test_pred_prob<threshold, 0, 1)
