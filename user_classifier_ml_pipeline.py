"""
Can send in updated data directly into this model.
Performs processing, fit and prediction for the Ad classifier dataset.
Author:
    Sabareesh Mamidipaka
"""

__all__ = ['Ad_Classifier']


def get_contain(x, word_list: list):
    return any([keyword.casefold() in x for keyword in word_list])

def if_influential_words(x, yes_words:list, no_words:list):
    if any([keyword.casefold() in x for keyword in yes_words]):
        return 1
    elif any([keyword.casefold() in x for keyword in no_words]):
        return -1
    return 0

def get_param_map(model):
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
    
def get_model_map(model):
    model_map = {'log_re': LogisticRegression(random_state=42, penalty="l2", solver='lbfgs'),\
                 'dec_tree': DecisionTreeClassifier(),\
                 'rf': RandomForestClassifier()}
    return model_map[model]

class Data_Scaler(object):
    
    def __init__(self, data_frame : pd.DataFrame, cols : list):
        """
        Takes in the dataframe and the list with column names that need to be scaled
        """
        self.data = data_frame[cols]
        self.min_data = np.min(self.data)
        self.max_data = np.max(self.data)
        self.cols = cols
    
    # Scales down the values in the dataframe.
    def transform_data(self, data_frame_to_scale):
        data_frame_to_scale[self.cols] = (data_frame_to_scale[self.cols] - self.min_data)/(self.max_data-self.min_data)
        return data_frame_to_scale
    
    # inverse scales the values in the dataframe
    def inverse_transform_data(self, data_frame_to_inv):
        data_frame_to_inv[self.cols] = data_frame_to_inv[self.cols]*(self.max_data-self.min_data)+self.min_data
        return data_frame_to_inv
    
class Ad_Classifier():
    
    def __init__(self, param_grid = None, threshold=0.4, model = 'log_re'):

        if param_grid == None:
            self.param_grid = get_param_map(model)
        else:
            self.param_grid = param_grid
        self.model = get_model_map(model)
        
    def __repr__(self):
        return "I'm a classifier model to predict if a user will click on the data or not. \
                I'm a Logistic regression model unless you specified otherwise. \
                Please don't give me null values. I'll get updated later to account for the null values."
    
    def pre_process(self, data):
        data.dropna(axis=0, inplace=True)
        data = data.drop_duplicates()
        data = data[(data['Age'] >= 18) & (data['Age'] < 100)]
        data = data[(data['Daily Internet Usage'] > data['Daily Time Spent on Site'])]
        
    def feature_engineering(self, data):
        data['influential_words'] = data['Ad Topic Line'].apply(lambda x: \
                                                                if_influential_words(x, self.most_used_yes, self.most_used_no))
        data.drop(["Ad Topic Line", "Timestamp", "City"], axis=1, inplace= True)
               
    def scaling(self):
        numerical_vars = ["Daily Time Spent on Site", "Area Income", "Daily Internet Usage", "Age"]
        self.scaler = Data_Scaler(self.data, numerical_vars) 
        self.data[numerical_vars] = self.scaler.transform_data(self.data[numerical_vars])
    
    def dummies(self, data):
        return pd.get_dummies(data)
        
    def find_influential_words(self, data):
        cachedStopWords = stopwords.words("english")

        result  = [word for word in data[data[self.target]==1]['Ad Topic Line']]
        yes_words = ' '.join(result)

        result  = [word for word in data[data[self.target]==0]['Ad Topic Line']]
        no_words = ' '.join(result)

        yes_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, \
                             stopwords=cachedStopWords).generate(yes_words)        
        no_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, \
                             stopwords=cachedStopWords).generate(no_words)

        self.most_used_no = set(islice(no_cloud.words_, 10))
        self.most_used_yes = set(islice(yes_cloud.words_, 10))
    
    def process_train(self, data):
        self.pre_process(data)
        self.find_influential_words(data)
        self.feature_engineering(data)
        self.scaling()
        return self.dummies(data)
    
    def process_test(self, data):
        self.feature_engineering(data)
        data = self.scaler.transform_data(data)
        return self.dummies(data)
    
    def add_missing_dummy_columns(self, data):
        missing_cols = set(self.columns) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
    
    def fix_columns(self, data):  
        self.add_missing_dummy_columns(data)
        data = data[self.columns]
        return data
        
    def validation(self):
        grid_obj = GridSearchCV(self.model, param_grid=self.param_grid, cv=5)
        grid_fit = grid_obj.fit(self.data.drop([target], axis=1), self.data[target])
        return grid_fit.best_estimator_

    def fit(self, data : pd.DataFrame, target : str, validation = True):
        self.data = data.copy()
        self.target = target
        self.data = self.process_train(self.data)
        self.model.fit(self.data.drop([target], axis=1), self.data[target])
        if validation:
            self.model = self.validation()
        self.columns = self.data.drop([target], axis=1).columns
        self.fitted_values = self.predict(data.drop([target], axis=1))
        
    def predict(self, data):
        data = self.process_test(data.copy())
        data = self.fix_columns(data)
        return self.model.predict(data)
