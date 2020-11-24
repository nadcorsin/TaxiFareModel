# imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder())
        time_cols = ['pickup_datetime']
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        preprocessing = ColumnTransformer([('time', pipe_time, time_cols),
                                  ('distance', pipe_distance, dist_cols)]
                                  )
    
        pipe_cols = Pipeline(steps=[('preprocessing', preprocessing),
                            ('regressor', LinearRegression())])
        return pipe_cols

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        return self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
