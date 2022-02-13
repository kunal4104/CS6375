import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor

class Data:
  def __init__(self, loc):
    self.data = pd.read_csv(loc)
    header = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
    self.data.columns = header

  def getData(self):
    return self.data

  def processData(self, Data):

    Data.replace('?',np.nan, inplace=True)
    Data.dropna(inplace=True)

    Data['normalized-losses'] = Data['normalized-losses'].astype('int')
    Data['bore'] = Data['bore'].astype('float')
    Data['stroke'] = Data['stroke'].astype('float')
    Data['horsepower'] = Data['horsepower'].astype('float')
    Data['peak-rpm'] = Data['peak-rpm'].astype('int')
    Data['price'] = Data['price'].astype('int')
    Data = Data.select_dtypes(exclude=['object'])

    return Data

  def getFeaturesAndTarget(self, Data):  
    X = Data[["highway-mpg", "curb-weight", "horsepower", "engine-size"]]
    Y = Data[["price"]]
    Y = Y.values.ravel()
    s = StandardScaler()
    X = pd.DataFrame(s.fit(X).fit_transform(X))
    return (X, Y)

  def getDataSplit(self, X, Y):  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
    return (X_train, X_test, Y_train, Y_test)



class Model:
  model = SGDRegressor()
  def __init__(self, X, Y):
    self.model.fit(X,Y)

  def pred(self, X):  
    return self.model.predict(X);

  def rmseAndR2(self, expected, predicted):
    rmse = (np.sqrt(mean_squared_error(expected, predicted)))
    r2 = r2_score(expected, predicted)
    return (rmse,r2)




if __name__ == "__main__":
    data = Data('imports-85.data'); 
    df = data.getData();
    df = data.processData(df);
    X, Y = data.getFeaturesAndTarget(df)
    X_train, X_test, Y_train, Y_test = data.getDataSplit(X, Y);
    sgdLinear = Model(X_train, Y_train)
    Y_pred_test = sgdLinear.pred(X_test);
    rmse, r2 = sgdLinear.rmseAndR2(Y_test, Y_pred_test);
    print(rmse, r2)

