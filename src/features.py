import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def add_engineered_features(df):
    data=df.copy()
    data['hour']=np.floor(data['Time']/3600)%24
    data['amount_log']=np.log1p(data['Amount'])
    data=data.drop(['Time', 'Amount'], axis=1)
    return data
def scale_data(X_train, X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler