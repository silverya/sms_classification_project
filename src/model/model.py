# lightGBM import
from lightgbm import LGBMClassifier
# preprocess
from preprocessing.sms import SMSDataPreprocessingManager

import pickle
import numpy as np
import pandas as pd

import mlflow

from sklearn import metrics

# LGBM 분류기 객체 생성
lgbm_model = LGBMClassifier(n_estimators=400)

def training_model(X_train_tfidf, X_test_tfidf, y_train, y_test):
    
    # 조기 중단 기능에 필요한 파라미터 정의
    evals = [(X_test_tfidf, y_test),(X_train_tfidf, y_train)]
    lgbm_model.fit(X_train_tfidf, y_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals, verbose=True)
    
    return lgbm_model
    

def evaluate_model(lgbm_model, X_test_tfidf):
    
    # model test
    y_pred = lgbm_model.predict(X_test_tfidf)
    
    return y_pred
    
      
def save_model(lgbm_model, filename):    
    with open(filename, 'wb') as w:
        pickle.dump(lgbm_model, w)
        
        
def predict(x, vect, nb):
    x_dtm = vect.transform(x)
    y_pred = nb.predict(x_dtm)
    return y_pred        
        
        
def test_predict(file_path, model_path, save_path):
    test_df = pd.read_csv(file_path)
    
    test_x = test_df['번역'].values
    
    with open(model_path, 'rb') as f:
        vect, nb = pickle.load(f)
    
    y_pred = predict(test_x, vect, nb)
    test_df['분류'] = y_pred
    
    test_df.to_csv(save_path, index=False)
