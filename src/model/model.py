# lightGBM import
from lightgbm import LGBMClassifier
# preprocess
from preprocessing.sms import PreprocessingManager, SMSDataPreprocessingManager

import numpy as np
import pandas as pd

from sklearn import metrics

# LGBM 분류기 객체 생성
lgbm_model = LGBMClassifier(n_estimators=400)



def data_preprocessing():
    
    pre_manager = SMSDataPreprocessingManager(
        feature_column_name='message',
        label_column_name='label'   
    )

def training_model():
    
    






def main():
    
    # 조기 중단 기능에 필요한 파라미터 정의
    evals = [(X_test_tfidf, y_test),(X_train_tfidf, y_train)]
    lgbm_model.fit(X_train_tfidf, y_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals, verbose=True)