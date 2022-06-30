import string

import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 

# common
from src.preprocessing.common import PreprocessingManager


class SMSDataPreprocessingManager(PreprocessingManager):

    def __init__(
        self,
        feature_column_name: str = 'message',
        label_column_name: str = 'label',
        label_1: str = 'ham',
        label_2: str = 'spam',
        regex_standard: str = '\(?(http|https|ftp|ftps)?\:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?\)?|\d{2,3}-?\d{3,4}-?\d{4}|([\w\.-]+)@([\w\.-]+)(\.[\w\.]+)'
    ) -> None:
        super().__init__(feature_column_name, label_column_name, regex_standard,label_1, label_2)
        
        
    def read_sample_data(
        self,
        path: str,
        ratio: int = 0.1,
    ):
        df_sms = pd.read_csv(path, encoding='latin-1')
        df_sms.dropna(how="any", inplace=True, axis=1)
        df_sms.columns = [self.label_column_name, self.feature_column_name]
        return df_sms[:int(len(df_sms)*ratio)]

    def read_entire_data(
        self,
        path: str
    ):
        df_sms = pd.read_csv(path, encoding='latin-1')
        df_sms.dropna(how="any", inplace=True, axis=1)
        df_sms.columns = [self.label_column_name, self.feature_column_name]
        return df_sms


    def remove_stopwords(
        self,
        df: pd.DataFrame,
    ):

        def fn(msg):
            STOPWORDS = stopwords.words('english') + \
                ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
                
            # 1. 철자 단위 검사 후 문장부호(punctuation) 제거
            nopunc = [char for char in msg if char not in string.punctuation]
            nopunc = ''.join(nopunc)
            
            # 2. 불용어(stop word) 제거
            return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
        
        df[self.feature_column_name] = df[self.feature_column_name].apply(lambda x: fn(x))
        
        return df

    def sentence_to_lowercase(
        self,
        df: pd.DataFrame,
    ):
      
        df[self.feature_column_name] = df[self.feature_column_name].apply(lambda x: x.lower())
        
        return df

    def text_cleansing(
        self,
        df: pd.DataFrame,
    ):
        
        df[self.feature_column_name] = df[self.feature_column_name].str.replace(self.regex_standard,'')
        
        return df


    def label_convert(
        self,
        df: pd.DataFrame
    ):
        
        df.loc[(df[self.label_column_name]==self.label_1), self.label_column_name] = 0
        df.loc[(df[self.label_column_name]==self.label_2), self.label_column_name] = 1
        df[self.label_column_name] = df[self.label_column_name].astype('float32')
        
        return df
        
    def split_data(
        self,
        df: pd.DataFrame,
        ratio: int = 0.2
    ):
        
        X_train, X_test, y_train, y_test = train_test_split(df[self.feature_column_name] ,df[self.label_column_name], test_size=ratio, random_state=111, shuffle=True)
        
        return X_train, X_test, y_train, y_test    

    def get_tfidf(
        self,
        df: pd.DataFrame,
        X_train_dtm: np.ndarray,
        X_test_dtm: np.ndarray        
    ):
        tfidf_vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, use_idf=True, smooth_idf=True)
        X_data_tfidf = tfidf_vec.fit_transform(df[self.feature_column_name])
        X_train_tfidf = tfidf_vec.transform(X_train_dtm)
        X_test_tfidf = tfidf_vec.transform(X_test_dtm)
        
        return X_data_tfidf, X_train_tfidf, X_test_tfidf
    
    
    def data_preprocess(
        self,
        df: pd.DataFrame
    ): 
        df = SMSDataPreprocessingManager.label_convert(df)
        df = SMSDataPreprocessingManager.sentence_to_lowercase(df)
        df = SMSDataPreprocessingManager.text_cleansing(df)
        df = SMSDataPreprocessingManager.remove_stopwords(df)
        
        X_train, X_test, y_train, y_test = SMSDataPreprocessingManager.split_data(df)
        X_data_tfidf, X_train_tfidf, X_test_tfidf = SMSDataPreprocessingManager.get_tfidf(df, X_train, X_test)
        
        return X_data_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test
        
    def test_data_preprocess(
        self,
        
    )    