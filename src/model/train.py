from preprocessing.sms import SMSDataPreprocessingManager
from model.model import *
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="set arguments. ")
    parser.add_argument("--data_path", default="./data/spam.csv", type=str, help="Traing and Validation data")
    parser.add_argument("--model_path", default="./models/model/model.pkl", type=str, help="save model path")
    
    args = parser.parse_args()   
    
    # 데이터 불러오기
    train_df = SMSDataPreprocessingManager.read_entire_data(path=args.data_path)
    
    # 데이터 preprocessing
    X_data_tfidf, X_train_tfidf, X_test_tfidf, y_train, y_test = SMSDataPreprocessingManager.data_preprocess(train_df)
    
    # training model
    lgbm_model = training_model(X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    # save model
    save_model(lgbm_model, args.model_path)
    
    # evaluate model
    y_pred = evaluate_model(lgbm_model, X_test_tfidf)
    

    
    
    