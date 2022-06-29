from preprocessing.sms import PreprocessingManager
from model.model import *
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="set arguments. ")
    parser.add_argument("--data_path", default="./data/spam.csv", type=str, help="Traing and Validation data")
    parser.add_argument("--model_path", default="./models/model/model.pkl", type=str, help="save model path")
    
    args = parser.parse_args()   
    
    # 데이터 불러오기
     
    