from preprocessing.sms import SMSDataPreprocessingManager
from model.model import *
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="set arguments. ")
    parser.add_argument("--file_path", default="./data/test/sns_test.csv", type=str, help="predict dataframe path")
    parser.add_argument("--model_path", default="./models/model/model.pkl", type=str, help="save model path")
    parser.add_argument("--save_path", default="./data/predict/result.csv", type=str, help="result save path")
    
    args = parser.parse_args()
    
    test_predict(args.file_path, args.model_path, args.save_path)
    
    print("End predict")