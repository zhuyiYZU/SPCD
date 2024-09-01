import pandas as pd
import re

train_file_path = './datasets/TextClassification/ohsumed/train.csv'
test_file_path = './datasets/TextClassification/ohsumed/test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

print(train_data.shape[0] + test_data.shape[0])

train_text_colum = train_data.iloc[:, 1].astype(str)
test_text_colum = test_data.iloc[:, 1].astype(str)

def cal_word_av_len(text):
    words = re.findall(r'\w+', text)
    return len(words)
    # total_word_len = sum(len(word) for word in words)
    # if len(words) > 0:
    #     return total_word_len /len(words)
    # else:
    #     return 0

train_word_av_len = train_text_colum.apply(cal_word_av_len)
test_word_av_len = test_text_colum.apply(cal_word_av_len)

print(train_word_av_len)

average_length = (train_word_av_len.mean() + test_word_av_len.mean()) / 2

print(average_length)