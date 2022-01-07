import csv
import os
import pandas as pd
from nltk.corpus import stopwords
import re
from util.log_util import logger, spent_time_measure

current_path = os.path.abspath(os.path.dirname(__file__))


@logger
def read_csv_file(file_name):
    path = os.path.join(current_path, '..', 'Dataset')
    os.chdir(path)
    data = pd.read_csv(file_name, encoding='latin-1')
    return data


@logger
def get_test_data():
    file_name = 'Corona_NLP_test.csv'
    # file_name = 'Test_Data.csv'
    test_data = read_csv_file(file_name)
    return test_data


@logger
def get_train_data():
    file_name = 'Corona_NLP_train.csv'
    # file_name = 'Train_Data.csv'
    train_data = read_csv_file(file_name)
    return train_data


def cont_rep_char(raw_data_col):
    tchr = raw_data_col.group(0)

    if len(tchr) > 1:
        return tchr[0:2]


def data_preprocess(raw_data_col):
    # remove user name
    re.sub('@[^\s]+', '', raw_data_col)

    # remove tag
    # raw_data_col.replace("<br/>", " ")

    # remove url
    url = re.compile(r'https?://\S+|www\.\S+')
    raw_data_col = url.sub(r'', raw_data_col)

    # remove emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    raw_data_col = emoji_pattern.sub(r'', raw_data_col)

    # Decontraction text
    raw_data_col = re.sub(r"won\'t", " will not", raw_data_col)
    raw_data_col = re.sub(r"won\'t've", " will not have", raw_data_col)
    raw_data_col = re.sub(r"can\'t", " can not", raw_data_col)
    raw_data_col = re.sub(r"don\'t", " do not", raw_data_col)
    raw_data_col = re.sub(r"can\'t've", " can not have", raw_data_col)
    raw_data_col = re.sub(r"ma\'am", " madam", raw_data_col)
    raw_data_col = re.sub(r"let\'s", " let us", raw_data_col)
    raw_data_col = re.sub(r"ain\'t", " am not", raw_data_col)
    raw_data_col = re.sub(r"shan\'t", " shall not", raw_data_col)
    raw_data_col = re.sub(r"sha\n't", " shall not", raw_data_col)
    raw_data_col = re.sub(r"o\'clock", " of the clock", raw_data_col)
    raw_data_col = re.sub(r"y\'all", " you all", raw_data_col)
    raw_data_col = re.sub(r"n\'t", " not", raw_data_col)
    raw_data_col = re.sub(r"n\'t've", " not have", raw_data_col)
    raw_data_col = re.sub(r"\'re", " are", raw_data_col)
    raw_data_col = re.sub(r"\'s", " is", raw_data_col)
    raw_data_col = re.sub(r"\'d", " would", raw_data_col)
    raw_data_col = re.sub(r"\'d've", " would have", raw_data_col)
    raw_data_col = re.sub(r"\'ll", " will", raw_data_col)
    raw_data_col = re.sub(r"\'ll've", " will have", raw_data_col)
    raw_data_col = re.sub(r"\'t", " not", raw_data_col)
    raw_data_col = re.sub(r"\'ve", " have", raw_data_col)
    raw_data_col = re.sub(r"\'m", " am", raw_data_col)
    raw_data_col = re.sub(r"\'re", " are", raw_data_col)

    # Separate alphanumeric
    raw_data_col = re.findall(r"[^\W\d_]+|\d+", raw_data_col)
    raw_data_col = " ".join(raw_data_col)

    # rep = cont_rep_char(raw_data_col)
    raw_data_col = re.sub(r'(\w)\1+', cont_rep_char, raw_data_col)

    # remove digit
    raw_data_col = re.sub(r'[^a-zA-Z]', ' ', raw_data_col)

    raw_data_col = raw_data_col.lower()

    # remove stopwords
    # costly operation
    raw_data_col = ' '.join([word for word in raw_data_col.split() if word not in (stopwords.words('english'))])

    return raw_data_col


def convert_sentiment(sentiment):
    if sentiment == 'Extremely Negative':
        return -2
    elif sentiment == 'Negative':
        return -1
    elif sentiment == 'Neutral':
        return 0
    elif sentiment == 'Positive':
        return 1
    elif sentiment == 'Extremely Positive':
        return 2


@spent_time_measure
def get_clean_data(raw_data_frame):
    data_frame = raw_data_frame
    data_frame['OriginalTweet'] = data_frame['OriginalTweet'].apply(lambda raw_data_col: data_preprocess(raw_data_col))
    data_frame['Sentiment'] = data_frame['Sentiment'].apply(
        lambda sentiment_data_col: convert_sentiment(sentiment_data_col))

    # print(data_frame['OriginalTweet'])
    # print(data_frame['Sentiment'])
    return data_frame

