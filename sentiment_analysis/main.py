import util
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


'''
def data_vectorize_TF_IDF(data_frame):
    data_col = data_frame['OriginalTweet']
    # y = data_frame['']
    # Apply TFIDF on cleaned data
    TF_IDF_Vecotor = TfidfVectorizer()
    data_col_vector = TF_IDF_Vecotor.fit_transform(data_col)
    return data_col_vector
'''


@util.spent_time_measure
def implement_SVM_Using_TF_IDF(data_col_vector, label_col, test_data_col_vector, test_label_col):
    svm = SVC()
    svm.fit(data_col_vector, label_col)
    svm_prediction = svm.predict(test_data_col_vector)
    # print(type(svm_prediction))
    result_acc = round(accuracy_score(svm_prediction, test_label_col), 2)
    print("The accuracy is: ", round(result_acc * 100, 2), '% \t [SVM-TfIdf]')
    return svm_prediction


@util.spent_time_measure
def implement_Naive_Bayes_Using_TF_IDF(data_col_vector, label_col, test_data_col_vector, test_label_col):
    nb = MultinomialNB()
    nb.fit(data_col_vector, label_col)
    nb_prediction = nb.predict(test_data_col_vector)
    # print(type(nb_prediction))
    result_acc = round(accuracy_score(nb_prediction, test_label_col), 2)
    print("The accuracy is: ", round(result_acc * 100, 2), '% \t [NB-TfIdf]')
    return nb_prediction


@util.spent_time_measure
def implement_Decision_Tree_Using_TF_IDF(data_col_vector, label_col, test_data_col_vector, test_label_col):
    dt = DecisionTreeClassifier()
    dt.fit(data_col_vector, label_col)
    dt_prediction = dt.predict(test_data_col_vector)
    # print(type(dt_prediction))
    result_acc = round(accuracy_score(dt_prediction, test_label_col), 2)
    print("The accuracy is: ", round(result_acc * 100, 2), '% \t [DT-TfIdf]')
    return dt_prediction


@util.spent_time_measure
def implement_Random_Forest_Using_TF_IDF(data_col_vector, label_col, test_data_col_vector, test_label_col):
    rf = RandomForestClassifier()
    rf.fit(data_col_vector, label_col)
    rf_prediction = rf.predict(test_data_col_vector)
    # print(type(rf_prediction))
    result_acc = round(accuracy_score(rf_prediction, test_label_col), 2)
    print("The accuracy is: ", round(result_acc * 100, 2), '% \t [RF-TfIdf]')
    return rf_prediction


if __name__ == '__main__':
    TF_IDF_Vecotor = TfidfVectorizer()

    raw_data = util.get_train_data()
    data_frame = util.get_clean_data(raw_data)
    # data_col_vector = data_vectorize_TF_IDF(data_frame)
    data_col_vector = TF_IDF_Vecotor.fit_transform(data_frame['OriginalTweet'])
    label_col = data_frame['Sentiment']

    test_data = util.get_test_data()
    test_data_frame = util.get_clean_data(test_data)
    # test_data_col_vector = data_vectorize_TF_IDF(test_data_frame)
    test_data_col_vector = TF_IDF_Vecotor.transform(test_data_frame['OriginalTweet'])
    test_label_col = test_data_frame['Sentiment']

    implement_SVM_Using_TF_IDF(data_col_vector, label_col, test_data_col_vector, test_label_col)
    implement_Naive_Bayes_Using_TF_IDF(data_col_vector, label_col, test_data_col_vector, test_label_col)
    implement_Decision_Tree_Using_TF_IDF(data_col_vector, label_col, test_data_col_vector, test_label_col)
    implement_Random_Forest_Using_TF_IDF(data_col_vector, label_col, test_data_col_vector, test_label_col)

