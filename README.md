# sentiment_analysis

- Used twitter data (Covid tweets). Dataset consists of two sets: a training set (Corona NLP train.csv) and a testing set(Corona NLP test.csv). Tweets in both training and testing sets are labeled as: ”Extremely Positive”, ”Extremely Negative”, ”Positive”, ”Negative”, or ”Neutral”.

- Data preprocessing steps: 
1. remove user name
2. remove url
3. remove emoji
4. Decontraction text
5. Separate alphanumeric
6. remove digit
7. remove stopwords

- Vectorize train and test data using sklearn.feature_extraction.text. [TfidfVectorizer()]

- Used following ML models for prediction:
1. SVM
2. Multinomial Naive Bayes 
3. Decision Tree Classifier
4. Random Forest Classifier

- Time and accurecy for ML models (No of tweets - Train data: 44955, Test data: 3798)
1. SVM accurecy - 58.0% time - 669.2258639335632 sec
2. NB accurecy - 35.0% time -	0.0150299072265625 sec
3. DT accurecy - 42.0% time - 17.01743483543396 sec
4. RF accurecy - 49.0% time - 81.24456477165222 sec
