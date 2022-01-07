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
- Used SVM for prediction [svm.predict()] 
