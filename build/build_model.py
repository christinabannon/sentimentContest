from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
from build import util

# def create_train_data(file_lines):
#     tweet_ids = [];
#     labels = [];
#     features = [];
#
#     for line in file_lines:
#         split_line = line.split();
#         if len(split_line) > 2:
#             tweet_ids.append(split_line[0]);
#             labels.append(split_line[1]);
#             feature = "";
#             for i in range(2, len(split_line)):
#                 feature = feature + " " + (split_line[i]);
#             features.append(feature);
#
#     return [tweet_ids, labels, features];

file_lines = util.readFile('Training.txt');
training_data, testing_data = train_test_split(file_lines, test_size=0.20, random_state=0)

train_data_lines = util.create_train_data(training_data);
train_tweet_ids = train_data_lines[0];
train_labels = train_data_lines[1];
train_features = train_data_lines[2];
train_features = util.preprocess_features(train_features);

test_data_lines = util.create_train_data(testing_data);
test_tweet_ids = test_data_lines[0];
test_labels = test_data_lines[1];
test_features = test_data_lines[2];
test_features = util.preprocess_features(test_features);

tfidf = TfidfVectorizer(use_idf=True, max_features=3000)

def tfidf_features(txt, flag):
    if flag == "train":
      x = tfidf.fit_transform(txt)
    else:
      x = tfidf.transform(txt)
    x = x.astype('float16')
    return x

XX = tfidf_features(train_features, flag="train")
XX_test = tfidf_features(test_features, flag="test")

lb = LabelEncoder()
yy = lb.fit_transform(train_labels)

lr_model = LogisticRegression(max_iter=500);
lr_model.fit(XX, yy)
yy_test = lr_model.predict(XX_test)
yy_pred = lb.inverse_transform(yy_test)

print("Accuracy:",metrics.accuracy_score(test_labels, yy_pred))


pickle.dump(lr_model, open('finalized_model.sav', 'wb'))
pickle.dump(tfidf, open("vectorizer.pickle", "wb"))
