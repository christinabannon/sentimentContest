import pickle
from build import util
from sklearn.feature_extraction.text import TfidfVectorizer
import csv



file_lines = util.readFile('run/test1_public.txt');
tweet_ids, features = util.create_data(file_lines);
features = util.preprocess_features(features);

tfidf = TfidfVectorizer(use_idf=True, max_features=3000)

loaded_model = pickle.load(open('build/finalized_model.sav', 'rb'))

XX_test = tfidf.fit_transform(features);
yy_test = loaded_model.predict(XX_test)

for i in range(len(yy_test)):
    print(str(yy_test[i]) + ", " + features[i])

with open('test1.csv', mode='w') as file:
    writer = csv.writer(file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(1, len(yy_test)):
        sentiment = '+'
        if yy_test[i] == 0:
            sentiment = '-';
        writer.writerow([tweet_ids[i], sentiment])
