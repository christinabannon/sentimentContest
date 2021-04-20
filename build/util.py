import re

def readFile(fileName):
    fileLines = []
    with open(fileName, encoding="ISO-8859-1") as file:
        fileLines = file.readlines();
    return fileLines;

# from https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
def preprocess_features(features):
    processed_features = []

    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
        processed_features.append(processed_feature)
    return processed_features;


def create_train_data(file_lines):
    tweet_ids = [];
    labels = [];
    features = [];

    for line in file_lines:
        split_line = line.split();
        if len(split_line) > 2:
            tweet_ids.append(split_line[0]);
            labels.append(split_line[1]);
            feature = "";
            for i in range(2, len(split_line)):
                feature = feature + " " + (split_line[i]);
            features.append(feature);

    return [tweet_ids, labels, features];

def create_data(file_lines):
    tweet_ids = [];
    features = [];

    for line in file_lines:
        split_line = line.split();
        if len(split_line) > 1:
            tweet_ids.append(split_line[0]);
            feature = "";
            for i in range(2, len(split_line)):
                feature = feature + " " + (split_line[i]);
            features.append(feature);

    return tweet_ids, features;