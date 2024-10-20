import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# Load the trained model and vectorizer
vectorizer = TfidfVectorizer()
classifier = SGDClassifier(max_iter=1000)
df = pd.read_csv("dataset.csv")
vectorizer.fit(df['full_text'])
classifier.fit(vectorizer.transform(df['full_text']), df['label'])

# Save the trained model using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Load the scikit-learn model and convert it to h5 format
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Save the scikit-learn model in h5 format
with open('model.h5', 'wb') as f:
    pickle.dump(model, f)
