import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
import numpy as np

# Load the trained model and vectorizer
vectorizer = TfidfVectorizer()
classifier = SGDClassifier(max_iter=1000)
df = pd.read_csv("dataset.csv")
vectorizer.fit(df['full_text'])
classifier.fit(vectorizer.transform(df['full_text']), df['label'])

# Save the trained model using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Convert the scikit-learn model to TensorFlow format
model = tf.convert_to_tensor(classifier.coef_, dtype=tf.float32)
model = tf.expand_dims(model, axis=0)

# Create a wrapper class for the model
class MyModel(tf.Module):
    def __init__(self, model):
        self.model = tf.Variable(model)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def predict(self, x):
        x = tf.cast(x, dtype=tf.float32)
        return tf.matmul(x, tf.transpose(self.model))

# Instantiate the wrapper class
my_model = MyModel(model)

# Save the TensorFlow model with a signature
signatures = {"serving_default": my_model.predict}
tf.saved_model.save(my_model, 'saved_model', signatures)

# Convert the TensorFlow SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
