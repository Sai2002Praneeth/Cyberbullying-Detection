from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
app = Flask(__name__)

# Load the trained model and vectorizer
vectorizer = TfidfVectorizer()
classifier = LogisticRegression(max_iter=1000)
df = pd.read_csv("ren.csv")
vectorizer.fit(df['full_text'])
classifier.fit(vectorizer.transform(df['full_text']), df['label'])

def classify_text(text):
    prediction = classifier.predict(vectorizer.transform([text]))[0]
    if prediction == 0:
        return "Message Sent"
    else:
        return "Warning: Message contains bullying content"

# Define Flask routes
@app.route('/')
def index():
    return render_template('gpt1.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    result = classify_text(text)
    return jsonify({'result': result, 'text': text})

@app.route('/end')
def end():
    text = request.args.get('text', '')
    result = "non_bullying"
    return render_template('gpt2.html', text=text, result=result)

if __name__ == '__main__':
    app.run(debug=True)
