from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from pushbullet import Pushbullet

app = Flask(__name__)

# Load the trained model and vectorizer
vectorizer = TfidfVectorizer()
classifier = SGDClassifier(max_iter=1000)
df = pd.read_csv("dataset.csv")
vectorizer.fit(df['full_text'])
classifier.fit(vectorizer.transform(df['full_text']), df['label'])

# Initialize Pushbullet
pb = Pushbullet("o.yZiF7RuZvNzuiMSEqJM4H3i4JVAi4li9")#replace with your own id

def classify_text(text):
    prediction = classifier.predict(vectorizer.transform([text]))[0]
    if prediction == 0:
        # Send notification
        title=""
        body = text
        push = pb.push_note(title, body)
        return "Message Sent"
    else:
        # Send notification
        title = "Cyber Bullying Alert"
        body = "Cyber bullying is a crime. Please refrain from such actions in the future."
        push = pb.push_note(title, body)
        return "Warning: Message contains bullying content"

# Define Flask routes
@app.route('/')
def index():
    return render_template('gpt1.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    result = classify_text(text)
    if result == "Message Sent":
        return jsonify({'result': result})
    else:
        return jsonify({'result': result, 'text': text})

@app.route('/end')
def end():
    text = request.args.get('text', '')
    result = "non_bullying"
    return render_template('gpt2.html', text=text, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
