from flask import Flask, render_template, request, jsonify
import pandas as pd
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
pb = Pushbullet("o.yZiF7RuZvNzuiMSEqJM4H3i4JVAi4li9")  # Replace with your Pushbullet API key

def classify_text(text):
    prediction = classifier.predict(vectorizer.transform([text]))[0]
    if prediction == 0:
        # Send notification to friends
        friends_email_addresses = ["jeevanreddy263@gmail.com", "saip63470@gmail.com"]
        title = "Akhil"
        body = text
        friend_email = friends_email_addresses[0]
        pb.push_note(title, body, email=friend_email)
        return "Notification Sent"
    else:
        friends_email_addresses = ["rachamallamahesh852@gmail.com", "jeevanreddy263@gmail.com"]
        # Send notification to yourself
        title = "Cyber Bullying Alert"
        body = "The message cannot be sent as it contains sensitive content."
        push = pb.push_note(title, body)
        body2 = "This user is trying to send bullying content. Do you want to block him?"
        friend_mail = friends_email_addresses[1]
        push = pb.push_note(title, body2, email=friend_mail)
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
