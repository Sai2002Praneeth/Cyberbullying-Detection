# Cyberbullying-Detection

This project focuses on detecting and blocking bullying content in social media posts and comments using machine learning and natural language processing (NLP) techniques. Built as a Flask web application, it provides real-time monitoring to prevent the spread of harmful content.

# Table of Contents

1) Overview
2) Features
3) Installation
4) Usage
5) Model and Approach
6) License

# Overview

Cyberbullying on social media can have a significant impact on mental health. This project aims to create a safe online environment by identifying bullying content in real-time and blocking it before it reaches users. The application uses a trained machine learning model to classify content and alert moderators when bullying is detected.

# Features

1) Real-time Detection: Automatically detects bullying content in posts and comments.
2) NLP-based Analysis: Uses natural language processing to analyze the tone and intent of content.
3) Customizable: Supports adding new content types or categories.
4) Admin Interface: Allows moderators to view flagged content and take action if necessary.


# Installation

To run the project locally, follow these steps:

1) Clone the repository:

   git clone https://github.com/yourusername/cyberbullying-detection.git
   cd cyberbullying-detection

2) Create and activate a virtual environment:

   python3 -m venv env
   source env/bin/activate  ( On Windows use `env\Scripts\activate` )

3) Install the dependencies:

   pip install -r requirements.txt

4) Run the Flask application:

   python app.py

5) Open the application in your browser at http://127.0.0.1:5000.


# Usage

1) Open the application.
2) Enter the social media post or comment text in the provided input field.
3) Click on Analyze. The model will classify whether the content is bullying-related.
4) If bullying content is detected, the post will be flagged, and moderators can review it.


# Model and Approach

The model was developed using machine learning and NLP techniques to analyze social media content. Key steps in the development process included:

1) Data Collection: We compiled a dataset of social media posts and comments with labeled bullying content.
2) Preprocessing: Text data was cleaned and preprocessed for model training.
3) Feature Extraction: We extracted relevant features using techniques such as TF-IDF.
4) Model Training: Multiple classifiers were tested, with the final model chosen based on accuracy and performance.
