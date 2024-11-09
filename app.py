# Import the necessary libraries
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample data: Text samples and labels (1 = positive, 0 = negative)
texts = ["I love coding", "I hate bugs", "Python is awesome", "Debugging is hard"]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Create a vectorizer to convert text to numbers
vectorizer = CountVectorizer()

# Convert the texts into numbers
X = vectorizer.fit_transform(texts)

# Create and train the model
classifier = LogisticRegression()
classifier.fit(X, np.array(labels))

# Function to classify new text input
def classify_text(input_text):
    # Convert the input text into the same format the model expects
    X_input = vectorizer.transform([input_text])

    # Make a prediction: 1 = Positive, 0 = Negative
    prediction = classifier.predict(X_input)

    # Return the result
    return "Positive" if prediction[0] == 1 else "Negative"

# Streamlit user interface
st.title("Text Sentiment Classifier")

# Create a text box for the user to input text
input_text = st.text_area("Enter your text for classification:")

# When the user clicks the 'Classify' button
if st.button("Classify"):
    if input_text:
        # Classify the input text and display the result
        result = classify_text(input_text)
        st.write(f"The sentiment of the text is: {result}")
    else:
        st.write("Please enter some text.")
