import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('dummy_dataset - Sheet1 (1).csv')

# Inspect the dataset
print(data.head())

# Encode the 'Category' column
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Select features (excluding ID and Name for simplicity)
X = data[['Description', 'Price']]
y = data['Category']

# Vectorize the 'Description' column
tfidf = TfidfVectorizer()
X_description = tfidf.fit_transform(X['Description'])

# Convert the sparse matrix to a dense matrix and add the 'Price' column
X_combined = np.hstack((X_description.toarray(), X[['Price']].values))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Save the trained model and the TF-IDF vectorizer
joblib.dump(model, 'classification_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Create a Flask app
app = Flask(__name__)

@app.route('/classification', methods=['POST'])
def classify():
    data = request.json
    description = data['Description']
    price = data['Price']
    
    # Vectorize the description
    description_vector = tfidf.transform([description]).toarray()
    
    # Combine description vector with price
    features = np.hstack((description_vector, [[price]]))
    
    # Predict the category
    prediction = model.predict(features)
    
    # Decode the predicted label
    predicted_category = label_encoder.inverse_transform(prediction)
    
    return jsonify({'prediction': predicted_category[0]})

if __name__ == '__main__':
    app.run(debug=True)
