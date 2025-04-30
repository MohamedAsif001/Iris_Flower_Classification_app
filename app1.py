
import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load('iris_model.pkl')

# Load Iris dataset for feature names and target names
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Function to make predictions
def predict_iris(features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0]
    return prediction, probability

def main():
    st.title("Iris Flower Classification App")
    st.write("Enter the sepal and petal measurements to predict the Iris species.")

    # Input fields for features
    sepal_length = st.slider(feature_names[0], min_value=4.0, max_value=8.0, value=5.0, step=0.1)
    sepal_width = st.slider(feature_names[1], min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    petal_length = st.slider(feature_names[2], min_value=1.0, max_value=7.0, value=4.0, step=0.1)
    petal_width = st.slider(feature_names[3], min_value=0.1, max_value=2.5, value=1.3, step=0.1)

    features = [sepal_length, sepal_width, petal_length, petal_width]

    if st.button("Predict"):
        prediction, probability = predict_iris(features)

        st.subheader("Prediction Result:")
        predicted_class_name = target_names[prediction]
        st.write(f"The predicted Iris species is: **{predicted_class_name}**")

        st.subheader("Prediction Probabilities:")
        for i, prob in enumerate(probability):
            st.write(f"- **{target_names[i]}**: {prob:.4f}")

if __name__ == '__main__':
    main()