import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Function to preprocess input data
def preprocess_input(input_data, feature_names, scaler):
    # Convert input data to DataFrame with correct feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    # Scale input data
    input_scaled = scaler.transform(input_df)
    return input_scaled

# Function to make predictions
def predict(input_data, model, scaler, feature_names):
    input_scaled = preprocess_input(input_data, feature_names, scaler)
    prediction = model.predict(input_scaled)
    return prediction

# Page for feature descriptions and images
def feature_page():
    st.title('Breast Cancer Predictor - Feature Descriptions')

    st.write("""
    Welcome to the Breast Cancer Predictor app! 
    This app predicts whether a breast mass is benign or malignant based on cell nuclei measurements.

    Below are the descriptions of the features used for prediction:
    """)

    # Feature images and descriptions
    feature_images = {
        'mean_radius': '/Users/mann/Desktop/SEM-IV/Machine Learning Labs/ml project files/mean-radius-of-malignant-patients.png',
        'mean_texture': '/Users/mann/Desktop/SEM-IV/Machine Learning Labs/ml project files/images.jpeg',
        # Add more feature images here
    }
    feature_descriptions = {
        'mean_radius': 'Mean radius of the tumor.',
        'mean_texture': 'Mean texture of the tumor.',
        # Add more feature descriptions here
    } 

    # Display images and descriptions for each feature
    # Display images and descriptions for each feature
    for feature, image_path in feature_images.items():
      st.subheader(feature)
      st.image(Image.open(image_path).resize((200, 200)), caption=feature_descriptions[feature], use_column_width=True)


# Page for prediction
def prediction_page():
    st.title('Breast Cancer Predictor - Prediction')

    st.write("""
    To make a prediction, adjust the sliders on the sidebar to input the cell nuclei measurements 
    and click the 'Predict' button. The model will then predict whether the breast mass is benign or malignant.
    """)

    # Load model and scaler
    model_path = "model/model.pkl"
    scaler_path = "model/scaler.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        # Get feature names used during model training
        feature_names = scaler.get_feature_names_out()
        
    except FileNotFoundError:
        st.error("Model files not found. Please make sure the model and scaler files exist.")
        return

    # Sidebar with input form
    st.sidebar.header('Input Features')
    input_features = {}
    for feature in feature_names:
        input_features[feature] = st.sidebar.slider(feature, 0.0, 300.0, 150.0)

    if st.sidebar.button('Predict'):
        prediction = predict(list(input_features.values()), model, scaler, feature_names)
        if prediction[0] == 0:
            st.success('Prediction: Benign')
        else:
            st.error('Prediction: Malignant')

# Main function to control page navigation
def main():
    page = st.sidebar.selectbox("Choose a page", ["Feature Descriptions", "Prediction"])
    if page == "Feature Descriptions":
        feature_page()
    elif page == "Prediction":
        prediction_page()

if __name__ == '__main__':
    main()
