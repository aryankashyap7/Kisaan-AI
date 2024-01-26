import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import pickle
import shutil
import plotly.graph_objects as go
from PIL import Image, ImageOps
import numpy as np
import base64
import path
import json
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.models import load_model
import _pickle

# Load Crop Recommender model
try:
    with open("./models/CropRecommender.pkl", "rb") as crop_recommender_pickle:
        crop_recommender_model = pickle.load(crop_recommender_pickle)
except _pickle.UnpicklingError as e:
    print(f"Error loading CropRecommender model: {e}")
    crop_recommender_model = None

# Load Plant Disease Detection model
plant_disease_model = load_model("./models/Plant_Disease.hdf5")

# Function to preprocess image and make predictions
def import_and_predict(image_data, model):
    img = ImageOps.fit(image_data, size=(220, 220))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255
    result = model.predict([np.expand_dims(x, axis=0)])
    return result

# Function to predict crop yield
yield_model = pickle.load(open('models\dtr.pkl', 'rb'))
yield_preprocessor = pickle.load(open('models\preprocessor.pkl', 'rb'))

def predict_yield(year, rain_mm, pesticides, temp, area, crop):
    # Create an array of input features
    features = np.array([[year, rain_mm, pesticides, temp, area, crop]], dtype=object)

    # Transform features using the preprocessor
    transformed_features = yield_preprocessor.transform(features)

    # Make the prediction
    predicted_yield = yield_model.predict(transformed_features).reshape(1, -1)

    return predicted_yield[0]

# Function to convert image to base64 bytes
def img_to_bytes(img_path):
    img_bytes = path.Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Streamlit application
def main():
    # HTML template for styling
    html_template = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"><font size="10"> AgriSense Smart Harvest🌱 </font></h1>
    </div>
    <!-- Background styling -->
    <style>
    .reportview-container .main {
        background: url("https://plasticseurope.org/wp-content/uploads/2021/10/5.6._aaheader.png");
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background: url("https://plasticseurope.org/wp-content/uploads/2021/10/5.6._aaheader.png")
    }
    </style>
    <style>
    body {
    background-image: url("https://plasticseurope.org/wp-content/uploads/2021/10/5.6._aaheader.png");
    background-size: cover;
    }
    </style>
    """

    st.markdown(html_template, unsafe_allow_html=True)

    # UI for selecting functionalities
    selection = st.radio(
        "",
        [
            "Crop Disease Detection",
            "Crop Recommendation",
            "Yield Prediction",
        ],
    )

    # Streamlit styling
    st.write(
        """<style>
            .reportview-container .markdown-text-container {
                font-family: monospace;
            }
            .sidebar .sidebar-content {
                background-image: linear-gradient(#FFFFFF,#FFFFFF);
                color: white;
            }
            .Widget>label {
                color: white;
                font-family: monospace;
            }
            [class^="st-b"]  {
                color: white;
                font-family: monospace;
            }
            .st-bb {
                background-color: transparent;
            }
            .st-at {
                
            }
            footer {
                font-family: monospace;
            }
            .reportview-container .main footer, .reportview-container .main footer a {
                color: #FFFFFF;
            }
            header .decoration {
                background-image: none;
            }

            </style>""",
        unsafe_allow_html=True,
    )

    if selection == "Crop Disease Detection":
        # UI and logic for Crop Disease Detection
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Diseases Detection</b></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(textbg.format(bgcolor, fontcolor), unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b> In recent times, drastic climate changes and lack of immunity in crops has caused substantial increase in growth of crop diseases. This causes large scale demolition of crops, decreases cultivation and eventually leads to financial loss of farmers. Due to rapid growth in variety of diseases , identification and treatment of the disease is a major importance.</b></font></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(text.format(bgcolor, fontcolor), unsafe_allow_html=True)

        st.markdown(
            """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        file = st.sidebar.file_uploader("Please upload a crop image")

        if st.button("Detect"):
            if file is None:
                st.sidebar.text("please upload an image file")
            else:
                image = Image.open(file)
                st.image(image, use_column_width=True)
                predictions = import_and_predict(image, plant_disease_model)
                file_json = open("./models/class_indices.json", "r")
                class_indices = json.load(file_json)
                classes = list(class_indices.keys())
                classresult = np.argmax(predictions, axis=1)
                word = classes[classresult[0]].split("__")
                word[0] = word[0].replace("_", " ")
                word[1] = word[1].replace("_", " ")
                st.success("This crop is {} and it has {} ".format(
                    word[0], word[1]))

    elif selection == "Crop Recommendation":
        # UI and logic for Crop Recommendation
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Recommendation</b></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(textbg.format(bgcolor, fontcolor), unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b> Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.</b></font></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(text.format(bgcolor, fontcolor), unsafe_allow_html=True)

        st.markdown(
            """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#3CB371,#3CB391);
            color: white;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown(
            " Find out the most suitable crop to grow in your farm 👨‍🌾")
        nitrogen = st.sidebar.number_input("Nitrogen", 1, 10000)
        phosphorus = st.sidebar.number_input("Phosporus", 1, 10000)
        potassium = st.sidebar.number_input("Potassium", 1, 10000)
        temperature = st.sidebar.number_input("Temperature", 0.0, 100000.0)
        humidity = st.sidebar.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.sidebar.number_input("Ph", 0.0, 100000.0)
        rainfall = st.sidebar.number_input("Rainfall in mm", 0.0, 100000.0)

        features_list = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        single_prediction = np.array(features_list).reshape(1, -1)

        if st.button("Predict"):
            prediction = crop_recommender_model.predict(single_prediction)
            st.success(
                f"{prediction.item().title()} are recommended by the A.I for your farm."
            )
            
    elif selection == "Yield Prediction":
        # UI and logic for Crop Yield Prediction
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Yield Prediction</b></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(textbg.format(bgcolor, fontcolor), unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b>Forecasting or predicting the crop yield well ahead of its harvest time would assist the strategists and farmers for taking suitable measures for selling and storage. In addition to such human errors, the fluctuations in the prices themselves make creating a stable and robust forecasting solution a necessity.</b></font></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(text.format(bgcolor, fontcolor), unsafe_allow_html=True)

        st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#3CB371,#3CB391);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

        st.sidebar.header('Crop Yield Prediction')

        # Input parameters
        year = st.sidebar.slider('Select Year:', 1990, 2023, 1990)
        rain_mm = st.sidebar.number_input('Average Rainfall (mm):', 0.0, 5000.0, 1485.0)
        pesticides = st.sidebar.number_input('Pesticides (tonnes):', 0.0, 1000.0, 121.0)
        temp = st.sidebar.number_input('Average Temperature:', 0.0, 40.0, 16.37)

        # Dropdown for Area
        area_options = ['India','Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon','Canada', 'Central African Republic', 'Chile', 'Colombia', 'Croatia','Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea','Estonia', 'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala','Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Indonesia', 'Iraq','Ireland', 'Italy', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia','Lebanon', 'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi', 'Malaysia','Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro', 'Morocco','Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua','Niger', 'Norway', 'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal','Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia','South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden','Switzerland', 'Tajikistan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda','Ukraine', 'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe']
        area = st.sidebar.selectbox('Select Area:', area_options)

        # Dropdown for Crop
        crop_options = ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat', 'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams']
        crop = st.sidebar.selectbox('Select Crop:', crop_options)

        # Button to trigger prediction
        if st.sidebar.button('Predict Yield'):
            result = predict_yield(year, rain_mm, pesticides, temp, area, crop)
            st.success(f'The predicted yield for {crop} in {area} for the year {year} is {result}')

            # Hide Streamlit style
            hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

            # Footer with team information
            st.markdown(
                """<style>footer {visibility: hidden;} footer:after {content:'Capstone Project By Team 21 💥';visibility: visible;display: block;position: relative;#background-color: red;padding: 5px; top: 2px;}</style>""",
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
