import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import pickle
import shutil
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
from PIL import Image, ImageOps
import numpy as np
import base64
import path
import json
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.models import load_model


CropRecommenderPkl = open("./models/CropRecommender.pkl", "rb")
model4 = pickle.load(CropRecommenderPkl)

model5 = load_model("./models/Plant_Disease.hdf5")

AgriPesticidePkl = open("./models/Agri_Pesticide.pkl", "rb")
model6 = pickle.load(AgriPesticidePkl)


def predict_crop_damage(
    Estimated_Insects_Count,
    Crop_Type,
    Soil_Type,
    Pesticide_Use_Category,
    Number_Doses_Week,
    Number_Weeks_Used,
    Number_Weeks_Quit,
    Season,
):

    prediction = model6.predict(
        [
            [
                Estimated_Insects_Count,
                Crop_Type,
                Soil_Type,
                Pesticide_Use_Category,
                Number_Doses_Week,
                Number_Weeks_Used,
                Number_Weeks_Quit,
                Season,
            ]
        ]
    )
    print(prediction)
    return prediction


def import_and_predict(image_data, model):
    img = ImageOps.fit(image_data, size=(220, 220))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255
    result = model.predict([np.expand_dims(x, axis=0)])

    return result


def crop_price_prediction(input_list):
    ohe = {}
    pickle_file = open("./models/ohe.pkl", "rb")
    ohe = pickle.load(pickle_file)
    model = load_model("./models/crop_price_prediction.h5")
    input_array = np.array(input_list).reshape(-1, 5)
    encoded_input = ohe.transform(input_array)
    prediction = model.predict(encoded_input)
    prediction = float(prediction)

    return prediction


def yield_prediction(input_list_yield):
    ohe = {}
    pickle_file = open("./models/OneHotEncoder.pkl", "rb")
    ohe = pickle.load(pickle_file)
    pickle_file = open("./models/classifier.pkl", "rb")
    model = pickle.load(pickle_file)
    pickle_file = open("./models/list_mapping.pkl", "rb")
    encodings = pickle.load(pickle_file)
    input_array_df = pd.DataFrame(input_list_yield)
    input_array_df = input_array_df.T
    columns = [0, 1]
    oneHotEncodedFeature = ohe.transform(input_array_df[columns]).toarray()
    df_encoded = pd.DataFrame(oneHotEncodedFeature)
    df_final = pd.concat(
        [df_encoded, input_array_df.drop(columns, axis=1)], axis=1)
    X = df_final.values
    X[0, 680] = encodings[0][X[0, 680]]
    X[0, 681] = encodings[1][X[0, 681]]
    prediction = model.predict(X.reshape(1, -1))
    prediction = float(prediction)
    return prediction


def img_to_bytes(img_path):
    img_bytes = path.Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


# @st.cache(allow_output_mutation=True)


def main():

    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"><font size="10"> KISAAN AI 🌱 </font></h1>
    </div>
    <style>
    .reportview-container .main{
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

    st.markdown(html_temp, unsafe_allow_html=True)

    # image = Image.open("./assets/logo.jpg")

    # st.image(
    #     image,
    #     caption=None,
    #     width=200,
    #     use_column_width=None,
    #     clamp=True,
    #     channels="RGB",
    #     output_format="auto",
    # )

    selection = st.radio(
        "",
        [
            "Crop Disease Detection",
            "Crop Recommendation",
            "Yield Prediction",
            "Crop Price Recommendation",
            "Crop Health",
        ],
    )
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

    if selection == "Crop Health":
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Health</b></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(textbg.format(bgcolor, fontcolor), unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b> A good harvest is ensured by several factors such as availability of water, soil fertility, protecting crops from rodents, timely use of pesticides & other useful chemicals and nature. While a lot of these factors are difficult to control for, the amount and frequency of pesticides is something the farmer can control.</b></font></h1>
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
        Estimated_Insects_Count = st.sidebar.text_input(
            "Estimated Insects Count per square meter", "Type Here"
        )
        Crop_Type = st.sidebar.selectbox(
            "Select Your Crop Type", ("Food Crop", "Cash Crop")
        )
        Soil_Type = st.sidebar.selectbox(
            "Select Your Soil Type", ("Dry", "Wet"))
        Pesticide_Use_Category = st.sidebar.selectbox(
            "Pesticide Use Category", ("Never",
                                       "Previously Used", "Currently Using")
        )
        Number_Doses_Week = st.sidebar.text_input(
            "Number Doses per Week", "Type Here")
        Number_Weeks_Used = st.sidebar.text_input(
            "Number of Weeks Used", "Type Here")
        Number_Weeks_Quit = st.sidebar.text_input(
            "Number of Weeks Quit", "Type Here")
        Season = st.sidebar.selectbox("Season", ("Kharif", "Rabi", "Zaid"))

        if Crop_Type == "Food Crop":
            Crop_Type = 0
        else:
            Crop_Type = 1

        if Soil_Type == "Dry":
            Soil_Type = 0
        else:
            Soil_Type = 1

        if Pesticide_Use_Category == "Never":
            Pesticide_Use_Category = 1
        elif Pesticide_Use_Category == "Previously Used":
            Pesticide_Use_Category = 2
        else:
            Pesticide_Use_Category = 3

        if Season == "Kharif":
            Season = 1
        elif Season == "Rabi":
            Season = 2
        else:
            Season = 3

        result = ""
        if st.button("Predict"):
            result = predict_crop_damage(
                Estimated_Insects_Count,
                Crop_Type,
                Soil_Type,
                Pesticide_Use_Category,
                Number_Doses_Week,
                Number_Weeks_Used,
                Number_Weeks_Quit,
                Season,
            )
        if result == 0:
            result = "Alive"
        elif result == 1:
            result = "Possible Damage due to other causes"
        elif result == 2:
            result = "Damage due to Pesticides"

        st.success("The crop is {}".format(result))

    elif selection == "Crop Disease Detection":

        # st.set_option('depreciation.showfileUploaderEncoding',False)
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

        # text_io = io.TextIOWrapper(file)
        if st.button("Detect"):
            if file is None:
                st.sidebar.text("please upload an image file")

            else:
                image = Image.open(file)
                st.image(image, use_column_width=True)
                predictions = import_and_predict(image, model5)
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
        N = st.sidebar.number_input("Nitrogen", 1, 10000)
        P = st.sidebar.number_input("Phosporus", 1, 10000)
        K = st.sidebar.number_input("Potassium", 1, 10000)
        temp = st.sidebar.number_input("Temperature", 0.0, 100000.0)
        humidity = st.sidebar.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.sidebar.number_input("Ph", 0.0, 100000.0)
        rainfall = st.sidebar.number_input("Rainfall in mm", 0.0, 100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        if st.button("Predict"):

            prediction = model4.predict(single_pred)

            st.success(
                f"{prediction.item().title()} are recommended by the A.I for your farm."
            )

    elif selection == "Crop Price Recommendation":
        # st.sidebar.markdown('<p>Graphs are auto-updated on each change.</p>', unsafe_allow_html=True)
        textbg = """
        <div style="background-color:{};background: rgba(60, 179, 113, 0.8)">
        <h1 style="color:{};text-align:center;"><b>Crop Price Recommendation</b></h1>
        </div>
        """
        bgcolor = ""
        fontcolor = "white"
        st.markdown(textbg.format(bgcolor, fontcolor), unsafe_allow_html=True)

        text = """
        <div style="background-color:{};">
        <h1 style="color:{};text-align:center;"><font size=4><b>Crop price forecasting techniques are essential because they enable supply chain planners and government bodies to take appropriate actions by estimating market factors such as demand and supply. In emerging economies such as India, the crop prices at marketplaces are manually entered every day, prone to human-induced errors like incorrect data or access to no data for many days. In addition to such human errors, the fluctuations in the prices themselves make creating a stable and robust forecasting solution a necessity.</b></font></h1>
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
        list1 = []
        state_selection = st.sidebar.selectbox(
            "Select Your State",
            (
                "Andaman and Nicobar",
                "Andhra Pradesh",
                "Assam",
                "Chattisgarh",
                "Goa",
                "Gujarat",
                "Haryana",
                "Himachal Pradesh",
                "Jammu and Kashmir",
                "Karnataka",
                "Kerala",
                "Madhya Pradesh",
                "Maharashtra",
                "Manipur",
                "Meghalaya",
                "Nagaland",
                "Odisha",
                "Pondicherry",
                "Punjab",
                "Rajasthan",
                "Tamil Nadu",
                "Telangana",
                "Tripura",
                "Uttar Pradesh",
                "Uttrakhand",
                "West Bengal",
            ),
        )

        district_selection = st.sidebar.selectbox(
            "Select Your District",
            (
                "South Andaman",
                "Chittor",
                "Kurnool",
                "West Godavari",
                "Cachar",
                "Darrang",
                "Dhubri",
                "Jorhat",
                "Kamrup",
                "Sonitpur",
                "Bastar",
                "Kanker",
                "Surajpur",
                "North Goa",
                "Amreli",
                "Anand",
                "Bharuch",
                "Kachchh",
                "Kheda",
                "Panchmahals",
                "Surat",
                "Vadodara(Baroda)",
                "Valsad",
                "Ambala",
                "Faridabad",
                "Gurgaon",
                "Kurukshetra",
                "Mewat",
                "Panipat",
                "Kangra",
                "Kullu",
                "Badgam",
                "Bangalore",
                "Kolar",
                "Tumkur",
                "Alappuzha",
                "Ernakulam",
                "Kannur",
                "Kasargod",
                "Kollam",
                "Kottayam",
                "Malappuram",
                "Thirssur",
                "Thiruvananthapuram",
                "Anupur",
                "Badwani",
                "Dhar",
                "Dindori",
                "Jhabua",
                "Khandwa",
                "Narsinghpur",
                "Sheopur",
                "Ahmednagar",
                "Buldhana",
                "Jalgaon",
                "Kolhapur",
                "Nagpur",
                "Nanded",
                "Nashik",
                "Pune",
                "Satara",
                "Sholapur",
                "Bishnupur",
                "Chandel",
                "Imphal East",
                "Imphal West",
                "Thoubal",
                "East Khasi Hills",
                "Mokokchung",
                "Angul",
                "Balasore",
                "Bargarh",
                "Dhenkanal",
                "Gajapati",
                "Ganjam",
                "Jharsuguda",
                "Mayurbhanja",
                "Nowarangpur",
                "Sundergarh",
                "Karaikal",
                "Amritsar",
                "Barnala",
                "Bhatinda",
                "Gurdaspur",
                "Hoshiarpur",
                "Jalandhar",
                "Ludhiana",
                "Mansa",
                "Muktsar",
                "Patiala",
                "Ropar (Rupnagar)",
                "Tarntaran",
                "Baran",
                "Barmer",
                "Bikaner",
                "Chittorgarh",
                "Jalore",
                "Jhalawar",
                "Kota",
                "Sikar",
                "Tonk",
                "Coimbatore",
                "Cuddalore",
                "Dharmapuri",
                "Dindigul",
                "Erode",
                "Kancheepuram",
                "Krishnagiri",
                "Madurai",
                "Nagercoil (Kannyiakumari)",
                "Namakkal",
                "Ramanathapuram",
                "Salem",
                "Sivaganga",
                "Thanjavur",
                "Thiruvannamalai",
                "Vellore",
                "Villupuram",
                "Virudhunagar",
                "Hyderabad",
                "Karimnagar",
                "Khammam",
                "Nalgonda",
                "Nizamabad",
                "Ranga Reddy",
                "North Tripura",
                "Sepahijala",
                "South District",
                "Agra",
                "Aligarh",
                "Allahabad",
                "Baghpat",
                "Bahraich",
                "Ballia",
                "Bhadohi(Sant Ravi Nagar)",
                "Bulandshahar",
                "Chitrakut",
                "Etah",
                "Etawah",
                "Farukhabad",
                "Fatehpur",
                "Gautam Budh Nagar",
                "Ghaziabad",
                "Jalaun (Orai)",
                "Jhansi",
                "Kannuj",
                "Kanpur",
                "Khiri (Lakhimpur)",
                "Lakhimpur",
                "Mahoba",
                "Mau(Maunathbhanjan)",
                "Mirzapur",
                "Muradabad",
                "Muzaffarnagar",
                "Raebarelli",
                "Rampur",
                "Saharanpur",
                "Shahjahanpur",
                "Sitapur",
                "Sonbhadra",
                "Dehradoon",
                "UdhamSinghNagar",
                "Bankura",
                "Burdwan",
                "Hooghly",
                "Jalpaiguri",
                "Malda",
                "Medinipur(W)",
                "Murshidabad",
                "Nadia",
                "North 24 Parganas",
                "Puruliya",
                "Sounth 24 Parganas",
                "Uttar Dinajpur",
            ),
        )

        market_selection = st.sidebar.selectbox(
            "Select Your Market",
            (
                "Port Blair",
                "Kalikiri",
                "Mulakalacheruvu",
                "Vayalapadu",
                "Banaganapalli",
                "Attili",
                "Cachar",
                "Kharupetia",
                "Gauripur",
                "Jorhat",
                "Pamohi(Garchuk)",
                "Dhekiajuli",
                "Jagdalpur",
                "Charama",
                "Lakhanpuri",
                "Narharpur",
                "Pratappur",
                "Sanquelim",
                "Damnagar",
                "Khambhat(Grain Market)",
                "Umreth",
                "Ankleshwar",
                "Bachau",
                "Nadiyad(Chaklasi)",
                "Gogamba",
                "Gogamba(Similiya)",
                "Bardoli(Katod)",
                "Mandvi",
                "S.Mandvi",
                "Vyra",
                "Bodeli",
                "Bodeli(Hadod)",
                "Bodeli(Kalediya)",
                "Bodeli(Modasar)",
                "Padra",
                "Chikli(Khorgam)",
                "Shahzadpur",
                "Faridabad",
                "Pataudi",
                "Pipli",
                "Taura",
                "Madlauda",
                "Kangra(Jaisinghpur)",
                "Kangra(Jassour)",
                "Palampur",
                "Bhuntar",
                "Zaloosa-Chararishrief (F&V)",
                "Ramanagara",
                "Kolar",
                "Gubbi",
                "Aroor",
                "Chengannur",
                "Mannar",
                "Piravam",
                "Taliparamba",
                "Kasargod",
                "Manjeswaram",
                "Anchal",
                "Chathanoor",
                "Kaliyanchanda",
                "Ettumanoor",
                "Kottayam",
                "Kondotty",
                "Chelakkara",
                "Irinjalakkuda",
                "Kodungalloor",
                "Maranelloor",
                "Parassala",
                "Jaithari",
                "Sendhwa",
                "Dhar",
                "Dhar(F&V)",
                "Gorakhpur",
                "Thandla",
                "Pandhana(F&V)",
                "Gadarwada",
                "Sheopurkalan",
                "Syopurkalan(F&V)",
                "Newasa(Ghodegaon)",
                "Rahata",
                "Deoulgaon Raja",
                "Yawal",
                "Kolhapur",
                "Kolhapur(Malkapur)",
                "Savner",
                "Bhokar",
                "Lasalgaon(Niphad)",
                "Pune",
                "Pune(Hadapsar)",
                "Pune(Khadiki)",
                "Pune(Pimpri)",
                "Karad",
                "Akluj",
                "Mangal Wedha",
                "Pandharpur",
                "Bishenpur",
                "Moreh",
                "Lamlong Bazaar",
                "Imphal",
                "Thoubal",
                "Shillong",
                "Mangkolemba",
                "Angul",
                "Angul(Jarapada)",
                "Pallahara",
                "Jaleswar",
                "Nilagiri",
                "Godabhaga",
                "Hindol",
                "Kasinagar",
                "Parlakhemundi",
                "Bhanjanagar",
                "Digapahandi",
                "Jharsuguda",
                "Betnoti",
                "Saraskana",
                "Nawarangpur",
                "Bonai",
                "Panposh",
                "Karaikal",
                "Chogawan",
                "Mehta",
                "Dhanaula",
                "Bhucho",
                "Dera Baba Nanak",
                "Dhariwal",
                "Kalanaur",
                "Dasuya",
                "Adampur",
                "Lohian Khas",
                "Mehatpur",
                "Doraha",
                "Sahnewal",
                "Mansa",
                "Malout",
                "Ghanaur",
                "Rajpura",
                "Morinda",
                "Harike",
                "Patti",
                "Anta",
                "Barmer",
                "Lunkaransar",
                "Begu",
                "Jalore",
                "Khanpur",
                "Itawa",
                "Sri Madhopur",
                "Surajgarh",
                "Deoli",
                "Uniyara",
                "Anaimalai",
                "Annur",
                "Coimbatore",
                "Karamadai",
                "Madathukulam",
                "Negamam",
                "Palladam",
                "Pethappampatti",
                "Pollachi",
                "Pongalur",
                "Pudupalayam",
                "Senjeri",
                "Sevur",
                "Thondamuthur",
                "Udumalpet",
                "Cuddalore",
                "Kurinchipadi",
                "Dharampuri",
                "Palakode",
                "Papparapatti",
                "Pappireddipatti",
                "Pennagaram",
                "Gopalpatti",
                "Palani",
                "Dharapuram",
                "Elumathur",
                "Kodumudi",
                "Kunnathur",
                "Muthur",
                "Vellakkoil",
                "Acharapakkam",
                "Chengalpattu",
                "Gummidipoondy",
                "Kanchipuram",
                "Maduranthagam",
                "Pallipattu",
                "Ponner",
                "Sunguvarchatram",
                "Uthiramerur",
                "Bargur",
                "Pochampalli",
                "Uthangarai",
                "Melur",
                "Thirumangalam",
                "Kalliakavillai",
                "Namagiripettai",
                "Namakkal",
                "Rasipuram",
                "Tiruchengode",
                "Velur",
                "Sivagangai",
                "Attur",
                "Gangavalli",
                "Karumanturai",
                "Kolathur",
                "Konganapuram",
                "Omalur",
                "Salem",
                "Thalaivasal",
                "Thammampati",
                "Vazhapadi",
                "Singampuneri",
                "Budalur",
                "Kumbakonam",
                "Orathanadu",
                "Papanasam",
                "Thanjavur",
                "Thiruppananthal",
                "Vallam",
                "Cheyyar",
                "Ammoor",
                "Kalavai",
                "Vellore",
                "Avalurpet",
                "Chinnasalem",
                "Gingee",
                "Kallakurichi",
                "Manalurpet",
                "Sangarapuram",
                "Tindivanam",
                "Tiruvennainallur",
                "Vikkiravandi",
                "Villupuram",
                "Rajapalayam",
                "Sathur",
                "Mahboob Manison",
                "Koratla",
                "Vemulawada",
                "Madhira",
                "Nalgonda",
                "Ramannapet",
                "Voligonda",
                "Pitlam",
                "Chevella",
                "Dasda",
                "Bishalgarh",
                "Bishramganj",
                "Barpathari",
                "Achnera",
                "Fatehabad",
                "Jagnair",
                "Jarar",
                "Khairagarh",
                "Samsabad",
                "Atrauli",
                "Sirsa",
                "Bagpat",
                "Baraut",
                "Mihipurwa",
                "Risia",
                "Ruperdeeha",
                "Chitwadagaon",
                "Gopiganj",
                "Divai",
                "Gulavati",
                "Siyana",
                "Mau(Chitrakut)",
                "Aliganj",
                "Jasvantnagar",
                "Kamlaganj",
                "Bindki",
                "Kishunpur",
                "Javer",
                "Hapur",
                "Ait",
                "Kadaura",
                "Baruwasagar",
                "Chirgaon",
                "Chhibramau(Kannuj)",
                "Rura",
                "Mohammdi",
                "Tikonia",
                "Paliakala",
                "Charkhari",
                "Doharighat",
                "Mirzapur",
                "Sambhal",
                "Muzzafarnagar",
                "Shahpur",
                "Thanabhawan",
                "Lalganj",
                "Milak",
                "Chutmalpur",
                "Devband",
                "Gangoh",
                "Nakud",
                "Nanuta",
                "Sultanpurchilkana",
                "Jalalabad",
                "Viswan",
                "Dudhi",
                "Robertsganj",
                "Dehradoon",
                "Kashipur",
                "Bishnupur(Bankura)",
                "Khatra",
                "Asansol",
                "Burdwan",
                "Guskara(Burdwan)",
                "Kalna",
                "Katwa",
                "Memari",
                "Pandua",
                "Dhupguri",
                "Samsi",
                "Ghatal",
                "Beldanga",
                "Jangipur",
                "Nadia",
                "Habra",
                "Balarampur",
                "Purulia",
                "Baruipur(Canning)",
                "Islampur",
                "Raiganj",
            ),
        )

        commodity_selection = st.sidebar.selectbox(
            "Select Your Commodity",
            (
                "Amaranthus",
                "Banana - Green",
                "Bhindi(Ladies Finger)",
                "Bitter gourd",
                "Black pepper",
                "Bottle gourd",
                "Brinjal",
                "Cabbage",
                "Carrot",
                "Cauliflower",
                "Cluster beans",
                "Coconut",
                "Colacasia",
                "Onion",
                "Potato",
                "Tomato",
                "Bengal Gram(Gram)(Whole)",
                "Jowar(Sorghum)",
                "Paddy(Dhan)(Common)",
                "Lentil (Masur)(Whole)",
                "Rice",
                "Cucumbar(Kheera)",
                "Field Pea",
                "French Beans (Frasbean)",
                "Green Chilli",
                "Knool Khol",
                "Pumpkin",
                "Raddish",
                "Black Gram (Urd Beans)(Whole)",
                "Green Gram (Moong)(Whole)",
                "Jute",
                "Maida Atta",
                "Mustard",
                "Wheat Atta",
                "Garlic",
                "Masur Dal",
                "Ridgeguard(Tori)",
                "Arecanut(Betelnut/Supari)",
                "Arhar (Tur/Red Gram)(Whole)",
                "Maize",
                "Dry Chillies",
                "Groundnut",
                "Capsicum",
                "Guar",
                "Lemon",
                "Bajra(Pearl Millet/Cumbu)",
                "Castor Seed",
                "Coriander(Leaves)",
                "Cowpea(Veg)",
                "Drumstick",
                "Elephant Yam (Suran)",
                "Ginger(Green)",
                "Indian Beans (Seam)",
                "Methi(Leaves)",
                "Onion Green",
                "Peas cod",
                "Pegeon Pea (Arhar Fali)",
                "Sponge gourd",
                "Surat Beans (Papadi)",
                "Sweet Potato",
                "Tinda",
                "Guar Seed(Cluster Beans Seed)",
                "Cotton",
                "Wheat",
                "Gram Raw(Chholia)",
                "Little gourd (Kundru)",
                "Round gourd",
                "Leafy Vegetable",
                "Mint(Pudina)",
                "Papaya (Raw)",
                "Spinach",
                "Pointed gourd (Parval)",
                "Banana",
                "Ber(Zizyphus/Borehannu)",
                "Grapes",
                "Kinnow",
                "Peas Wet",
                "Apple",
                "Orange",
                "Pomegranate",
                "Papaya",
                "Chikoos(Sapota)",
                "Mashrooms",
                "Mousambi(Sweet Lime)",
                "Pineapple",
                "Guava",
                "Turnip",
                "Ginger(Dry)",
                "Squash(Chappal Kadoo)",
                "Beans",
                "Beetroot",
                "Chilly Capsicum",
                "Green Avare (W)",
                "Seemebadnekai",
                "Snakeguard",
                "Suvarna Gadde",
                "Water Melon",
                "Copra",
                "Amphophalus",
                "Ashgourd",
                "Coconut Oil",
                "Rubber",
                "Cashewnuts",
                "Pepper garbled",
                "Coconut Seed",
                "Long Melon(Kakri)",
                "Tapioca",
                "Turmeric (raw)",
                "Mango (Raw-Ripe)",
                "Amla(Nelli Kai)",
                "Duster Beans",
                "Mango",
                "Soyabean",
                "Linseed",
                "Niger Seed (Ramtil)",
                "Green Gram Dal (Moong Dal)",
                "Lime",
                "Turmeric",
                "Karbuja(Musk Melon)",
                "Pear(Marasebu)",
                "Rajgir",
                "Sweet Pumpkin",
                "Tender Coconut",
                "Bengal Gram Dal (Chana Dal)",
                "Betal Leaves",
                "Broken Rice",
                "Gur(Jaggery)",
                "Sugar",
                "Sesamum(Sesame,Gingelly,Til)",
                "Moath Dal",
                "Corriander seed",
                "Ground Nut Seed",
                "Taramira",
                "Tobacco",
                "Tamarind Fruit",
                "Cowpea (Lobia/Karamani)",
                "Kulthi(Horse Gram)",
                "Ragi (Finger Millet)",
                "T.V. Cumbu",
                "Gingelly Oil",
                "Kodo Millet(Varagu)",
                "Hybrid Cumbu",
                "Karamani",
                "Thinai (Italian Millet)",
                "Wood",
                "Barley (Jau)",
                "Fish",
                "Green Peas",
                "Arhar Dal(Tur Dal)",
                "Black Gram Dal (Urd Dal)",
                "Mustard Oil",
                "Ghee",
                "Paddy(Dhan)(Basmati)",
                "White Pumpkin",
                "Peas(Dry)",
                "Plum",
            ),
        )

        variety_selection = st.sidebar.selectbox(
            "Select The Variety",
            (
                "Amaranthus",
                "Banana - Green",
                "Bhindi",
                "Other",
                "Cluster Beans",
                "Big",
                "Local",
                "Desi (Whole)",
                "Jowar ( White)",
                "Sona Mahsuri",
                "Paddy",
                "Masur Dal",
                "Common",
                "Fine",
                "Bitter Gourd",
                "Bottle Gourd",
                "Round",
                "Cabbage",
                "Carrot",
                "Cucumbar",
                "Field Pea",
                "French Beans (Frasbean)",
                "Green Chilly",
                "Knool Khol",
                "Badshah",
                "Pumpkin",
                "Raddish",
                "Hybrid",
                "Black Gram (Whole)",
                "Green (Whole)",
                "TD-5",
                "Maida Atta",
                "Mustard",
                "Medium",
                "Wheat Atta",
                "Garlic",
                "Onion",
                "Potato",
                "Brinjal",
                "Cauliflower",
                "Ridgeguard(Tori)",
                "Tomato",
                "Raw",
                "Dry",
                "Red",
                "Supari",
                "White",
                "Capsicum",
                "Gwar",
                "Lemon",
                "Deshi",
                "Castor seed",
                "Coriander",
                "Cowpea (Veg)",
                "Drumstick",
                "Elephant Yam (Suran)",
                "Green Ginger",
                "Indian Beans (Seam)",
                "Methi",
                "Onion Green",
                "Peas cod",
                "Pigeon Pea (Arhar Fali)",
                "Sponge gourd",
                "Surat Beans (Papadi)",
                "Sweet Potato",
                "Tinda",
                "Whole",
                "Deshi White",
                "G. R. 11",
                "Sabnam",
                "Shanker 6 (B) 30mm FIne",
                "Leafy Vegetables",
                "Ber(Zizyphus)",
                "Desi",
                "Green",
                "Peas Wet",
                "Spinach",
                "Kinnow",
                "Apple",
                "Beans (Whole)",
                "Beetroot",
                "Chilly Capsicum",
                "Green Avare (W)",
                "Papaya",
                "Seemebadanekai",
                "Snakeguard",
                "Suvarnagadde",
                "Water Melon",
                "other",
                "Amphophalus",
                "Nendra Bale",
                "Palayamthodan",
                "Robusta",
                "Ashgourd",
                "Copra",
                "Poovan",
                "Red Banana",
                "Banana - Ripe",
                "Mango - Raw-Ripe",
                "Tapioca",
                "Rasakathai",
                "Round/Long",
                "Colacasia",
                "Yellow",
                "Lokwan",
                "DCH-32  (Ginned)",
                "Green Gram Dal",
                "1001",
                "147 Average",
                "Hapus(Alphaso)",
                "Simla",
                "Hosur Red",
                "Bengal Gram Dal",
                "Turmeric (raw)",
                "New Variety",
                "Champa",
                "Orange",
                "api",
                "Sanna Bhatta",
                "Broken Rice",
                "NO 2",
                "Hybrid/Local",
                "B P T",
                "Pommani",
                "White Ponni",
                "Sesame",
                "American",
                "Black",
                "Nasik",
                "Sapota",
                "Guava",
                "Pine Apple",
                "Pomogranate",
                "Squash(Chappal Kadoo)",
                "999",
                "Bulb",
                "Coconut",
                "Ball",
                "DMV-7",
                "Hybrid Red (Cattle Feed)",
                "Chapathi",
                "ADT 37",
                "ADT 43",
                "ADT 36",
                "MCU 5",
                "Ponni",
                "Local(Raw)",
                "Finger",
                "GCH",
                "Super Ponni",
                "RCH-2",
                "TKM 9",
                "95/5",
                "A. Ponni",
                "ADT 39",
                "Bold Kernel",
                "1st Sort",
                "2nd Sort",
                "Cotton (Unginned)",
                " Subabul",
                "MTU-1010",
                "African Sarson",
                "Kala Masoor New",
                "Masuri",
                "Papaya (Raw)",
                "Jalander",
                "Dara",
                "Sarson(Black)",
                "III",
                "Delicious",
                "Pathari",
                "Mousambi",
                "Kasmir/Shimla - II",
                "Arkasheela Mattigulla",
                "(Red Nanital)",
                "Chips",
                "Paddy Coarse",
                "Rahu(Local)",
                "Laha(Sarsib)",
                "F.A.Q.",
                "777 New Ind",
                "Big 100 Kg",
                "1009 Kar",
                "Beedi",
                "Green Peas",
                "Arhar Dal(Tur)",
                "Black Gram Dal",
                "Lohi Black",
                "Mustard Oil",
                "Ghee",
                "Ankola",
                "Annigeri",
                "Hybrid Yellow (Cattle Feed)",
                "1121",
                "Beete (Rose)",
                "Khandsari",
                "White Pumpkin",
                "Annabesahai",
                "Nagpuri",
                "Coarse",
                "Chini",
                "Masoor Gola",
                "Peas(Dry)",
                "Plum",
                "Disara",
                "Jyoti",
                "Sonalika",
                "Swarna Masuri (New)",
                "Sweet Pumpkin",
                "Green Gram Dhall-I",
                "Kalyan",
                "Yellow (Black)",
                "Ratna",
                "Ratnachudi (718 5-749)",
                "H.Y.V.",
                "Moath Dal",
                "Beelary-Red",
                "Super Fine",
            ),
        )

        result = ""
        if st.button("Predict"):
            result = crop_price_prediction(
                [
                    state_selection,
                    district_selection,
                    market_selection,
                    commodity_selection,
                    variety_selection,
                ]
            )

        st.success("The suggested crop price (per quintal) is {}".format(result))

    elif selection == "Yield Prediction":
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

        state_selection = st.sidebar.selectbox(
            "Select Your State",
            (
                "Andaman and Nicobar Islands",
                "Andhra Pradesh",
                "Arunachal Pradesh",
                "Assam",
                "Bihar",
                "Chandigarh",
                "Chhattisgarh",
                "Dadra and Nagar Haveli",
                "Goa",
                "Gujarat",
                "Haryana",
                "Himachal Pradesh",
                "Jammu and Kashmir ",
                "Jharkhand",
                "Karnataka",
                "Kerala",
                "Madhya Pradesh",
                "Maharashtra",
                "Manipur",
                "Meghalaya",
                "Mizoram",
                "Nagaland",
                "Odisha",
                "Puducherry",
                "Punjab",
                "Rajasthan",
                "Sikkim",
                "Tamil Nadu",
                "Telangana ",
                "Tripura",
                "Uttar Pradesh",
                "Uttarakhand",
                "West Bengal",
            ),
        )

        district_selection = st.sidebar.selectbox(
            "Select Your District",
            (
                "NICOBARS",
                "NORTH AND MIDDLE ANDAMAN",
                "SOUTH ANDAMANS",
                "ANANTAPUR",
                "CHITTOOR",
                "EAST GODAVARI",
                "GUNTUR",
                "KADAPA",
                "KRISHNA",
                "KURNOOL",
                "PRAKASAM",
                "SPSR NELLORE",
                "SRIKAKULAM",
                "VISAKHAPATANAM",
                "VIZIANAGARAM",
                "WEST GODAVARI",
                "ANJAW",
                "CHANGLANG",
                "DIBANG VALLEY",
                "EAST KAMENG",
                "EAST SIANG",
                "KURUNG KUMEY",
                "LOHIT",
                "LONGDING",
                "LOWER DIBANG VALLEY",
                "LOWER SUBANSIRI",
                "NAMSAI",
                "PAPUM PARE",
                "TAWANG",
                "TIRAP",
                "UPPER SIANG",
                "UPPER SUBANSIRI",
                "WEST KAMENG",
                "WEST SIANG",
                "BAKSA",
                "BARPETA",
                "BONGAIGAON",
                "CACHAR",
                "CHIRANG",
                "DARRANG",
                "DHEMAJI",
                "DHUBRI",
                "DIBRUGARH",
                "DIMA HASAO",
                "GOALPARA",
                "GOLAGHAT",
                "HAILAKANDI",
                "JORHAT",
                "KAMRUP",
                "KAMRUP METRO",
                "KARBI ANGLONG",
                "KARIMGANJ",
                "KOKRAJHAR",
                "LAKHIMPUR",
                "MARIGAON",
                "NAGAON",
                "NALBARI",
                "SIVASAGAR",
                "SONITPUR",
                "TINSUKIA",
                "UDALGURI",
                "ARARIA",
                "ARWAL",
                "AURANGABAD",
                "BANKA",
                "BEGUSARAI",
                "BHAGALPUR",
                "BHOJPUR",
                "BUXAR",
                "DARBHANGA",
                "GAYA",
                "GOPALGANJ",
                "JAMUI",
                "JEHANABAD",
                "KAIMUR (BHABUA)",
                "KATIHAR",
                "KHAGARIA",
                "KISHANGANJ",
                "LAKHISARAI",
                "MADHEPURA",
                "MADHUBANI",
                "MUNGER",
                "MUZAFFARPUR",
                "NALANDA",
                "NAWADA",
                "PASHCHIM CHAMPARAN",
                "PATNA",
                "PURBI CHAMPARAN",
                "PURNIA",
                "ROHTAS",
                "SAHARSA",
                "SAMASTIPUR",
                "SARAN",
                "SHEIKHPURA",
                "SHEOHAR",
                "SITAMARHI",
                "SIWAN",
                "SUPAUL",
                "VAISHALI",
                "CHANDIGARH",
                "BALOD",
                "BALODA BAZAR",
                "BALRAMPUR",
                "BASTAR",
                "BEMETARA",
                "BIJAPUR",
                "BILASPUR",
                "DANTEWADA",
                "DHAMTARI",
                "DURG",
                "GARIYABAND",
                "JANJGIR-CHAMPA",
                "JASHPUR",
                "KABIRDHAM",
                "KANKER",
                "KONDAGAON",
                "KORBA",
                "KOREA",
                "MAHASAMUND",
                "MUNGELI",
                "NARAYANPUR",
                "RAIGARH",
                "RAIPUR",
                "RAJNANDGAON",
                "SUKMA",
                "SURAJPUR",
                "SURGUJA",
                "DADRA AND NAGAR HAVELI",
                "NORTH GOA",
                "SOUTH GOA",
                "AHMADABAD",
                "AMRELI",
                "ANAND",
                "BANAS KANTHA",
                "BHARUCH",
                "BHAVNAGAR",
                "DANG",
                "DOHAD",
                "GANDHINAGAR",
                "JAMNAGAR",
                "JUNAGADH",
                "KACHCHH",
                "KHEDA",
                "MAHESANA",
                "NARMADA",
                "NAVSARI",
                "PANCH MAHALS",
                "PATAN",
                "PORBANDAR",
                "RAJKOT",
                "SABAR KANTHA",
                "SURAT",
                "SURENDRANAGAR",
                "TAPI",
                "VADODARA",
                "VALSAD",
                "AMBALA",
                "BHIWANI",
                "FARIDABAD",
                "FATEHABAD",
                "GURGAON",
                "HISAR",
                "JHAJJAR",
                "JIND",
                "KAITHAL",
                "KARNAL",
                "KURUKSHETRA",
                "MAHENDRAGARH",
                "MEWAT",
                "PALWAL",
                "PANCHKULA",
                "PANIPAT",
                "REWARI",
                "ROHTAK",
                "SIRSA",
                "SONIPAT",
                "YAMUNANAGAR",
                "CHAMBA",
                "HAMIRPUR",
                "KANGRA",
                "KINNAUR",
                "KULLU",
                "LAHUL AND SPITI",
                "MANDI",
                "SHIMLA",
                "SIRMAUR",
                "SOLAN",
                "UNA",
                "ANANTNAG",
                "BADGAM",
                "BANDIPORA",
                "BARAMULLA",
                "DODA",
                "GANDERBAL",
                "JAMMU",
                "KARGIL",
                "KATHUA",
                "KISHTWAR",
                "KULGAM",
                "KUPWARA",
                "LEH LADAKH",
                "POONCH",
                "PULWAMA",
                "RAJAURI",
                "RAMBAN",
                "REASI",
                "SAMBA",
                "SHOPIAN",
                "SRINAGAR",
                "UDHAMPUR",
                "BOKARO",
                "CHATRA",
                "DEOGHAR",
                "DHANBAD",
                "DUMKA",
                "EAST SINGHBUM",
                "GARHWA",
                "GIRIDIH",
                "GODDA",
                "GUMLA",
                "HAZARIBAGH",
                "JAMTARA",
                "KHUNTI",
                "KODERMA",
                "LATEHAR",
                "LOHARDAGA",
                "PAKUR",
                "PALAMU",
                "RAMGARH",
                "RANCHI",
                "SAHEBGANJ",
                "SARAIKELA KHARSAWAN",
                "SIMDEGA",
                "WEST SINGHBHUM",
                "BAGALKOT",
                "BANGALORE RURAL",
                "BELGAUM",
                "BELLARY",
                "BENGALURU URBAN",
                "BIDAR",
                "CHAMARAJANAGAR",
                "CHIKBALLAPUR",
                "CHIKMAGALUR",
                "CHITRADURGA",
                "DAKSHIN KANNAD",
                "DAVANGERE",
                "DHARWAD",
                "GADAG",
                "GULBARGA",
                "HASSAN",
                "HAVERI",
                "KODAGU",
                "KOLAR",
                "KOPPAL",
                "MANDYA",
                "MYSORE",
                "RAICHUR",
                "RAMANAGARA",
                "SHIMOGA",
                "TUMKUR",
                "UDUPI",
                "UTTAR KANNAD",
                "YADGIR",
                "ALAPPUZHA",
                "ERNAKULAM",
                "IDUKKI",
                "KANNUR",
                "KASARAGOD",
                "KOLLAM",
                "KOTTAYAM",
                "KOZHIKODE",
                "MALAPPURAM",
                "PALAKKAD",
                "PATHANAMTHITTA",
                "THIRUVANANTHAPURAM",
                "THRISSUR",
                "WAYANAD",
                "AGAR MALWA",
                "ALIRAJPUR",
                "ANUPPUR",
                "ASHOKNAGAR",
                "BALAGHAT",
                "BARWANI",
                "BETUL",
                "BHIND",
                "BHOPAL",
                "BURHANPUR",
                "CHHATARPUR",
                "CHHINDWARA",
                "DAMOH",
                "DATIA",
                "DEWAS",
                "DHAR",
                "DINDORI",
                "GUNA",
                "GWALIOR",
                "HARDA",
                "HOSHANGABAD",
                "INDORE",
                "JABALPUR",
                "JHABUA",
                "KATNI",
                "KHANDWA",
                "KHARGONE",
                "MANDLA",
                "MANDSAUR",
                "MORENA",
                "NARSINGHPUR",
                "NEEMUCH",
                "PANNA",
                "RAISEN",
                "RAJGARH",
                "RATLAM",
                "REWA",
                "SAGAR",
                "SATNA",
                "SEHORE",
                "SEONI",
                "SHAHDOL",
                "SHAJAPUR",
                "SHEOPUR",
                "SHIVPURI",
                "SIDHI",
                "SINGRAULI",
                "TIKAMGARH",
                "UJJAIN",
                "UMARIA",
                "VIDISHA",
                "AHMEDNAGAR",
                "AKOLA",
                "AMRAVATI",
                "BEED",
                "BHANDARA",
                "BULDHANA",
                "CHANDRAPUR",
                "DHULE",
                "GADCHIROLI",
                "GONDIA",
                "HINGOLI",
                "JALGAON",
                "JALNA",
                "KOLHAPUR",
                "LATUR",
                "MUMBAI",
                "NAGPUR",
                "NANDED",
                "NANDURBAR",
                "NASHIK",
                "OSMANABAD",
                "PALGHAR",
                "PARBHANI",
                "PUNE",
                "RAIGAD",
                "RATNAGIRI",
                "SANGLI",
                "SATARA",
                "SINDHUDURG",
                "SOLAPUR",
                "THANE",
                "WARDHA",
                "WASHIM",
                "YAVATMAL",
                "BISHNUPUR",
                "CHANDEL",
                "CHURACHANDPUR",
                "IMPHAL EAST",
                "IMPHAL WEST",
                "SENAPATI",
                "TAMENGLONG",
                "THOUBAL",
                "UKHRUL",
                "EAST GARO HILLS",
                "EAST JAINTIA HILLS",
                "EAST KHASI HILLS",
                "NORTH GARO HILLS",
                "RI BHOI",
                "SOUTH GARO HILLS",
                "SOUTH WEST GARO HILLS",
                "SOUTH WEST KHASI HILLS",
                "WEST GARO HILLS",
                "WEST JAINTIA HILLS",
                "WEST KHASI HILLS",
                "AIZAWL",
                "CHAMPHAI",
                "KOLASIB",
                "LAWNGTLAI",
                "LUNGLEI",
                "MAMIT",
                "SAIHA",
                "SERCHHIP",
                "DIMAPUR",
                "KIPHIRE",
                "KOHIMA",
                "LONGLENG",
                "MOKOKCHUNG",
                "MON",
                "PEREN",
                "PHEK",
                "TUENSANG",
                "WOKHA",
                "ZUNHEBOTO",
                "ANUGUL",
                "BALANGIR",
                "BALESHWAR",
                "BARGARH",
                "BHADRAK",
                "BOUDH",
                "CUTTACK",
                "DEOGARH",
                "DHENKANAL",
                "GAJAPATI",
                "GANJAM",
                "JAGATSINGHAPUR",
                "JAJAPUR",
                "JHARSUGUDA",
                "KALAHANDI",
                "KANDHAMAL",
                "KENDRAPARA",
                "KENDUJHAR",
                "KHORDHA",
                "KORAPUT",
                "MALKANGIRI",
                "MAYURBHANJ",
                "NABARANGPUR",
                "NAYAGARH",
                "NUAPADA",
                "PURI",
                "RAYAGADA",
                "SAMBALPUR",
                "SONEPUR",
                "SUNDARGARH",
                "KARAIKAL",
                "MAHE",
                "PONDICHERRY",
                "YANAM",
                "AMRITSAR",
                "BARNALA",
                "BATHINDA",
                "FARIDKOT",
                "FATEHGARH SAHIB",
                "FAZILKA",
                "FIROZEPUR",
                "GURDASPUR",
                "HOSHIARPUR",
                "JALANDHAR",
                "KAPURTHALA",
                "LUDHIANA",
                "MANSA",
                "MOGA",
                "MUKTSAR",
                "NAWANSHAHR",
                "PATHANKOT",
                "PATIALA",
                "RUPNAGAR",
                "S.A.S NAGAR",
                "SANGRUR",
                "TARN TARAN",
                "AJMER",
                "ALWAR",
                "BANSWARA",
                "BARAN",
                "BARMER",
                "BHARATPUR",
                "BHILWARA",
                "BIKANER",
                "BUNDI",
                "CHITTORGARH",
                "CHURU",
                "DAUSA",
                "DHOLPUR",
                "DUNGARPUR",
                "GANGANAGAR",
                "HANUMANGARH",
                "JAIPUR",
                "JAISALMER",
                "JALORE",
                "JHALAWAR",
                "JHUNJHUNU",
                "JODHPUR",
                "KARAULI",
                "KOTA",
                "NAGAUR",
                "PALI",
                "PRATAPGARH",
                "RAJSAMAND",
                "SAWAI MADHOPUR",
                "SIKAR",
                "SIROHI",
                "TONK",
                "UDAIPUR",
                "EAST DISTRICT",
                "NORTH DISTRICT",
                "SOUTH DISTRICT",
                "WEST DISTRICT",
                "ARIYALUR",
                "COIMBATORE",
                "CUDDALORE",
                "DHARMAPURI",
                "DINDIGUL",
                "ERODE",
                "KANCHIPURAM",
                "KANNIYAKUMARI",
                "KARUR",
                "KRISHNAGIRI",
                "MADURAI",
                "NAGAPATTINAM",
                "NAMAKKAL",
                "PERAMBALUR",
                "PUDUKKOTTAI",
                "RAMANATHAPURAM",
                "SALEM",
                "SIVAGANGA",
                "THANJAVUR",
                "THE NILGIRIS",
                "THENI",
                "THIRUVALLUR",
                "THIRUVARUR",
                "TIRUCHIRAPPALLI",
                "TIRUNELVELI",
                "TIRUPPUR",
                "TIRUVANNAMALAI",
                "TUTICORIN",
                "VELLORE",
                "VILLUPURAM",
                "VIRUDHUNAGAR",
                "ADILABAD",
                "HYDERABAD",
                "KARIMNAGAR",
                "KHAMMAM",
                "MAHBUBNAGAR",
                "MEDAK",
                "NALGONDA",
                "NIZAMABAD",
                "RANGAREDDI",
                "WARANGAL",
                "DHALAI",
                "GOMATI",
                "KHOWAI",
                "NORTH TRIPURA",
                "SEPAHIJALA",
                "SOUTH TRIPURA",
                "UNAKOTI",
                "WEST TRIPURA",
                "AGRA",
                "ALIGARH",
                "ALLAHABAD",
                "AMBEDKAR NAGAR",
                "AMETHI",
                "AMROHA",
                "AURAIYA",
                "AZAMGARH",
                "BAGHPAT",
                "BAHRAICH",
                "BALLIA",
                "BANDA",
                "BARABANKI",
                "BAREILLY",
                "BASTI",
                "BIJNOR",
                "BUDAUN",
                "BULANDSHAHR",
                "CHANDAULI",
                "CHITRAKOOT",
                "DEORIA",
                "ETAH",
                "ETAWAH",
                "FAIZABAD",
                "FARRUKHABAD",
                "FATEHPUR",
                "FIROZABAD",
                "GAUTAM BUDDHA NAGAR",
                "GHAZIABAD",
                "GHAZIPUR",
                "GONDA",
                "GORAKHPUR",
                "HAPUR",
                "HARDOI",
                "HATHRAS",
                "JALAUN",
                "JAUNPUR",
                "JHANSI",
                "KANNAUJ",
                "KANPUR DEHAT",
                "KANPUR NAGAR",
                "KASGANJ",
                "KAUSHAMBI",
                "KHERI",
                "KUSHI NAGAR",
                "LALITPUR",
                "LUCKNOW",
                "MAHARAJGANJ",
                "MAHOBA",
                "MAINPURI",
                "MATHURA",
                "MAU",
                "MEERUT",
                "MIRZAPUR",
                "MORADABAD",
                "MUZAFFARNAGAR",
                "PILIBHIT",
                "RAE BARELI",
                "RAMPUR",
                "SAHARANPUR",
                "SAMBHAL",
                "SANT KABEER NAGAR",
                "SANT RAVIDAS NAGAR",
                "SHAHJAHANPUR",
                "SHAMLI",
                "SHRAVASTI",
                "SIDDHARTH NAGAR",
                "SITAPUR",
                "SONBHADRA",
                "SULTANPUR",
                "UNNAO",
                "VARANASI",
                "ALMORA",
                "BAGESHWAR",
                "CHAMOLI",
                "CHAMPAWAT",
                "DEHRADUN",
                "HARIDWAR",
                "NAINITAL",
                "PAURI GARHWAL",
                "PITHORAGARH",
                "RUDRA PRAYAG",
                "TEHRI GARHWAL",
                "UDAM SINGH NAGAR",
                "UTTAR KASHI",
                "24 PARAGANAS NORTH",
                "24 PARAGANAS SOUTH",
                "BANKURA",
                "BARDHAMAN",
                "BIRBHUM",
                "COOCHBEHAR",
                "DARJEELING",
                "DINAJPUR DAKSHIN",
                "DINAJPUR UTTAR",
                "HOOGHLY",
                "HOWRAH",
                "JALPAIGURI",
                "MALDAH",
                "MEDINIPUR EAST",
                "MEDINIPUR WEST",
                "MURSHIDABAD",
                "NADIA",
                "PURULIA",
            ),
        )

        crop_year = 2014

        season_select = st.sidebar.selectbox(
            "Select The Season",
            ("Kharif", "Whole Year", "Autumn", "Rabi", "Summer", "Winter"),
        )

        if season_select == "Kharif":
            season_select = "Kharif     "

        if season_select == "Whole Year":
            season_select = "Whole Year "

        if season_select == "Autumn":
            season_select = "Autumn     "

        if season_select == "Rabi":
            season_select = "Rabi       "

        if season_select == "Summer":
            season_select = "Summer     "

        else:
            season_select = "Winter     "

        crop_selection = st.sidebar.selectbox(
            "Select Your Crop",
            (
                "Arecanut",
                "Other Kharif pulses",
                "Rice",
                "Banana",
                "Cashewnut",
                "Coconut ",
                "Dry ginger",
                "Sugarcane",
                "Sweet potato",
                "Tapioca",
                "Black pepper",
                "Dry chillies",
                "other oilseeds",
                "Turmeric",
                "Maize",
                "Moong(Green Gram)",
                "Urad",
                "Arhar/Tur",
                "Groundnut",
                "Sunflower",
                "Bajra",
                "Castor seed",
                "Cotton(lint)",
                "Horse-gram",
                "Jowar",
                "Korra",
                "Ragi",
                "Tobacco",
                "Gram",
                "Wheat",
                "Masoor",
                "Sesamum",
                "Linseed",
                "Safflower",
                "Onion",
                "other misc. pulses",
                "Samai",
                "Small millets",
                "Coriander",
                "Potato",
                "Other  Rabi pulses",
                "Soyabean",
                "Beans & Mutter(Vegetable)",
                "Bhindi",
                "Brinjal",
                "Citrus Fruit",
                "Cucumber",
                "Grapes",
                "Mango",
                "Orange",
                "other fibres",
                "Other Fresh Fruits",
                "Other Vegetables",
                "Papaya",
                "Pome Fruit",
                "Tomato",
                "Rapeseed &Mustard",
                "Mesta",
                "Cowpea(Lobia)",
                "Lemon",
                "Pome Granet",
                "Sapota",
                "Cabbage",
                "Peas  (vegetable)",
                "Niger seed",
                "Bottle Gourd",
                "Sannhamp",
                "Varagu",
                "Garlic",
                "Ginger",
                "Oilseeds total",
                "Pulses total",
                "Jute",
                "Peas & beans (Pulses)",
                "Blackgram",
                "Paddy",
                "Pineapple",
                "Barley",
                "Khesari",
                "Guar seed",
                "Moth",
                "Other Cereals & Millets",
                "Cond-spcs other",
                "Turnip",
                "Carrot",
                "Redish",
                "Arcanut (Processed)",
                "Atcanut (Raw)",
                "Cashewnut Processed",
                "Cashewnut Raw",
                "Cardamom",
                "Rubber",
                "Bitter Gourd",
                "Drum Stick",
                "Jack Fruit",
                "Snak Guard",
                "Pump Kin",
                "Tea",
                "Coffee",
                "Cauliflower",
                "Other Citrus Fruit",
                "Water Melon",
                "Total foodgrain",
                "Kapas",
                "Colocosia",
                "Lentil",
                "Bean",
                "Jobster",
                "Perilla",
                "Rajmash Kholar",
                "Ricebean (nagadal)",
                "Ash Gourd",
                "Beet Root",
                "Lab-Lab",
                "Ribed Guard",
                "Yam",
                "Apple",
                "Peach",
                "Pear",
                "Plums",
                "Litchi",
                "Ber",
                "Other Dry Fruit",
                "Jute & mesta",
            ),
        )

        area_selection = st.sidebar.text_input(
            "Enter the Area of Your Field (in square meters)"
        )

        result = ""
        if st.button("Predict"):
            result = yield_prediction(
                [
                    state_selection,
                    district_selection,
                    crop_year,
                    season_select,
                    crop_selection,
                    area_selection,
                ]
            )

        st.success(
            "The estimated Crop Production (Kg per hectare) is {}".format(
                result)
        )

    hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown(
        """<style>footer {visibility: hidden;} footer:after {content:'Made with 💖 by Aryan and Khushi';visibility: visible;display: block;position: relative;#background-color: red;padding: 5px; top: 2px;}</style>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
