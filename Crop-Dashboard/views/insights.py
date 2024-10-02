# import streamlit as st
# import pickle
# import joblib
# import pandas as pd

# # --FUNCTIONS --
# # Load the model from the .pkl file
# with open('backend/RandomForest.pkl', 'rb') as f:
#     rec_model = pickle.load(f)

# # Crop recommendation system form
# @st.dialog("Get Crop Recommendation")
# def crop_recommendation_sys():
#     file = st.file_uploader("Upload dataset", type=["csv"], accept_multiple_files=False)
#     st.caption("or")
#     with st.form("rec_form"):
#         n = st.number_input('Nitrogen (mg/L)')
#         p = st.number_input('Phosphorus (mg/L)')
#         k = st.number_input('Potassium (mg/L)')
#         temp = st.number_input('Temperature (°C)')
#         humi = st.number_input('Humidity (%)')
#         ph = st.number_input('PH (3-9)')
#         submit = st.form_submit_button('Submit Parameters')
    
#     # !! check valid
#     if file is not None:
#         prediction="insert smtg bro" #!!read file get avg
#         st.success(prediction)

#     # !!! change output display
#     if submit:
#         x = [[n, p, k, temp, humi, ph]]  # Replace this with actual data
#         prediction = rec_model.predict(x)
#         st.success(prediction)

# yield_model = joblib.load('backend/best_model.pkl')
# scaler = joblib.load('backend/scaler.pkl')

# unique_crops = ['Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Castor seed', 'Coriander', 
#                     'Cotton(lint)', 'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 
#                     'Gram', 'Groundnut', 'Guar seed', 'Jowar', 'Jute', 'Linseed', 'Maize', 
#                     'Masoor', 'Moong(Green Gram)', 'Moth', 'Oilseeds total', 'Onion', 
#                     'Other  Rabi pulses', 'Other Kharif pulses', 'Peas & beans (Pulses)', 
#                     'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Sannhamp', 'Sesamum', 
#                     'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 
#                     'Tobacco', 'Total foodgrain', 'Turmeric', 'Urad', 'Wheat']

# # Function to preprocess the data to include crop name columns
# def preprocess_data(df, crop_name, unique_crops):
#     for crop in unique_crops:
#         if f'{crop}' not in df.columns:
#             df[f'{crop}'] = 0
    
#     # Set the relevant crop column to 1
#     df[f'{crop_name}'] = 1
#     return df 



# # YIELD PREDICTION
# @st.dialog("Get Crop Yield Prediction")
# def crop_yield_prediction():
#     st.title("Crop Yield Prediction System")

#     file = st.file_uploader("Upload dataset (CSV)", type=["csv"], accept_multiple_files=False)
#     st.caption("or")
#     with st.form("rec_form"):
#         crop_name = st.selectbox("Select Crop", unique_crops)  # Let user select crop
#         n = st.number_input('Nitrogen (mg/L)')
#         p = st.number_input('Phosphorus (mg/L)')
#         k = st.number_input('Potassium (mg/L)')
#         temp = st.number_input('Temperature (°C)')
#         humi = st.number_input('Humidity (%)')
#         submit = st.form_submit_button('Submit Parameters')

#     if file is not None:
#         uploaded_df = pd.read_csv(file)
#         st.write("Processing uploaded dataset...")

#         # Preprocess the data
#         uploaded_df = preprocess_data(uploaded_df, crop_name, unique_crops)

#         features_order = ['temperature', 'humidity', 'N', 'P', 'K', 'Arhar/Tur', 'Bajra',
#                           'Banana', 'Barley', 'Castor seed', 'Coriander', 'Cotton(lint)',
#                           'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 'Gram', 'Groundnut',
#                           'Guar seed', 'Jowar', 'Jute', 'Linseed', 'Maize', 'Masoor',
#                           'Moong(Green Gram)', 'Moth', 'Oilseeds total', 'Onion',
#                           'Other  Rabi pulses', 'Other Kharif pulses', 'Peas & beans (Pulses)',
#                           'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Sannhamp', 'Sesamum',
#                           'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato',
#                           'Tobacco', 'Total foodgrain', 'Turmeric', 'Urad', 'Wheat']

#         uploaded_df = uploaded_df[features_order]

#         # Scale the data
#         uploaded_df_scaled = scaler.transform(uploaded_df)

#         # Make the prediction
#         predictions = yield_model.predict(uploaded_df_scaled)

#         uploaded_df['Predicted_Yield'] = predictions
#         st.write(uploaded_df[['temperature', 'humidity', 'N', 'P', 'K', 'Predicted_Yield']])

#     if submit:
#         input_data = pd.DataFrame([[temp, humi, n, p, k]], columns=['temperature', 'humidity', 'N', 'P', 'K'])
#         input_data = preprocess_data(input_data, crop_name, unique_crops)

#         features_order = ['temperature', 'humidity', 'N', 'P', 'K', 'Arhar/Tur', 'Bajra',
#                           'Banana', 'Barley', 'Castor seed', 'Coriander', 'Cotton(lint)',
#                           'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 'Gram', 'Groundnut',
#                           'Guar seed', 'Jowar', 'Jute', 'Linseed', 'Maize', 'Masoor',
#                           'Moong(Green Gram)', 'Moth', 'Oilseeds total', 'Onion',
#                           'Other  Rabi pulses', 'Other Kharif pulses', 'Peas & beans (Pulses)',
#                           'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Sannhamp', 'Sesamum',
#                           'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato',
#                           'Tobacco', 'Total foodgrain', 'Turmeric', 'Urad', 'Wheat']
#         input_data = input_data[features_order]

#         # Scale the data
#         input_data_scaled = scaler.transform(input_data)

#         # Make the prediction
#         prediction = yield_model.predict(input_data_scaled)

#         st.success(f"Predicted Yield: {prediction[0]}")

# # -- INSIGHTS PAGE --

# st.title("Crop Insights")
# st.write("Get insights on your soil data now using the following features")

# col1, col2, col3 = st.columns(3, gap='medium')
# with col1: 
#     st.image('assets/rec.jpg')
#     st.subheader("Crop Recommender")
#     st.divider()
#     st.write("Discover the best crops to grow based on your soil data. Our machine learning algorithm will analyze your specific conditions to provide optimal crop recommendations.")
#     if st.button("Get Reccommedation"):
#         crop_recommendation_sys()

# with col2: 
#     st.image('assets/yield.png')
#     st.subheader("Crop Yield Predictor")
#     st.divider()
#     st.write("Accurately forecast your crop yields with machine learning that leverages historical. Optimize your planning and maximize productivity.")
#     if st.button("Calculate Yield"):
#         pass

# with col3: 
#     st.image('assets/info.png')
#     st.subheader("Smart Crop Chatbot?")
#     st.divider()
#     st.write("Get instant answers to your crop-related questions and personalized farming advice. Powered by AI, this chatbot provides insights based on your specific needs.")
#     st.link_button(label="Chat Now", url="/chatbot") ## !! fix transistion cacat bozo

import streamlit as st
import pickle
import pandas as pd
import joblib

# --FUNCTIONS --
# Load the model from the .pkl file
with open('backend/RandomForest.pkl', 'rb') as f:
    rec_model = pickle.load(f)

# Crop recommendation system form
@st.dialog("Get Crop Recommendation")
def crop_recommendation_sys():
    file = st.file_uploader("Upload dataset", type=["csv"], accept_multiple_files=False)
    st.caption("or")
    with st.form("rec_form"):
        n = st.number_input('Nitrogen (mg/L)')
        p = st.number_input('Phosphorus (mg/L)')
        k = st.number_input('Potassium (mg/L)')
        temp = st.number_input('Temperature (°C)')
        humi = st.number_input('Humidity (%)')
        ph = st.number_input('PH (3-9)')
        submit = st.form_submit_button('Submit Parameters')
    
    # !! check valid
    if file is not None:
        prediction="insert smtg bro" #!!read file get avg
        st.success(prediction)

    # !!! change output display
    if submit:
        x = [[n, p, k, temp, humi, ph]]  # Replace this with actual data
        prediction = rec_model.predict(x)
        st.success(prediction)

# with open('backend/best_model.pkl', 'rb') as f:
#     yield_model = pickle.load(f)
yield_model = joblib.load('backend/best_model.pkl')
scaler = joblib.load('backend/scaler.pkl')

unique_crops = ['Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Castor seed', 'Coriander', 
                    'Cotton(lint)', 'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 
                    'Gram', 'Groundnut', 'Guar seed', 'Jowar', 'Jute', 'Linseed', 'Maize', 
                    'Masoor', 'Moong(Green Gram)', 'Moth', 'Oilseeds total', 'Onion', 
                    'Other  Rabi pulses', 'Other Kharif pulses', 'Peas & beans (Pulses)', 
                    'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Sannhamp', 'Sesamum', 
                    'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 
                    'Tobacco', 'Total foodgrain', 'Turmeric', 'Urad', 'Wheat']

# Function to preprocess the data to include crop name columns
def preprocess_data(df, crop_name, unique_crops):
    for crop in unique_crops:
        if f'{crop}' not in df.columns:
            df[f'{crop}'] = 0
    
    # Set the relevant crop column to 1
    df[f'{crop_name}'] = 1
    return df 

# YIELD PREDICTION
@st.dialog("Get Crop Yield Prediction")
def crop_yield_prediction():
    st.title("Crop Yield Prediction System")

    file = st.file_uploader("Upload dataset (CSV)", type=["csv"], accept_multiple_files=False)
    st.caption("or")
    with st.form("rec_form"):
        crop_name = st.selectbox("Select Crop", unique_crops)  # Let user select crop
        n = st.number_input('Nitrogen (mg/L)')
        p = st.number_input('Phosphorus (mg/L)')
        k = st.number_input('Potassium (mg/L)')
        temp = st.number_input('Temperature (°C)')
        humi = st.number_input('Humidity (%)')
        submit = st.form_submit_button('Submit Parameters')

    if file is not None:
        uploaded_df = pd.read_csv(file)
        st.write("Processing uploaded dataset...")

        # Preprocess the data
        uploaded_df = preprocess_data(uploaded_df, crop_name, unique_crops)

        features_order = ['temperature', 'humidity', 'N', 'P', 'K', 'Arhar/Tur', 'Bajra',
                          'Banana', 'Barley', 'Castor seed', 'Coriander', 'Cotton(lint)',
                          'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 'Gram', 'Groundnut',
                          'Guar seed', 'Jowar', 'Jute', 'Linseed', 'Maize', 'Masoor',
                          'Moong(Green Gram)', 'Moth', 'Oilseeds total', 'Onion',
                          'Other  Rabi pulses', 'Other Kharif pulses', 'Peas & beans (Pulses)',
                          'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Sannhamp', 'Sesamum',
                          'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato',
                          'Tobacco', 'Total foodgrain', 'Turmeric', 'Urad', 'Wheat']

        uploaded_df = uploaded_df[features_order]

        # Scale the data
        uploaded_df_scaled = scaler.transform(uploaded_df)

        # Make the prediction
        predictions = yield_model.predict(uploaded_df_scaled)

        uploaded_df['Predicted_Yield'] = predictions
        st.write(uploaded_df[['temperature', 'humidity', 'N', 'P', 'K', 'Predicted_Yield']])

    if submit:
        input_data = pd.DataFrame([[temp, humi, n, p, k]], columns=['temperature', 'humidity', 'N', 'P', 'K'])
        input_data = preprocess_data(input_data, crop_name, unique_crops)

        features_order = ['temperature', 'humidity', 'N', 'P', 'K', 'Arhar/Tur', 'Bajra',
                          'Banana', 'Barley', 'Castor seed', 'Coriander', 'Cotton(lint)',
                          'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 'Gram', 'Groundnut',
                          'Guar seed', 'Jowar', 'Jute', 'Linseed', 'Maize', 'Masoor',
                          'Moong(Green Gram)', 'Moth', 'Oilseeds total', 'Onion',
                          'Other  Rabi pulses', 'Other Kharif pulses', 'Peas & beans (Pulses)',
                          'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Sannhamp', 'Sesamum',
                          'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato',
                          'Tobacco', 'Total foodgrain', 'Turmeric', 'Urad', 'Wheat']
        input_data = input_data[features_order]

        # Scale the data
        input_data_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = yield_model.predict(input_data_scaled)

        st.success(f"Predicted Yield: {prediction[0]}")

# -- INSIGHTS PAGE --
st.title("Crop Insights")
st.write("Get insights on your soil data now using the following features")

col1, col2, col3 = st.columns(3)
with col1: 
    st.image('assets/rec.jpg')
    st.subheader("Crop Recommendation Tool")
    st.divider()
    st.write("Discover the best crops to grow based on your soil data. Our machine learning algorithm will analyze your specific conditions to provide optimal crop recommendations.")
    if st.button("Get Recommendation"):
        crop_recommendation_sys()

with col2: 
    st.image('assets/yield.png')
    st.subheader("Crop Yield Predictor")
    st.divider()
    st.write("Accurately forecast your crop yields with machine learning that leverages historical datasets. Optimize your planning and maximize productivity.")
    if st.button("Calculate Yield"):
        crop_yield_prediction()

with col3: 
    st.image('assets/info.png')
    st.subheader("Smart Crop Chatbot?")
    st.divider()
    st.write("Get instant answers to your crop-related questions and personalized farming advice. Powered by AI, this chatbot provides insights based on your specific needs.")
    st.link_button(label="Chat Now", url="/chatbot") ## !! fix transistion cacat bozo
