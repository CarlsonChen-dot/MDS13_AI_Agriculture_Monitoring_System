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


import os
import streamlit as st
import pickle
import pandas as pd
import joblib


# Maximum file size in bytes (200MB = 200 * 1024 * 1024 bytes)
MAX_FILE_SIZE = 200 * 1024 * 1024

# with open('backend/best_model.pkl', 'rb') as f:
#     yield_model = pickle.load(f)
model_file_path = os.path.join(os.path.dirname(__file__), '../backend/best_model.pkl')
scaler_file_path = os.path.join(os.path.dirname(__file__), '../backend/scaler.pkl')
yield_model = joblib.load(model_file_path)
scaler = joblib.load(scaler_file_path)

unique_crops = ['Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Castor seed', 'Coriander', 
                    'Cotton(lint)', 'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 
                    'Gram', 'Groundnut', 'Guar seed', 'Jowar', 'Jute', 'Linseed', 'Maize', 
                    'Masoor', 'Moong(Green Gram)', 'Moth', 'Oilseeds total', 'Onion', 
                    'Other  Rabi pulses', 'Other Kharif pulses', 'Peas & beans (Pulses)', 
                    'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Sannhamp', 'Sesamum', 
                    'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 
                    'Tobacco', 'Total foodgrain', 'Turmeric', 'Urad', 'Wheat']



# --FUNCTIONS --
# Load the model from the .pkl file
with open('backend/RandomForest.pkl', 'rb') as f:
    rec_model = pickle.load(f)

def is_valid(file):
    # Open the file and check the headers
    file.seek(0)  # Reset the file pointer after upload
    first_line = file.readline().decode('utf-8').strip()
    columns = first_line.split(',')
        
    # Expected columns
    expected_columns = ["date", "time", "Humi" , "Temp", "EC", "PH", "N", "P", "K"]

   # Check if the columns match
    for col in expected_columns:
        if col not in columns:
            return False, "Column headers do not match."
    
#    # Check if the columns match
#     if columns != expected_columns:
#         return False, "Column headers do not match."
    
    # Define expected data types for each column (example: float for sensor readings, str for date/time)
    expected_dtypes = {
        "date": "object",   # str for date, could also use datetime if needed
        "time": "object",   # str for time, could also use datetime if needed
        "Humi": "float64",  # float for humidity
        "Temp": "float64",  # float for temperature
        "EC": "int64",    # float for EC
        "PH": "float64",    # float for pH
        "N": "int64",     # float for nitrogen
        "P": "int64",     # float for phosphorus
        "K": "int64"      # float for potassium
    }

    # Read the rest of the file as a DataFrame
    file.seek(0)  # Reset the file pointer to the beginning again
    df = pd.read_csv(file)

    # Validate column data types
    actual_dtypes = df.dtypes.to_dict()

    for col, expected_dtype in expected_dtypes.items():
        if actual_dtypes[col] != expected_dtype:
            return False, f"Column '{col}' has incorrect data type."
    
     # Validate date and time formats
    date_format = "%d/%m/%Y"  # Month/Day/Year
    time_format = "%H:%M:%S"  # Hour:Minute:Second

    for index, row in df.iterrows():
            # Validate date format for the current row
            try:
                pd.to_datetime(row['date'], format=date_format, errors='raise')
            except ValueError:
                return False, f"Row {index + 1}: Date format is incorrect. Expected format: DD/MM/YYYY."

            # Validate time format for the current row
            try:
                pd.to_datetime(row['time'], format=time_format, errors='raise')
            except ValueError:
                return False, f"Row {index + 1}: Time format is incorrect. Expected format: HH:MM:SS."

    return True, "File is valid."


# Check file size
def is_valid_size(file):
    # Check if the file size is within the limit
    file.seek(0, os.SEEK_END)  # Move pointer to the end of the file
    file_size = file.tell()  # Get current position (gives the size in bytes)
    file.seek(0)  # Reset file pointer to the start
    return file_size <= MAX_FILE_SIZE, file_size

# Function to calculate averages of required columns
def calculate_averages(df):
    return df[['N', 'P', 'K', 'Temperature', 'Humidity', 'PH']].mean()

def rename_columns(df):
    # Define the mapping from old column names to new column names
    new_column_names = {
        "Humi": "Humidity",
        "Temp": "Temperature",
        "date": "Date",
        "time": "Time"
    }
    
    # Rename the columns
    df.rename(columns=new_column_names, inplace=True)
    
    return df

def transform_data(df):
    """ Formats the data to create visualizations """
    df['datetime'] = df['Date'] + ' ' + df['Time']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')
    return df

# Crop recommendation system form
@st.dialog("Get Crop Recommendation")
def crop_recommendation_sys():
    file = st.file_uploader("Upload dataset", type=["csv"], accept_multiple_files=False)
     # !! check valid
    if file is not None:
        # prediction="insert smtg bro" #!!read file get avg
        # st.success(prediction)
        if st.button("Submit"):
            # Validate file size
            size_valid, file_size = is_valid_size(file)
            if size_valid:
                status, msg = is_valid(file)
                # Validate file content
                if status == True:
                    file.seek(0)
                    df = pd.read_csv(file)
                    # Rename the columns
                    df = rename_columns(df)
                    df = transform_data(df)

                    avg_values = calculate_averages(df)
                    x = [avg_values]
                    prediction = rec_model.predict(x)
                    st.success(f"File sucessfully uploaded. Recommended Crop: {prediction[0]}")

                else:
                    st.error(msg)  # Should not accept
            else:
                st.error(f"File size exceeded 200MB. Current size: {file_size / (1024 * 1024):.2f} MB.")

    st.caption("or")

     # Flag to track if inputs are valid
    valid_inputs = True

    with st.form("rec_form"):
        n = st.number_input('Nitrogen (mg/L)', min_value=0.0)
        p = st.number_input('Phosphorus (mg/L)', min_value=0.0)
        k = st.number_input('Potassium (mg/L)', min_value=0.0)
        temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=100.0, help="Temperature cannot be 0°C")
        humi = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, help="Humidity cannot be 0%")
        ph = st.number_input('PH (3-9)', min_value=3.0, max_value=9.0, help="PH less than 3")
        submit = st.form_submit_button('Submit Parameters')

    # Check if the inputs are valid only after clicking submit
    if submit:
        valid_inputs = True  # Flag to track the validity of inputs
        
        if temp <= 0:
            st.warning("Temperature must be greater than 0°C.")
            valid_inputs = False
        if humi <= 0:
            st.warning("Humidity must be greater than 0%.")
            valid_inputs = False
        if ph <= 0:
            st.warning("PH must be greater than 0.")
            valid_inputs = False

     # !!! change output display
    if submit and valid_inputs:
       # Processing only happens if the inputs are valid
        x = [[n, p, k, temp, humi, ph]]  # Input data for prediction
        prediction = rec_model.predict(x)
        st.success(f"Recommended Crop: {prediction[0]}")
    elif submit and not valid_inputs:
        st.error("Please fix the input values before submitting.")

    # # !!! change output display
    # if submit and valid:
    #     x = [[n, p, k, temp, humi, ph]]  # Replace this with actual data
    #     prediction = rec_model.predict(x)
    #     st.success(prediction)

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


#######################################################

# YIELD PREDICTION
@st.dialog("Get Crop Yield Prediction")
def crop_yield_prediction():
    st.title("Crop Yield Prediction System")

    crop_name = st.selectbox("Select Crop", unique_crops)  # Let user select crop

    st.caption("then")

    file = st.file_uploader("Upload dataset (CSV)", type=["csv"], accept_multiple_files=False)
    if file is not None:
         if st.button("Submit"):
            # Validate file size
            size_valid, file_size = is_valid_size(file)
            if size_valid:
                    file.seek(0)
                    uploaded_df = pd.read_csv(file)
                    uploaded_df = rename_columns(uploaded_df)

                    st.write("Processing uploaded dataset...")

                    # Preprocess the data
                    uploaded_df = preprocess_data(uploaded_df, crop_name, unique_crops)

                    features_order = ['Temperature', 'Humidity', 'N', 'P', 'K', 'Arhar/Tur', 'Bajra',
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
                    st.write(uploaded_df[['Temperature', 'Humidity', 'N', 'P', 'K', 'Predicted_Yield']])

            elif size_valid is False:
                    st.error(f"File size exceeded 200MB. Current size: {file_size / (1024 * 1024):.2f} MB.")

    st.caption("or")
    with st.form("rec_form"):
        n = st.number_input('Nitrogen (mg/L)', min_value=0.0)
        p = st.number_input('Phosphorus (mg/L)', min_value=0.0)
        k = st.number_input('Potassium (mg/L)', min_value=0.0)
        temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=100.0, help="Temperature cannot be 0°C")
        humi = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, help="Humidity cannot be 0%")
        submit = st.form_submit_button('Submit Parameters')

    # Check if the inputs are valid only after clicking submit
    if submit:
        valid_inputs = True  # Flag to track the validity of inputs
        
        if temp <= 0:
            st.warning("Temperature must be greater than 0°C.")
            valid_inputs = False
        if humi <= 0:
            st.warning("Humidity must be greater than 0%.")
            valid_inputs = False

    if submit and valid_inputs:
        input_data = pd.DataFrame([[temp, humi, n, p, k]], columns=['Temperature', 'Humidity', 'N', 'P', 'K'])
        input_data = preprocess_data(input_data, crop_name, unique_crops)

        features_order = ['Temperature', 'Humidity', 'N', 'P', 'K', 'Arhar/Tur', 'Bajra',
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

        st.success(f"Predicted Yield: {prediction[0]:.2f} kg/ha")
        
    elif submit and not valid_inputs:
        st.error("Please fix the input values before submitting.")

# Function to preprocess the data to include crop name columns
def preprocess_data(df, crop_name, unique_crops):
    for crop in unique_crops:
        if f'{crop}' not in df.columns:
            df[f'{crop}'] = 0
    
    # Set the relevant crop column to 1
    df[f'{crop_name}'] = 1
    return df 

# -- INSIGHTS PAGE --
st.title("Crop Insights")
st.write("Get insights on your soil data now using the following features")

col1, col2, col3 = st.columns(3)
with col1: 
    st.image('assets/rec.jpg')
    st.subheader("Crop Recommender")
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
