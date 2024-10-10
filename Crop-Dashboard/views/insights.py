import os
import streamlit as st
import pickle
import pandas as pd
import joblib



# Maximum file size in bytes (200MB = 200 * 1024 * 1024 bytes)
MAX_FILE_SIZE = 200 * 1024 * 1024

# with open('backend/best_model.pkl', 'rb') as f:
#     yield_model = pickle.load(f)
bestmodel_file_path = os.path.join(os.path.dirname(__file__), '../backend/best_model.pkl')
scaler_file_path = os.path.join(os.path.dirname(__file__), '../backend/scaler.pkl')
rfmodel_file_path = os.path.join(os.path.dirname(__file__), '../backend/RandomForest.pkl')
ideal_ranges_file_path = os.path.join(os.path.dirname(__file__), '../backend/ideal_ranges.csv')
yield_model = joblib.load(bestmodel_file_path)
scaler = joblib.load(scaler_file_path)

# --FUNCTIONS --
# Load the model from the .pkl file
with open(rfmodel_file_path, 'rb') as f:
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

    # Check if file is valid
    if file is not None:
        # prediction="insert smtg bro" #!!read file get avg
        # st.success(prediction)
        if st.button("Submit", key= crop_recommendation_sys):
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

    if submit and valid_inputs:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(ideal_ranges_file_path)

        # Processing only happens if the inputs are valid
        x = [[n, p, k, temp, humi, ph]]  # Input data for prediction
        prediction = rec_model.predict(x)
        prediction = prediction[0]

        # Extracting the ideal ranges for the recommended crop
        df = df[df['label']==prediction]
        n_range = f"{df['N'].iloc[0]} mg/L"
        p_range = f"{df['P'].iloc[0]} mg/L"
        k_range = f"{df['K'].iloc[0]} mg/L"
        temp_range = f"{df['temperature'].iloc[0]}°C"
        humi_range = f"{df['humidity'].iloc[0]}%"
        ph_range = f"{df['ph'].iloc[0]}"

        # Displaying the information in Streamlit with markdown
        st.write("Crop Recommendation:")
        st.success(
            f"""
            ### {prediction.capitalize()}
            
            This crop ideally grows in the following environmental ranges:
            - **Nitrogen (N)**: {n_range}
            - **Phosphorus (P)**: {p_range}
            - **Potassium (K)**: {k_range}
            - **Temperature**: {temp_range}
            - **Humidity**: {humi_range}
            - **Soil pH**: {ph_range}

            Maintaining these conditions ensures healthy crops with optimal yield!
            """
        )

    elif submit and not valid_inputs:
        st.error("Please fix the input values before submitting.")

###############################################################################################################

# YIELD PREDICTION
unique_crops = ['Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Castor seed', 'Coriander', 
                    'Cotton(lint)', 'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 
                    'Gram', 'Groundnut', 'Guar seed', 'Jowar', 'Jute', 'Linseed', 'Maize', 
                    'Masoor', 'Moong(Green Gram)', 'Moth', 'Oilseeds total', 'Onion', 
                    'Other  Rabi pulses', 'Other Kharif pulses', 'Peas & beans (Pulses)', 
                    'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Sannhamp', 'Sesamum', 
                    'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 
                    'Tobacco', 'Total foodgrain', 'Turmeric', 'Urad', 'Wheat']

crop_selection = ['Banana', 'Barley', 'Castor seed', 'Coriander', 
                    'Dry ginger', 'Garlic', 
                    'Groundnut', 'Guar seed', 'Linseed', 'Maize', 
                    'Moong(Green Gram)', 'Onion', 
                    'Peas & beans (Pulses)', 
                    'Potato', 'Rice', 'Sannhamp', 'Sesamum', 
                    'Soyabean', 'Sugarcane', 'Sweet potato', 
                    'Tobacco', 'Turmeric', 'Urad', 'Wheat']

# Function to preprocess the data to include crop name columns
def preprocess_data(df, crop_name, unique_crops):
    for crop in unique_crops:
        if f'{crop}' not in df.columns:
            df[f'{crop}'] = 0
    
    # Set the relevant crop column to 1
    df[f'{crop_name}'] = 1
    return df 

@st.dialog("Get Crop Yield Prediction")
def crop_yield_prediction():
    st.title("Crop Yield Prediction System")

    crop_name = st.selectbox("Select Crop", crop_selection)  # Let user select crop

    st.caption("then")

    file = st.file_uploader("Upload dataset (CSV)", type=["csv"], accept_multiple_files=False)
    if file is not None:
         if st.button("Submit"):
            # Validate file size
            size_valid, file_size = is_valid_size(file)
            if size_valid:
                status, msg = is_valid(file)
                # Validate file content
                if status == True:
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
                    st.success(f"Predicted Yield: {predictions[0]:.2f} kg/ha")
                
                else:
                    st.error(msg)  # Should not accept

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

##########################################anomalies##############################################

z_thresholds = {
                "Humidity": 4,  # A larger threshold for humidity
                "Temperature": 4,  # Default threshold for temperature
                "EC": 5,
                "PH": 5,
                "N": 6,
                "P": 6,
                "K":6
            }

@st.dialog("Detect Anomalies")
def detect_anomalies():
    st.title("Crop Anomalies Detection System")

    file = st.file_uploader("Upload dataset (CSV)", type=["csv"], accept_multiple_files=False)
    if file is not None:
         if st.button("Submit"):
            # Validate file size
            size_valid, file_size = is_valid_size(file)
            if size_valid:
                status, msg = is_valid(file)
                # Validate file content
                if status == True:
                    file.seek(0)
                    df = pd.read_csv(file)
                    df = rename_columns(df)
                    df = transform_data(df)

                    st.write("Anomalies:")

                    # Columns to analyze for anomalies
                    columns_to_check = ["Humidity", "Temperature", "EC", "PH", "N", "P", "K"]

                    # Create an empty list to store detected anomalies
                    anomalies = {'datetime': [], 'column': [], 'value': [], 'anomaly_type': []}

                    # Loop through each column and detect anomalies
                    for column in columns_to_check:
                        # Use the specific z_threshold for the column (fall back to a default if not specified)
                        z_threshold = z_thresholds.get(column, 3)

                        # Calculate the mean and standard deviation for the column
                        mean_value = df[column].mean()
                        std_dev = df[column].std()

                        # Calculate Z-scores for the column
                        z_scores = (df[column] - mean_value) / std_dev

                        # Detect outliers using Z-scores
                        outliers = abs(z_scores) > z_threshold

                        # Detect sudden changes by comparing with previous and next values
                        diff_prev = df[column].diff()  # Difference with previous value
                        diff_next = df[column].shift(-1) - df[column]  # Difference with next value

                        # Detect where BOTH previous and next changes exceed the mean value
                        changes_both = (abs(diff_prev) > mean_value) & (abs(diff_next) > mean_value)

                        # Combine outliers and sudden changes from both sides
                        anomaly_mask = outliers | changes_both

                        # Record the anomalies
                        for idx, is_anomaly in anomaly_mask.items():
                            if is_anomaly:
                                # Determine anomaly type
                                if outliers[idx]:
                                    anomaly_type = f"{column} anomaly (outlier)"
                                elif changes_both[idx]:
                                    if diff_prev[idx] > 0 and diff_next[idx] > 0:
                                        anomaly_type = f"{column} anomaly (increased from both previous and next)"
                                    elif diff_prev[idx] < 0 and diff_next[idx] < 0:
                                        anomaly_type = f"{column} anomaly (decreased from both previous and next)"
                                    else:
                                        anomaly_type = f"{column} anomaly (sudden change)"

                                # Append anomaly details to the dictionary
                                anomalies['datetime'].append(df['datetime'][idx])
                                anomalies['column'].append(column)
                                anomalies['value'].append(df[column][idx])
                                anomalies['anomaly_type'].append(anomaly_type)

                    # Convert anomalies dictionary to a DataFrame
                    anomalies_df = pd.DataFrame(anomalies)
                    if anomalies_df.empty:
                        st.success("No anomalies were detected.")
                    else:
                        st.write(anomalies_df)




# -- INSIGHTS PAGE --
st.title("Crop Insights")
st.write("Get insights on your soil data now using the following features")

image1_file_path = os.path.join(os.path.dirname(__file__), '../assets/rec.jpg')
image2_file_path = os.path.join(os.path.dirname(__file__), '../assets/yield.png')
image3_file_path = os.path.join(os.path.dirname(__file__), '../assets/info.png')

col1, col2, col3 = st.columns(3)
with col1: 
    st.image(image1_file_path)
    st.subheader("Crop Recommender")
    st.divider()
    st.write("Discover the best crops to grow based on your soil data. Our machine learning algorithm will analyze your specific conditions to provide optimal crop recommendations.")
    if st.button("Get Recommendation"):
        crop_recommendation_sys()

with col2: 
    st.image(image2_file_path)
    st.subheader("Crop Yield Predictor")
    st.divider()
    st.write("Accurately forecast your crop yields with machine learning model that leverages historical India datasets. Optimize your planning and maximize productivity.")
    if st.button("Calculate Yield"):
        crop_yield_prediction()

with col3: 
    st.image(image3_file_path)
    st.subheader("Anomalies Detection")
    st.divider()
    st.write("This function allows you to detect anomalies in environmental sensor data such as temperature, humidity, and more!")
    # st.link_button(label="Detect Anomalies", url="/chatbot") ## !! fix transistion cacat bozo
    if st.button("Detect Anomalies"):
        detect_anomalies()
