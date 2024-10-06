
# import streamlit as st
# import pandas as pd

# # -- FUNCTIONS --
# # Check file validation 
# def is_valid(file):
#     #check
#     return False

# # File Uploader pop up form
# @st.dialog("Upload your soil data")
# def show_file_uploader():
#     file = st.file_uploader("Only CSV accepted", type=["csv"], accept_multiple_files=False)
#     st.write("\n")
#     st.caption("Ensure file uploaded follows the file following format")
#     df = pd.read_csv('backend/out_sensor.csv')
#     st.write(df.head())
#     st.caption("*download to use as template")
    
#     if file is not None:
#         if is_valid(file):
#             st.success("File successfully uploaded! Proceed to dashboard to get an overview")
#             df = pd.read_csv(file) # !!! pass this df to dashboard
#         else:
#             st.error("File does not meet format requirements") # !!! should not accept
        

# # -- HOME PAGE --
# st.title("Welcome to AgriMonitor 🌱")
# st.write("\n")
# st.subheader("Lorem ipsum dolor sit amet")
# st.write("Consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
# st.write("\n")
# if st.button("Import Soil Data"):
#     show_file_uploader()


import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import os

# Maximum file size in bytes (200MB = 200 * 1024 * 1024 bytes)
MAX_FILE_SIZE = 200 * 1024 * 1024

# -- FUNCTIONS --
# Check file validation 
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

# File Uploader op up form
def show_file_uploader():
    csv_file_path = os.path.join(os.path.dirname(__file__), '../backend/out_sensor.csv')
    file = st.file_uploader("Only CSV accepted", type=["csv"], accept_multiple_files=False)
    
    # Ensure file uploaded follows the file format
    st.write("\n")
    st.caption("Ensure file uploaded follows the file format:")
    df = pd.read_csv(csv_file_path)
    st.write(df.head())
    st.caption("*Download to use as template")
    
    return file

# Function to rename columns
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

@st.dialog("Upload your soil data")
def show_window():
    # Show file uploader
    uploaded_file = show_file_uploader()

    # Only show the submit button if a file is uploaded
    if uploaded_file is not None:
        if st.button("Submit"):
            # Validate file size
            size_valid, file_size = is_valid_size(uploaded_file)
            if size_valid:
                status, msg = is_valid(uploaded_file)
                # Validate file content
                if status == True:
                    st.success("File successfully uploaded! Proceed to dashboard to get an overview.")
                    uploaded_file.seek(0)  # Reset file pointer to the start

                    df = pd.read_csv(uploaded_file, header=0)  # Pass this df to dashboard
                    # Rename the columns
                    df = rename_columns(df)
                    df = transform_data(df)

                    st.session_state['uploaded_df'] = df

                     # View the DataFrame
                    st.subheader("DataFrame Overview")
                    st.dataframe(df)  # Display the DataFrame
                    
                else:
                    st.error(msg)  # Should not accept
            else:
                st.error(f"File size exceeded 200MB. Current size: {file_size / (1024 * 1024):.2f} MB.")


# -- HOME PAGE --

st.title("Welcome to AgriMonitor 🌱")
st.write("\n")
st.write("""
AgriMonitor is a cutting-edge web application designed by MDS13 to revolutionize agricultural monitoring and management. Our platform leverages advanced technologies to provide insightful analytics, and precise recommendations that empower farmers and agronomists to make informed decisions and maximize crop yield.
         """)
st.write("\n")


# Features Overview
st.subheader("Key Features")
st.write("""
- **File Upload & Validation**: Upload data in CSV format and ensure the file is validated.
- **Data Analysis Tools**: Perform statistical analysis, anomaly detection, and visualize trends in the data.
- **Comprehensive Reporting**: Summarize and extract best-value metrics from sensor readings.
- **Customizable Views**: Choose specific columns or full datasets to focus your analysis on relevant environmental factors.
""")
st.write("\n")

st.subheader("Get Started")
st.write("""
To begin, upload your sensor dataset. Once your file is validated, explore the various analysis options under the 'Dashboard' section.
""")

# Button to open the file upload dialog
if st.button("Upload Data"):
    show_window()

# # Features Overview
# st.subheader("Key Features")
# st.write("""
# - **File Upload & Validation**: Upload data in CSV format and ensure the file is validated.
# - **Data Analysis Tools**: Perform statistical analysis, anomaly detection, and visualize trends in the data.
# - **Comprehensive Reporting**: Summarize and extract best-value metrics from sensor readings.
# - **Customizable Views**: Choose specific columns or full datasets to focus your analysis on relevant environmental factors.
# """)
# st.write("\n")

# st.subheader("Get Started")
# # st.markdown(
# #     """
# #     <h1 style='color:#81c784;'>Get Started</h1>
# #     """,
# #     unsafe_allow_html=True
# # )

# st.write("""
# To begin, upload your sensor dataset. Once your file is validated, explore the various analysis options under the 'Dashboard' section.
# """)


# # Button to open the file upload dialog
# if st.button("Upload Data"):
#     show_window()
