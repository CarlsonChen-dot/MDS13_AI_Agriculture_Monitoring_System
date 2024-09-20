import streamlit as st # type: ignore
import pandas as pd # type: ignore
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
    expected_columns = ["date", "time", "Humi", "Temp", "EC", "PH", "N", "P", "K"]
    
    # Check if the columns match
    if columns != expected_columns:
        # return "Error: Column format is incorrect. Expected columns: " + str(expected_columns) + " but " + str(columns) + " instead."
        return False

    return True

#Check file size
def is_valid_size (file):
    # Check if the file size is within the limit
    file.seek(0, os.SEEK_END)  # Move pointer to the end of the file
    file_size = file.tell()  # Get current position (gives the size in bytes)
    file.seek(0)  # Reset file pointer to the start
    if file_size> MAX_FILE_SIZE:
        return False, file_size
    return True

# File Uploader op up form
@st.dialog("Upload your soil data")
def show_file_uploader():
    file = st.file_uploader("Only CSV accepted", type=["csv"], accept_multiple_files=False)
    st.write("\n")
    st.caption("Ensure file uploaded follows the file following format")
    df = pd.read_csv('backend/out_sensor.csv')
    st.write(df.head())
    st.caption("*download to use as template")
    
    if file is not None:
        if is_valid_size(file):
            if is_valid(file):
                st.success("File successfully uploaded! Proceed to dashboard to get an overview")
                df = pd.read_csv(file) # !!! pass this df to dashboard
            else:
                st.error("File does not meet format requirements") # !!! should not accept

        else:
            st.error("File size exceeded 200MB.")

    

# -- HOME PAGE --
st.title("Home")
st.write("\n")
st.subheader("Lorem ipsum dolor sit amet")
st.write("Consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
st.write("\n")
if st.button("Import Soil Data"):
    show_file_uploader()






