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
    expected_columns = ["date", "time", "Humi", "Temp", "EC", "PH", "N", "P", "K"]
    
    # Check if the columns match
    return columns == expected_columns

# Check file size
def is_valid_size(file):
    # Check if the file size is within the limit
    file.seek(0, os.SEEK_END)  # Move pointer to the end of the file
    file_size = file.tell()  # Get current position (gives the size in bytes)
    file.seek(0)  # Reset file pointer to the start
    return file_size <= MAX_FILE_SIZE, file_size

# File Uploader op up form
def show_file_uploader():
    file = st.file_uploader("Only CSV accepted", type=["csv"], accept_multiple_files=False)
    
    # Ensure file uploaded follows the file format
    st.write("\n")
    st.caption("Ensure file uploaded follows the file format:")
    df = pd.read_csv('backend/out_sensor.csv')
    st.write(df.head())
    st.caption("*Download to use as template")
    
    return file

# -- HOME PAGE --
st.title("Home")
st.write("\n")
st.subheader("Lorem ipsum dolor sit amet")
st.write("Consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
st.write("\n")

# Show file uploader
uploaded_file = show_file_uploader()

# Only show the submit button if a file is uploaded
if uploaded_file is not None:
    if st.button("Submit"):
        # Validate file size
        size_valid, file_size = is_valid_size(uploaded_file)
        if size_valid:
            # Validate file content
            if is_valid(uploaded_file):
                st.success("File successfully uploaded! Proceed to dashboard to get an overview.")
                uploaded_file.seek(0)  # Reset file pointer to the start
                df = pd.read_csv(uploaded_file, header=0)  # Pass this df to dashboard
                st.session_state['uploaded_df'] = df
                
            else:
                st.error("File does not meet format requirements.")  # Should not accept
        else:
            st.error(f"File size exceeded 200MB. Current size: {file_size / (1024 * 1024):.2f} MB.")
