import streamlit as st
import pandas as pd

# -- FUNCTIONS --
# Check file validation 
def is_valid(file):
    #check
    return False

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
        if is_valid(file):
            st.success("File successfully uploaded! Proceed to dashboard to get an overview")
            df = pd.read_csv(file) # !!! pass this df to dashboard
        else:
            st.error("File does not meet format requirements") # !!! should not accept
        

# -- HOME PAGE --
st.title("Home")
st.write("\n")
st.subheader("Lorem ipsum dolor sit amet")
st.write("Consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
st.write("\n")
if st.button("Import Soil Data"):
    show_file_uploader()






