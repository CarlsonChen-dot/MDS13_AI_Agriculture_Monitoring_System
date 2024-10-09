import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import openai as openai


# openai.api_key = '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'

# List of specific columns user may request
specific_columns = ["N", "P", "K", "Humidity", "Date", "Temperature", "EC", "PH", "Time"]

z_thresholds = {
    "Humidity": 4,  # A larger threshold for humidity
    "Temperature": 4,  # Default threshold for temperature
    "EC": 5,
    "PH": 5,
    "N": 6,
    "P": 6,
    "K":6
}


st.session_state["page"] = "chatbot_page"
st.title("Crop Chatbot")

# Add hint questions for user guidance
st.write("### Ask me anything about your dataset!")
st.write("Here are some example questions you can ask:")
st.markdown("""
    <p style='color: grey;'>
        - What are the unique values in each column?<br>
        - How many missing values are in each column?<br>
        - What is the average Humidity?<br>
        - Can you summarize the dataset for me?
        * For column prompt, please use the exact column name.
    </p>
""", unsafe_allow_html=True)

# Check if the DataFrame exists in session_state
if 'uploaded_df' in st.session_state:
    df = st.session_state['uploaded_df']  # Get the uploaded DataFrame

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What do you want to know about your dataset?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    #     # Generate response based on the prompt
    #     response = get_openai_response(prompt)  # Get response from OpenAI API
    #     st.chat_message("assistant").markdown(response)  # Display response in chat message container
    #     # Add assistant response to chat history
        response = ""
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Generate response based on the prompt


        if "summary" in prompt.lower() or "summarize" in prompt.lower() or "summarise" in prompt.lower():
            # Get the summary statistics, reset index so that "count", "mean", etc., appear as a column
            summary_stats = df.describe().reset_index()
            # Convert the summary statistics to markdown format
            response = summary_stats.to_markdown()

        elif "unique" in prompt.lower():
            if any(col in prompt for col in specific_columns):
                response = ""
                for col in specific_columns:
                    # Check if the requested column exists in the dataframe
                    if col in df.columns and col in prompt:
                        unique_values = df[col].unique()  # Get unique values
                        num_unique_values = df[col].nunique()  # Get number of unique values

                        # Append the unique values and their count for each specific column
                        response += f"\n'{col}': {num_unique_values} unique values : {', '.join(map(str, unique_values))}\n"
                    elif col not in df.columns:
                        response += f"\n'{col}' is not a valid column in the data.\n"

            else:
                # Create a dictionary to store the data for each column
                unique_data = {
                    "Column": [],
                    "Number of Unique Values": [],
                    "Unique Values": []
                }

                # Loop through each column to collect unique values and their counts
                for col in df.columns:
                    unique_values = df[col].unique()  # Get unique values
                    num_unique_values = df[col].nunique()  # Get number of unique values

                    # Append the information to the dictionary
                    unique_data["Column"].append(col)
                    unique_data["Number of Unique Values"].append(num_unique_values)
                    unique_data["Unique Values"].append(", ".join(map(str, unique_values)))  # Convert unique values to a comma-separated string

                # Convert the dictionary to a DataFrame
                unique_df = pd.DataFrame(unique_data)

                # Display the DataFrame as a table in markdown format
                response = unique_df.to_markdown()

        elif "missing" in prompt.lower():
            if any(col in prompt for col in specific_columns):
                response = ""
                for col in specific_columns:
                    # Check if the requested column exists in the dataframe
                    if col in df.columns and col in prompt:
                        # Calculate the missing values for the specific column
                        missing_count = df[col].isna().sum()

                        # Append the missing values count for the column
                        response += f"\n'{col}': {missing_count} missing values\n"
                    elif col not in df.columns:
                        response += f"\n'{col}' is not a valid column in the data.\n"
            else:
                # Calculate missing values for each column
                missing_info = df.isna().sum()

                # Create a DataFrame to format it nicely
                missing_df = pd.DataFrame({
                    "Column": missing_info.index,
                    "Missing Values": missing_info.values
                })

                # Convert the DataFrame to markdown format
                response = missing_df.to_markdown(index=False)

        elif "average" in prompt.lower() or "mean" in prompt.lower():
            if any(col in prompt for col in specific_columns):
                response = ""
                for col in specific_columns:
                    # Check if the requested column exists in the dataframe
                    if col in df.columns and col in prompt:
                        # Check if the column is numeric before calculating the average
                        if pd.api.types.is_numeric_dtype(df[col]):
                            avg_value = df[col].mean().round(2)  # Calculate average for the specific column
                            response += f"\n{col} average: {avg_value} \n"
                        else:
                            response += f"\n'{col}' is not a numeric column and cannot have an average.\n"
                    elif col not in df.columns:
                        response += f"\n'{col}' is not a valid column in the data.\n"
            else:
                # Calculate the average for each numeric column
                average_values = df.select_dtypes(include=[np.number]).mean().round(2)

                # Create a DataFrame to display the averages
                average_df = pd.DataFrame({
                    "Column": average_values.index,
                    "Average": average_values.values
                })

                # Convert the DataFrame to markdown format
                response = average_df.to_markdown(index=False)

        elif "median" in prompt.lower() or "best value" in prompt.lower():
            if any(col in prompt for col in specific_columns):
                response = ""
                for col in specific_columns:
                    # Check if the requested column exists in the dataframe
                    if col in df.columns and col in prompt:
                        # Check if the column is numeric before calculating the median
                        if pd.api.types.is_numeric_dtype(df[col]):
                            median_value = df[col].median().round(2)  # Calculate median for the specific column
                            response += f"\n{col}: {median_value} \n"
                        else:
                            response += f"\n'{col}' is not a numeric column and cannot have a {prompt.lower()}.\n"
                    elif col not in df.columns:
                        response += f"\n'{col}' is not a valid column in the data.\n"
            else:
                # Calculate the median for each numeric column
                median_values = df.select_dtypes(include=[np.number]).median().round(2)

                # Create a DataFrame to display the medians
                median_df = pd.DataFrame({
                    "COLUMN": median_values.index,
                    prompt.upper(): median_values.values
                })

                # Convert the DataFrame to markdown format
                response = median_df.to_markdown(index=False)

        elif "mode" in prompt.lower() or "most frequent" in prompt.lower():
            if any(col in prompt for col in specific_columns):
                response = ""
                for col in specific_columns:
                    # Check if the requested column exists in the dataframe
                    if col in df.columns and col in prompt:
                        # Calculate the mode for the specific column
                        mode_values = df[col].mode()
                        if not mode_values.empty:
                            # Join the mode values (if more than one) and round them if they are numeric
                            mode_str = ", ".join(map(str, mode_values.round(2) if pd.api.types.is_numeric_dtype(mode_values) else mode_values))
                            response += f"\n'{col}' mode(s): {mode_str}\n"
                        else:
                            response += f"\n'{col}' has no mode (empty column).\n"
                    elif col not in df.columns:
                        response += f"\n'{col}' is not a valid column in the data.\n"
            else:
                # Calculate the mode for each column in the dataset
                response = ""
                for col in df.columns:
                    mode_values = df[col].mode()
                    if not mode_values.empty:
                        # Join the mode values (if more than one) and round them if they are numeric
                        mode_str = ", ".join(map(str, mode_values.round(2) if pd.api.types.is_numeric_dtype(mode_values) else mode_values))
                        response += f"\n'{col}' mode(s): {mode_str} \n"
                    else:
                        response += f"\n'{col}' has no mode (empty column).\n"


        elif "anomalies" in prompt.lower():
            # Create a DataFrame to hold anomalies data
            anomalies_data = []
            anomalies = {'datetime': [], 'column': [], 'value': [], 'anomaly_type': []}

            # Check if a specific column is mentioned in the prompt
            specified_columns = []
            for col in specific_columns:
                if col in prompt:
                    specified_columns.append(col)

            # If specific columns are specified, analyze only those columns
            if specified_columns:
                for column in specified_columns:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        # Check for specific Z-score threshold
                        z_threshold = z_thresholds.get(column,3 )
                        mean_value = df[column].mean()
                        std_dev = df[column].std()


                         # Calculate Z-scores
                        z_scores = (df[column] - mean_value) / std_dev
                        outliers = abs(z_scores) > z_threshold

                        # Detect sudden changes using both previous and next values
                        diff_prev = df[column].diff()
                        diff_next = df[column].shift(-1) - df[column]
                        changes_both = (abs(diff_prev) > mean_value) & (abs(diff_next) > mean_value)

                        anomaly_mask = outliers | changes_both

                        # Filter out consecutive values that are the same
                        previous_value = None
                        filtered_anomalies = []
                        for idx, is_anomaly in anomaly_mask.items():
                            if is_anomaly:
                                if outliers[idx]:
                                    anomaly_type = f"{column} anomaly (outlier)"
                                elif changes_both[idx]:
                                    if diff_prev[idx] > 0 and diff_next[idx] > 0:
                                        anomaly_type = f"{column} anomaly (increased from both previous and next)"
                                    elif diff_prev[idx] < 0 and diff_next[idx] < 0:
                                        anomaly_type = f"{column} anomaly (decreased from both previous and next)"
                                    else:
                                        anomaly_type = f"{column} anomaly (sudden change)"

                                anomalies['datetime'].append(df['datetime'][idx])
                                anomalies['column'].append(column)
                                anomalies['value'].append(df[column][idx])
                                anomalies['anomaly_type'].append(anomaly_type)

                    anomalies_data = pd.DataFrame(anomalies)

                    # Step 5: Tabulate and return results as a string
                    if anomalies_data.empty:
                        response += f"No anomalies were detected."
                    else:
                        # Tabulate the anomalies DataFrame in a formal table format
                        # Create a response string without headers
                        response = "### Detected Anomalies:\n\n| Datetime           | Column | Value | Anomaly Type |\n|-------------------|--------|-------|--------------|\n"
                        for i, row in anomalies_data.iterrows():
                            response += f"| {row['datetime']} | {row['column']} | {row['value']} | {row['anomaly_type']} |\n"

            else:
                # # If no specific columns are specified, analyze all relevant columns
                # for col in df.columns:
                #     if pd.api.types.is_numeric_dtype(df[col]):
                #         # Check for specific Z-score threshold
                #         threshold = z_thresholds.get(col, 3.0)  # Default to 3.0 if not specified

                #         # Calculate Z-scores for the column
                #         z_scores = stats.zscore(df[col].dropna())  # Drop NaN values before calculating Z-scores

                #         # Identify anomalies: absolute Z-score greater than threshold
                #         anomalies_mask = abs(z_scores) > threshold
                #         anomalies_values = df[col].dropna()[anomalies_mask]  # Get the actual values that are anomalies

                #         # Filter out consecutive values that are the same
                #         previous_value = None
                #         filtered_anomalies = []
                #         for idx in anomalies_values.index:
                #             value = anomalies_values[idx]
                #             if value != previous_value:  # Check if the current value is different from the previous one
                #                 filtered_anomalies.append(value)
                #             previous_value = value

                #         num_anomalies = len(filtered_anomalies)

                #         if num_anomalies > 0:
                #             response += f"\n'{col}': {num_anomalies} anomalies detected: {', '.join(map(str, filtered_anomalies))}\n"
                #             anomalies_data.append((col, filtered_anomalies))  # Store column and its anomalies
                #         else:
                #             response += f"\n'{col}': No anomalies detected.\n"
                #     else:
                #         response += f"\n'{col}': Non-numeric column, anomalies cannot be detected.\n"  

                # Step 4: Anomaly detection process
                columns_to_check = ["Humidity", "Temperature", "EC", "PH", "N", "P", "K"]
                anomalies = {'datetime': [], 'column': [], 'value': [], 'anomaly_type': []}

                for column in columns_to_check:
                    z_threshold = z_thresholds.get(column,3 )
                    mean_value = df[column].mean()
                    std_dev = df[column].std()

                    # Calculate Z-scores
                    z_scores = (df[column] - mean_value) / std_dev
                    outliers = abs(z_scores) > z_threshold

                    # Detect sudden changes using both previous and next values
                    diff_prev = df[column].diff()
                    diff_next = df[column].shift(-1) - df[column]
                    changes_both = (abs(diff_prev) > mean_value) & (abs(diff_next) > mean_value)

                    anomaly_mask = outliers | changes_both

                    for idx, is_anomaly in anomaly_mask.items():
                        if is_anomaly:
                            if outliers[idx]:
                                anomaly_type = f"{column} anomaly (outlier)"
                            elif changes_both[idx]:
                                if diff_prev[idx] > 0 and diff_next[idx] > 0:
                                    anomaly_type = f"{column} anomaly (increased from both previous and next)"
                                elif diff_prev[idx] < 0 and diff_next[idx] < 0:
                                    anomaly_type = f"{column} anomaly (decreased from both previous and next)"
                                else:
                                    anomaly_type = f"{column} anomaly (sudden change)"

                            anomalies['datetime'].append(df['datetime'][idx])
                            anomalies['column'].append(column)
                            anomalies['value'].append(df[column][idx])
                            anomalies['anomaly_type'].append(anomaly_type)

                anomalies_df = pd.DataFrame(anomalies)

                 # Step 5: Tabulate and return results as a string
                if anomalies_df.empty:
                    response += f"No anomalies were detected."
                else:
                    # Tabulate the anomalies DataFrame in a formal table format
                     # Create a response string without headers
                    response = "### Detected Anomalies:\n\n| Datetime           | Column | Value | Anomaly Type |\n|-------------------|--------|-------|--------------|\n"
                    for i, row in anomalies_df.iterrows():
                        response += f"| {row['datetime']} | {row['column']} | {row['value']} | {row['anomaly_type']} |\n"


        elif "datatype" in prompt.lower():
             # Get the data types for each column
            datatypes = df.dtypes

            # Check if specific columns are mentioned in the prompt
            specified_columns = [col for col in df.columns if col in prompt]

            if specified_columns:
                # Filter the datatypes to include only specified columns
                datatypes = datatypes[specified_columns]

                # Create a DataFrame to format it nicely
                datatype_df = pd.DataFrame({
                    "Column": datatypes.index,
                    "Data Type": datatypes.values
                })

                # Convert the DataFrame to markdown format
                response = datatype_df.to_markdown(index=False)
            else:
                # Create a DataFrame to format it nicely for all columns
                datatype_df = pd.DataFrame({
                    "Column": datatypes.index,
                    "Data Type": datatypes.values
                })

                # Convert the DataFrame to markdown format
                response = datatype_df.to_markdown(index=False)

        elif "total number of data" in prompt.lower() or "number of rows" in prompt.lower() or "total rows" in prompt.lower() or "row" in prompt.lower():
            # Check if df is defined and has data
            if 'df' in locals() and not df.empty:
                total_rows = len(df)
                response = f"The total number of data points (rows) in the dataset is: {total_rows}."

        elif "column" in prompt.lower() or "number of columns" in prompt.lower():
            # Check if df is defined and has data
            if 'df' in locals() and not df.empty:
                total_columns = len(df.columns)
                column_names = ', '.join(df.columns)  # Get the column names
                response = f"The total number of columns in the dataset is: {total_columns}. The columns are: {column_names}."


        elif "hello" in prompt.lower() or "hi" in prompt.lower() or "wassup" in prompt.lower():
            response = "Hello! How can I assist you today?"

        elif "thank you" in prompt.lower():
            response = "You're welcome! If you have any other questions, feel free to ask."

        elif "humi" in  prompt.lower():
            response = "Do you mean Humidity?"

        elif "temp" in  prompt.lower():
            response = "Do you mean Temperature?"

        elif "ec" in prompt or "ph" in prompt or "n" in prompt or "k" in prompt or "p" in prompt or "temperature" in prompt or "humidity" in prompt:
            response = "I'm sorry please be more specific."

        elif "bye" in prompt.lower():
            response = "Goodbye!"

        else:
        # Handle unrecognized prompts
            response = "I'm sorry, I didn't understand that. Can you please rephrase your question or ask about something else? "

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:

    st.write("Please upload a Soil Data (CSV file) to proceed.")
