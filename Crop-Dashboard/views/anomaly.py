import pandas as pd
import os

def get_values(csv_file, use_median=False):
    # Open the file and skip irrelevant lines to find the correct header
    with open(csv_file, 'r') as file:
        skip_lines = 0
        for line in file:
            # Split the line by commas and check if it matches the valid header
            if line.strip().split(',') == ["date", "time", "Humi", "Temp", "EC", "PH", "N", "P", "K"]:
                break
            skip_lines += 1

    # Load the CSV file into a pandas DataFrame, skipping irrelevant lines
    df = pd.read_csv(csv_file, skiprows=skip_lines)

    # Columns to analyze
    columns_to_check = ["Humi", "Temp", "EC", "PH", "N", "P", "K"]

    # Create a dictionary to store the best values for each column
    best_values = {}

    # Loop through each column and calculate either the mean or median
    for column in columns_to_check:
        if use_median:
            # Calculate the median
            best_value = df[column].median()
        else:
            # Calculate the mean
            best_value = round(df[column].mean(), 2)
        
        # Store the best value in the dictionary
        best_values[column] = float(best_value)

    return best_values


def detect_anomalies(df, z_thresholds, change_threshold=0.5):

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

        # Detect sudden changes by comparing to the previous value
        diff = df[column].diff()
        changes = abs(diff) > change_threshold * mean_value

        # Combine outliers and sudden changes
        anomaly_mask = outliers | changes

        # Record the anomalies
        for idx, is_anomaly in anomaly_mask.items():
            if is_anomaly:
                # Determine anomaly type
                if outliers[idx]:
                    anomaly_type = f"{column} anomaly"
                else:
                    if diff[idx] > 0:
                        anomaly_type = f"{column} anomaly (increased)"
                    else:
                        anomaly_type = f"{column} anomaly (decreased)"
                
                anomalies['datetime'].append(df['datetime'][idx])
                anomalies['column'].append(column)
                anomalies['value'].append(df[column][idx])
                anomalies['anomaly_type'].append(anomaly_type)

    # Convert anomalies dictionary to a DataFrame
    anomalies_df = pd.DataFrame(anomalies)

    return anomalies_df

# Example usage:
# Define specific z_thresholds for each column
z_thresholds = {
    "Humi": 4,  # A larger threshold for humidity
    "Temp": 4,  # Default threshold for temperature
    "EC": 5,
    "PH": 5,
    "N": 6,
    "P": 6,
    "K":6
}

# # Detect anomalies with specific z_thresholds
# anomalies = detect_anomalies('Inside sensor.csv', z_thresholds)
# print(anomalies)

# # Get the average (mean) values for each column
# best_avg_values = get_values('Inside sensor.csv')
# print("Best Average Values:", best_avg_values)

# # Get the median values for each column
# best_median_values = get_values('Inside sensor.csv', use_median=True)
# print("Best Median Values:", best_median_values)

# validation_result = validate_file('Inside sensor.csv')
# print(validation_result)




