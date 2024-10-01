import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}<style>', unsafe_allow_html=True)

# -- FUNCTIONS --
def transform_data(file):
    """ Formats the data to create visualizations """
    df = pd.read_csv(file)
    df['datetime'] = df['date'] + ' ' + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')
    return df

def display_metrics(filtered_df, metric_placeholder):
    """ Displays metrics (average values) of all parameters for the filtered data """
    avg_n = round(filtered_df['N'].mean())
    avg_p = round(filtered_df['P'].mean())
    avg_k = round(filtered_df['K'].mean())
    avg_temp = round(filtered_df['Temp'].mean())
    avg_humi = round(filtered_df['Humi'].mean())

    with metric_placeholder:
        met1, met2, met3, met4, met5 = st.columns((1, 1, 1, 1.5, 1.5))
        met1.metric("Nutrient N", f"{avg_n} mg/L")
        met2.metric("Nutrient P", f"{avg_p} mg/L")
        met3.metric("Nutrient K", f"{avg_k} mg/L")
        met4.metric("Temperature", f"{avg_temp}°C")
        met5.metric("Humidity", f"{avg_humi}%")

def filter(df):
    """Creates a filter for the visualisations. Visualisations and metrics change according to the filters"""
    # Get the min and max dates from the dataframe
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()

    # Filter options
    st.write("#")
    selected_param = st.selectbox(
        "Parameters",
        ("NPK", "Temperature", "Humidity")
    )
    start_date = st.date_input('Start date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input('End date', min_value=min_date, max_value=max_date, value=max_date)
    st.write("total number of data:\n100")
    
    # Get filtered data based on selected parameters
    filtered_df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]

    return filtered_df, selected_param

def display_timegraph(filtered_df, selected_param):
    """ Display the time series graph of the selected parameter against time """

    param_map = {
        "NPK": ['N', 'P', "K"],
        "Temperature": "Temp",
        "Humidity": "Humi"
    }

    # Time series graph
    param = param_map[selected_param]
    time_fig = px.line(filtered_df, x='datetime', y=param, title=f'{selected_param} Levels Over Time', line_shape="spline")
    time_fig.update_xaxes(
        showgrid=True,
        gridcolor= 'white',
        rangeslider_visible=True,
        rangeslider=dict(
            visible=True,
            bgcolor='#f6f8fc',
            thickness=0.1
        ),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    time_fig.update_yaxes(showgrid=True,gridcolor= 'white')

    # Update layout for figure size, background, and title
    time_fig.update_layout(
        width=1500,  
        height=600, 
        title_font_size=20,  
        plot_bgcolor='#f6f8fc',   
        font=dict(
            color="black",  # Customize font color
            size=14  # Font size for labels
        )
    )
    st.plotly_chart(time_fig)

# -- DASHBOARD PAGE -- 
# Reducing whitespace on the top of the page
st.markdown("""
<style>

.block-container
{
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

st.subheader("Soil Analytics Dashboard")
st.write("")



# Load and transform data
df = transform_data('views/out_sensor.csv')

metric_placeholder = st.empty()
col1, col2 = st.columns((4,1), gap= "medium")
with col2:
    filtered_df, selected_param = filter(df)
    # NPK proportions donut chart
    npk_sums = filtered_df[['N', 'P', 'K']].sum()
    donut_fig = go.Figure(data=[go.Pie(labels=['N', 'P', 'K'], values=npk_sums, hole=.3)]) 
    col2.plotly_chart(donut_fig)

display_metrics(filtered_df, metric_placeholder)
with col1:
    display_timegraph(filtered_df, selected_param)
    st.code("descriptive stats", language="python")



