import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import altair as alt

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
        met1, met2, met3, met4, met5 = st.columns(5)
        with met1.container():
            st.markdown(f'<p class="n_text metrics">Nitrogen (N)<br></p><p class="avg">{avg_n} mg/L</p>', unsafe_allow_html = True)
        with met2.container():
            st.markdown(f'<p class="p_text metrics">Phosphorus (P)<br></p><p class="avg">{avg_p} mg/L</p>', unsafe_allow_html = True)
        with met3.container():
            st.markdown(f'<p class="k_text metrics">Potassium (K)<br></p><p class="avg">{avg_k} mg/L</p>', unsafe_allow_html = True)
        with met4.container():
            st.markdown(f'<p class="temp_text metrics">Temperature <br></p><p class="avg">{avg_temp} °C</p>', unsafe_allow_html = True)
        with met5.container():
            st.markdown(f'<p class="humi_text metrics">Humidity <br></p><p class="avg">{avg_humi} %</p>', unsafe_allow_html = True)

def filter(df):
    """Creates a filter for the visualisations. Visualisations and metrics change according to the filters"""
    # Get the min and max dates from the dataframe
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()

    st.markdown(f'<p class="params_text">Chart Data Parameters', unsafe_allow_html = True)
    # Filter options
    selected_feature = st.selectbox(
        "Features",
        ("NPK", "Temperature", "Humidity")
    )
    start_date = st.date_input('Start date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input('End date', min_value=min_date, max_value=max_date, value=max_date)

    # Get filtered data based on selected parameters
    filtered_df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]

    return filtered_df, selected_feature

def display_timegraph(filtered_df, selected_feature):
    """ Display the time series graph of the selected parameter against time """

    param_map = {
        "NPK": ['N', 'P', "K"],
        "Temperature": "Temp",
        "Humidity": "Humi"
    }

    # Time series graph
    param = param_map[selected_feature]
    time_fig = px.line(filtered_df, x='datetime', y=param, title=f'Soil {selected_feature} Levels Over Time', line_shape="spline")
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
        height=570, 
        title_font_size=20,  
        plot_bgcolor='#f6f8fc',   
        font=dict(
            color="black",  # Customize font color
            size=14  # Font size for labels
        )
    )
    
    st.plotly_chart(time_fig)

def display_donut_chart(df):
    """ Display a donut chart showing the distribution of N, P, and K in the soil """
    # Sum the nutrient columns
    npk_sums = df[['N', 'P', 'K']].sum()

    # Melt the DataFrame to create a category for N, P, K and their values
    npk_df = pd.DataFrame({
        'Nutrient': ['N', 'P', 'K'],
        'Value': [npk_sums['N'], npk_sums['P'], npk_sums['K']]
    })

    # Create the donut chart using Altair
    chart = alt.Chart(npk_df).mark_arc(innerRadius=30).encode(
        theta=alt.Theta(field="Value", type="quantitative"),
        color=alt.Color(field="Nutrient", type="nominal")
    ).properties(
        width=200,
        height=200  
    ).configure_legend(
        orient='bottom',  
        labelFontSize=12, 
        titleFontSize=14 
    )   

    # Display the chart in Streamlit
    st.markdown(f'<p class="donut_chart_title">Soil Nutrient Distribution', unsafe_allow_html = True)
    st.altair_chart(chart, use_container_width=True)

def display_scatterplot(df, x):
    """ Display a scatter plot of the selected features """

    against_map ={
        "NPK":["Temp","Humi", "Temperature", "Humidity", ['N', 'P', 'K']],
        "Temperature": ["Humi",['N', 'P', 'K'], "Humidity", "NPK","Temp"],
        "Humidity":["Temp", ['N', 'P', 'K'], "Temperature", "NPK","Humi"] 
    }

    # Create a scatter plot using Plotly
    fig1 = px.scatter(df, x=against_map[x][4], y=against_map[x][0], title=f'{x} vs {against_map[x][2]}')
    fig2 = px.scatter(df, x=against_map[x][4], y=against_map[x][1], title=f'{x} vs {against_map[x][3]}')
    
    # Update layout to adjust aesthetics
    fig1.update_layout(
        xaxis_title=x,   # X-axis title
        yaxis_title=against_map[x][2],   # Y-axis title
        width=100,       # Set width
        height=400       # Set height
    )
    fig2.update_layout(
        xaxis_title=x,   # X-axis title
        yaxis_title=against_map[x][3],   # Y-axis title
        width=100,       # Set width
        height=400       # Set height
    )

    # Display the scatter plot in Streamlit
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# -- DASHBOARD PAGE --
st.markdown('<p class="dashboard_title">Soil Dashboard</p>', unsafe_allow_html = True)
tab1, tab2, tab3 = st.tabs(["Analytics", "Data Overview", "Advanced Analytics"])

# Load and transform data
df = transform_data('views/out_sensor.csv')

# -- ANALYTICS TAB --
with tab1:
    metric_placeholder = st.empty()
    r2cols = st.columns((4,1.3), gap= "medium")
    with r2cols[1]:
        filtered_df, selected_feature = filter(df)
        st.divider()
        display_donut_chart(filtered_df)    
    display_metrics(filtered_df, metric_placeholder)
    with r2cols[0]:
        display_timegraph(filtered_df, selected_feature)
        st.code("descriptive stats", language="python")


# -- DATA OVERVIEW TAB --
with tab2:
    st.write(filtered_df)


# -- ADVANCED ANALYTICS TAB --
def display_boxplot(df, selected_feature):
    """ Display a box plot of the selected_feature """

    param_map = {
        "NPK": ['N', 'P', 'K'],
        "Temperature": "Temp",
        "Humidity": "Humi"
    }

    # Create a box plot for the 'temp' column using Plotly
    fig = px.box(df, y=param_map[selected_feature], title=f'Distribution of {selected_feature} Data')

    # Update layout to adjust aesthetics
    fig.update_layout(
        yaxis_title=f"{selected_feature}",   # Y-axis title
        boxmode='group',                  # Group boxes together
        width=100,                        # Set width
        height=400                        # Set height
    )

    

    # Display the box plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.caption("Analytics based on selected parameters:")
    st.success(selected_feature)
    cols = st.columns(2, gap="medium")
    with cols[0]:
        # Distribution of data
        display_boxplot(filtered_df, selected_feature)

    with cols[1]:
        # Calculate correlation matrix
        corr_matrix = filtered_df.drop(columns=['datetime','date','time']).corr()

        # Create a heatmap using Plotly
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto", 
                        color_continuous_scale="Viridis", 
                        title="Correlation Map of Soil Properties")

        # Display the heatmap in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    display_scatterplot(filtered_df, selected_feature)
