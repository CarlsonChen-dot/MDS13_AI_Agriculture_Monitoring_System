import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np




# -- FUNCTIONS --
def transform_data(file):
    """ Formats the data to create visualsations """
    df = pd.read_csv(file)
    df['datetime'] = df['date'] + ' ' + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')
    return df

def display_metrics(df):
    """ Displays metrics (average values) of all parameters"""
    avg_n = round(df['N'].mean())
    avg_p = round(df['P'].mean())
    avg_k = round(df['K'].mean())
    avg_temp = round(df['Temp'].mean())
    avg_humi = round(df['Humi'].mean())

    met1, met2, met3, met4, met5 = st.columns((1,1,1,1.5,1.5))
    met1.metric("Nutrient N", f"{avg_n} mg/L")
    met2.metric("Nutrient P", f"{avg_p} mg/L")
    met3.metric("Nutrient K", f"{avg_k} mg/L")
    met4.metric("Temperature", f"{avg_temp} °F")
    met5.metric("Humidity", f"{avg_humi}%")

def filter(df):
    """Creates a filter for the visualisations. Visualisations change according to the filters"""
    # Get the min and max dates from the dataframe
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()

    # Filter options
    selected_param = st.selectbox(
        "Parameters",
        ("NPK","Temperature","Humidity")
    )
    start_date = st.date_input('Start date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input('End date', min_value=min_date, max_value=max_date, value=max_date)

    # Get filtered data based on selected parameters
    filtered_df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]

    return filtered_df, selected_param

def display_timegraph(selected_param):
    """ Display the time series graph of selected parameter against time"""

    param_map = {
        "NPK":['N','P',"K"],
        "Temperature" : "Temp",
        "Humidity": "Humi"
    }

    # Time series graph
    param = param_map[selected_param]
    time_fig = px.line(filtered_df, x='datetime', y= param, title=f'{selected_param} Levels Over Time', line_shape= "spline")
    time_fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider=dict(
            visible=True,
            bgcolor='#97a29a', 
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

    st.plotly_chart(time_fig)



st.header("Soil Analytics")
df = transform_data('views/out_sensor.csv')
filtered_df, selected_param = filter(df)
display_metrics(filtered_df)
display_timegraph(selected_param)
    

#total data
#additional analytics
#highest captured values

    # # NPK proportions donut chart
    # npk_sums = filtered_df[['N', 'P', 'K']].sum()
    # donut_fig = go.Figure(data=[go.Pie(labels=['N', 'P', 'K'], values=npk_sums, hole=.3)]) 
    # col2.plotly_chart(donut_fig)


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -- FUNCTIONS --

### view raw data

def filter_data(file):
    df = pd.read_csv(file)
    df['datetime'] = df['date'] + ' ' + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')
    return df

def avg(df,param):
    return df[param].mean()

# range date function
df = filter_data('views/out_sensor.csv')

# -- DASHBOARD PAGE -- 
st.header("Soil Analytics")

# Metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Nitrogen (N)", "70 mg/L", "1.2 °F")
col2.metric("Phosporus (P)", "9 mg/L", "-8%")
col3.metric("Potassium (K)", "86 mg/L", "1.3")

# Plot NPK
col1, col2 = st.columns([3,1])

# Time series vs NPK
time_fig = px.line(df, x='datetime', y='Humi', title='Nutrient levels')
time_fig.update_xaxes(
    rangeslider_visible=True,
    rangeslider=dict(
        visible=True,
        bgcolor='#97a29a', 
        thickness=0.05 
    ),
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
col1.plotly_chart(time_fig)

# Donut chart showing NPK percentage
labels = ['Nitrogen', 'Phosphorus', 'Potassium']
values = [40, 30, 30]
donut_fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
donut_fig .update_traces(marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99']),
                  hoverinfo='label+percent', textinfo='value')
col2.plotly_chart(donut_fig)


# Plot area data that switch between temp, humidity, ph and ec
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.area_chart(chart_data)


# Example pH value
ph_value = 6.8

# Create a gauge chart using Plotly
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=ph_value,
    title={'text': "Soil pH Level"},
    gauge={'axis': {'range': [0, 14]},
           'steps': [
               {'range': [0, 3], 'color': "#ff6666"},
               {'range': [3, 6], 'color': "#ffcc66"},
               {'range': [6, 8], 'color': "#99ff99"},
               {'range': [8, 11], 'color': "#66b3ff"},
               {'range': [11, 14], 'color': "#cc99ff"}],
           'bar': {'color': "#000000"}}))

# Customize chart appearance (optional)
fig.update_layout(height=400)

# Display the gauge chart in Streamlit
st.plotly_chart(fig)



#### !!!!!! DUMP ####

st.title("Analytics Demo")

# metrics
col1, col2, col3 = st.columns(3)
col1.metric("Nitrogen", "70 mg/L", "1.2 °F")
col2.metric("Phosporus", "9 mg/L", "-8%")
col3.metric("Humidity", "86 mg/L", "1.3")

st.write("\n")
col1, col2 = st.columns([3, 1])

col1.subheader("Nutrient Levels")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
col1.line_chart(chart_data)

df = px.data.tips()
fig = px.pie(df, values='tip', names='day', color_discrete_sequence=px.colors.sequential.RdBu)
col2.plotly_chart(fig)

st.write("\n")
col1.subheader("Other Levels")
col3, col4, col5= st.columns([3,1,1])
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
col3.area_chart(chart_data)
col4.metric("Humidity", "86 mg/L", "1.3")
col4.metric("Humidity", "86 mg/L", "1.3")

import plotly.graph_objects as go

fig3 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 270,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Speed"}))
col5.plotly_chart(fig3)

# Data columns
df = pd.read_csv('backend/out_sensor.csv')
df['datetime'] = df['date'] + ' ' + df['time']
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')

# Create the Plotly figure
fig = px.line(df, x='datetime', y=['N', 'P', 'K'], title='Nutrient levels')

# Update x-axes with range slider and selectors
fig.update_xaxes(
    rangeslider_visible=True,
    rangeslider=dict(
        visible=True,
        bgcolor='#97a29a', 
        thickness=0.05 
    ),
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

st.plotly_chart(fig)


fig2 = px.area(df, x="datetime", y="Temp")
fig3 = px.area(df, x="datetime", y="Humi")
fig4 = px.area(df, x="datetime", y="PH")


fig2.update_yaxes(
    title = "Temperature",
    range=[0,40])

st.plotly_chart(fig2)
st.plotly_chart(fig3)
st.plotly_chart(fig4)


fig5 = px.scatter(df, x='Temp', y='Humi', title='Temperature vs. Humidity')
st.plotly_chart(fig5)

fig6 = px.violin(df, y='Temp', box=True, points='all', title='Temperature Violin Plot')
st.plotly_chart(fig6)

df_drop = df.drop(['datetime', 'date', 'time'], axis=1)
fig7 = px.imshow(df_drop)
st.plotly_chart(fig7)


fig8 = px.scatter(df, x="Temp", y="Humi")
fig9 = px.scatter(df, x="Temp", y="PH")
fig10 = px.scatter(df, x="Temp", y="EC")

fig11 = px.scatter(df, x="Humi", y="PH")
fig12 = px.scatter(df, x="Humi", y="EC")
fig13 = px.scatter(df, x="PH", y="EC")

col1, col2 = st.columns([3, 2])
with col1:
    st.plotly_chart(fig8)
    st.plotly_chart(fig9)
    st.plotly_chart(fig10)


with col2:
    st.plotly_chart(fig11)
    st.plotly_chart(fig12)
    st.plotly_chart(fig13)


import plotly.express as px
import pandas as pd

fig = px.line(df, x='datetime', y='N', title='Time Series with Rangeslider')

fig.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig)


import plotly.express as px
import streamlit as st

# Sample DataFrame
df = px.data.stocks()

# Create line plot with range slider
fig = px.line(df, x='date', y='GOOG', title='Google Stock Prices with Range Slider')

# Add range slider
fig.update_xaxes(rangeslider_visible=True)

# Display plot in Streamlit
st.plotly_chart(fig)
