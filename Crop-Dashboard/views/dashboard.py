import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import altair as alt
import os

css_file = os.path.join(os.path.dirname(__file__), '../style.css')
with open(css_file) as f:
    st.markdown(f'<style>{f.read()}<style>', unsafe_allow_html=True)

def display_metrics(filtered_df, metric_placeholder):
    """ Displays metrics (average values) of all parameters for the filtered data """
    avg_n = round(filtered_df['N'].mean())
    avg_p = round(filtered_df['P'].mean())
    avg_k = round(filtered_df['K'].mean())
    avg_temp = round(filtered_df['Temperature'].mean())
    avg_humi = round(filtered_df['Humidity'].mean())
    avg_ec = round(filtered_df['EC'].mean())
    avg_ph = round(filtered_df['PH'].mean())

    with metric_placeholder:
        met1, met2, met3, met4, met5 = st.columns(5)
        with met1.container():
            st.markdown(f'<p class="metrics">Nitrogen (N)<br></p><p class="avg">{avg_n} mg/L</p>', unsafe_allow_html = True)
        with met2.container():
            st.markdown(f'<p class="metrics">Phosphorus (P)<br></p><p class="avg">{avg_p} mg/L</p>', unsafe_allow_html = True)
        with met3.container():
            st.markdown(f'<p class="metrics">Potassium (K)<br></p><p class="avg">{avg_k} mg/L</p>', unsafe_allow_html = True)
        with met4.container():
            st.markdown(f'<p class="metrics">Temperature <br></p><p class="avg">{avg_temp} °C</p>', unsafe_allow_html = True)
        with met5.container():
            st.markdown(f'<p class="metrics">Humidity <br></p><p class="avg">{avg_humi} %</p>', unsafe_allow_html = True)

def filter(df):
    """Creates a filter for the visualisations. Visualisations and metrics change according to the filters"""
    
    # Get the min and max dates from the dataframe
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()

    st.markdown(f'<p class="params_text">Chart Data Parameters', unsafe_allow_html = True)
    # Filter options
    selected_feature = st.selectbox(
        "Features",
        ("NPK", "Temperature", "Humidity", "PH", "EC")
    )
    start_date = st.date_input('Start date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input('End date', min_value=min_date, max_value=max_date, value=max_date)

    # Get filtered data based on selected parameters
    filtered_df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]

    return filtered_df, selected_feature

def display_timegraph(filtered_df, selected_feature):
    """ Display the time series graph of the selected parameter against time """

    param = selected_feature
    if selected_feature == "NPK":
        param = ['N', 'P', "K"]

    custom_hover_text = {
        "NPK": {
            'N': 'Nitrogen (mg/L)', 
            'P': 'Phosphorus (mg/L)', 
            'K': 'Potassium (mg/L)'
        },
        "Temperature": 'Temperature (°C)',
        "Humidity": 'Humidity (%)',
        "PH": 'pH Level',
        "EC": 'Electrical Conductivity (uS/cm)'
    }

    # Time series graph
    if selected_feature == "NPK":
        # For NPK, display all three (N, P, K) with labels
        time_fig = px.line(
            filtered_df, x='datetime', 
            y= param,  # Plot N, P, and K together
            title=f'Soil {selected_feature} Levels Over Time', 
            template="plotly_dark",
            line_shape="spline"
        )
        # Customize hover labels for NPK
        time_fig.for_each_trace(
            lambda trace: trace.update(
                hovertemplate=(
                    "<b>Date:</b> %{x|%B %d, %Y}<br>" +  # Display the date part
                    "<b>Time:</b> %{x|%H:%M:%S}<br>" +
                    f"<b>{custom_hover_text['NPK'][trace.name]}:</b> "  # Correct N, P, K label
                    "%{y:.2f} units<br>" +  # Ensure y value is accessed and formatted to 2 decimal places
                    "<extra></extra>"
                )
            )
        )

    else:
        # For Temperature and Humidity, use the same method as NPK
        hover_label = custom_hover_text[selected_feature]
        time_fig = px.line(
            filtered_df, x='datetime', 
            y=param,  # Plot the selected single parameter (Temp or Humi)
            title=f'{selected_feature} Levels Over Time', 
            template="plotly_dark",
            line_shape="spline"
        )
        # Customize hover labels for Temperature and Humidity
        time_fig.for_each_trace(
            lambda trace: trace.update(
                hovertemplate=(
                    "<b>Date:</b> %{x|%B %d, %Y}<br>" +  # Display the date part
                    "<b>Time:</b> %{x|%H:%M:%S}<br>" +   # Display the time part
                    f"<b>{hover_label}:</b> "
                    "%{y:.2f}<br>" +  # Format with two decimals
                    "<extra></extra>"
                )
            )
        )
    
    time_fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider=dict(
            visible=True,
            # bgcolor='#f6f8fc', # range selector color
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

    # Update layout for figure size, background, and title
    time_fig.update_layout(
        width=1500,  
        height=570, 
        title_font_size=20,  
        plot_bgcolor='#26282E',   
        font=dict(
            color="#f6f6f6",  
            size=14  
        )
    )
    
    st.plotly_chart(time_fig)

def display_donut_chart(df):
    """ Display a donut chart showing the distribution of N, P, and K in the soil """
    # Sum the nutrient columns
    npk_sums = df[['N', 'P', 'K']].sum()

    total_npk = npk_sums.sum()

    # Melt the DataFrame to create a category for N, P, K and their values
    npk_df = pd.DataFrame({
        'Nutrient': ['N', 'P', 'K'],
        'Value': [npk_sums['N'], npk_sums['P'], npk_sums['K']],
        'Percentage': [npk_sums['N'] / total_npk * 100, npk_sums['P'] / total_npk * 100, npk_sums['K'] / total_npk * 100]
    })

    # Define specific colors for each nutrient
    nutrient_colors = {'N': '#6973FC',  
                       'P': '#E4573B',  # Orange
                       'K': '#48CA95'}  # Green

    # Create the donut chart using Altair
    chart = alt.Chart(npk_df).mark_arc(innerRadius=30).encode(
        theta=alt.Theta(field="Value", type="quantitative"),
        color=alt.Color(field="Nutrient", type="nominal", scale=alt.Scale(domain=list(nutrient_colors.keys()), range=list(nutrient_colors.values()))),
        tooltip=[alt.Tooltip('Nutrient:N', title='Nutrient Type'), 
                 alt.Tooltip('Percentage:Q', title='Percentage (%)', format='.2f')]
        
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
        "NPK":["Temperature","Humidity", "PH", "EC"],
        "Temperature": [['N', 'P', 'K'], "Humidity", "PH", "EC"],
        "Humidity":[['N', 'P', 'K'], "Temperature", "PH", "EC"],
        "PH":[['N', 'P', 'K'], "Temperature","Humidity","EC"],
        "EC":[['N', 'P', 'K'], "Temperature","Humidity","PH"]
    }

    param = x
    if x == "NPK":
        param = ['N', 'P', "K"]
        against = "Temperature"
    else:
        against = "NPK"

    # Create a scatter plot using Plotly
    fig1 = px.scatter(df, x=param, y=against_map[x][0], title=f'{x} vs {against}')
    fig2 = px.scatter(df, x=param, y=against_map[x][1], title=f'{x} vs {against_map[x][1]}')
    fig3 = px.scatter(df, x=param, y=against_map[x][2], title=f'{x} vs {against_map[x][2]}')
    fig4 = px.scatter(df, x=param, y=against_map[x][3], title=f'{x} vs {against_map[x][3]}')

    # Custom tooltips
    if x == "NPK":
        fig1.update_traces(
            hovertemplate=(
                "<b>Nutrient value:</b> %{x}<br>" +
                f"<b>{against_map[x][0]}: </b>" + "%{y:.2f}<br>" + 
                "<extra></extra>"
            )
        )
        fig2.update_traces(
            hovertemplate=(
                "<b>Nutrient value:</b> %{x}<br>" +
                f"<b>{against_map[x][1]}: </b>" + "%{y:.2f}<br>" + 
                "<extra></extra>"
            )
        )
        fig3.update_traces(
            hovertemplate=(
                "<b>Nutrient value:</b> %{x}<br>" +
                f"<b>{against_map[x][2]}: </b>" + "%{y:.2f}<br>" + 
                "<extra></extra>"
            )
        )
        fig4.update_traces(
            hovertemplate=(
                "<b>Nutrient value:</b> %{x}<br>" +
                f"<b>{against_map[x][3]}: </b>" + "%{y:.2f}<br>" + 
                "<extra></extra>"
            )
        )
    else:
        # Customize hovertemplate for y-axis with Nutrients (N, P, K)
        for fig, y_feature in zip([fig1, fig2, fig3, fig4], against_map[x]):
            if y_feature == ['N', 'P', 'K']:
                fig.update_traces(
                    hovertemplate=(
                        f"<b>{x}: </b>"
                        "%{x}<br>" +
                        "<b>Nutrient value:</b> %{y:.2f}<br>" +  # Display y as nutrient value
                        #f"<b>{x}:</b> %{x}<br>" +
                        "<extra></extra>"
                    )
                )
            else:
                fig.update_traces(
                    hovertemplate=(
                        f"<b>{x}: </b>"
                        "%{x}<br>" +
                        f"<b>{y_feature}: </b>"
                        "%{y:.2f}<br>" +
                        "<extra></extra>"
                    )
                )


    # if x == "NPK":
    #     # Customize hover labels for the scatter plot
    #     fig1.update_traces(
    #         hovertemplate=(
    #             "<b>Nutrient value:</b> %{x}<br>" +  # Display the nutrient (x-axis)
    #             f"<b>{against_map[x][2]}:</b>" 
    #             "%{y:.2f}<br>" +  # Display the y value (y-axis)
    #             "<extra></extra>"
    #         )
    #     )
    #     fig2.update_traces(
    #         hovertemplate=(
    #             "<b>Nutrient value:</b> %{x}<br>" +  # Display the nutrient (x-axis)
    #             f"<b>{against_map[x][3]}:</b>" 
    #             "%{y:.2f}<br>" +  # Display the y value (y-axis)
    #             "<extra></extra>"
    #         )
    #     )
    # else:
    #     # Customize hover labels for the scatter plot
    #     fig1.update_traces(
    #         hovertemplate=(
    #             f"<b>{x}: </b>"
    #             "%{x}<br>" +  # Display the x value (x-axis)
    #             f"<b>{against_map[x][2]}: </b>" 
    #             "%{y:.2f}<br>" +  # Display the y value (y-axis)
    #             "<extra></extra>"
    #         )
    #     )

    #     fig2.update_traces(
    #         hovertemplate=(
    #             f"<b>{x}: </b>" 
    #             "%{x}<br>" +  # Display the x value (x-axis)
    #             "Nutrient value: %{y:.2f}<br>" +  # Display the y value (y-axis)
    #             "<extra></extra>"
    #         )
    #     )

    #  Update layout to adjust aesthetics
    fig1.update_layout(
        xaxis_title=x,   
        yaxis_title=against,  
        width=100,       
        height=400      
    )
    fig2.update_layout(
        xaxis_title=x,  
        yaxis_title=against_map[x][1],   
        width=100,       
        height=400       
    )
    fig3.update_layout(
        xaxis_title=x,   
        yaxis_title=against_map[x][2],  
        width=100,       
        height=400      
    )
    fig4.update_layout(
        xaxis_title=x,  
        yaxis_title=against_map[x][3],   
        width=100,       
        height=400       
    )

    # Display the scatter plot in Streamlit
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
    
def display_boxplot(df, selected_feature):
    """ Display a box plot of the selected_feature """
    param = selected_feature
    if selected_feature == "NPK":
        param = ['N', 'P', "K"]

    custom_hover_text = {
        "NPK": {
            'N': 'Nitrogen (N)', 
            'P': 'Phosphorus (P)', 
            'K': 'Potassium (K)'
        },
        "Temperature": 'Temperature (°C)',
        "Humidity": 'Humidity (%)',
        "PH": 'pH Level',
        "EC": 'Electrical Conductivity (uS/cm)'
    }

    # Initialize the Plotly figure object
    fig = go.Figure()

    # If NPK is selected, plot all three (N, P, K) together
    if selected_feature == "NPK":
        for nutrient in ['N', 'P', "K"]:
            # Calculate boxplot statistics manually using pandas
            q1 = df[nutrient].quantile(0.25)
            q3 = df[nutrient].quantile(0.75)
            median = df[nutrient].median()
            min_val = df[nutrient].min()
            max_val = df[nutrient].max()
            mean_val = df[nutrient].mean()
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr
            lower_fence = q1 - 1.5 * iqr
            if lower_fence < 0:
                lower_fence = min_val

            # Add a boxplot for each nutrient
            fig.add_trace(go.Box(
                y=df[nutrient],
                name=custom_hover_text['NPK'][nutrient],  # Custom label (e.g., Nitrogen (N))
                boxmean=True,  # Display mean value as well
                hovertemplate=(
                    f"<b>Nutrient:</b> {custom_hover_text['NPK'][nutrient]}<br>"
                    f"<b>Max:</b> {max_val:.2f}<br>"  # Max value
                    f"<b>Upper Fence:</b> {upper_fence:.2f}<br>"  # Upper Fence
                    f"<b>Upper Quartile (Q3):</b> {q3:.2f}<br>"  # Q3
                    f"<b>Mean:</b> {mean_val:.2f}<br>"  # Mean value
                    f"<b>Median (Q2):</b> {median:.2f}<br>"  # Median
                    f"<b>Lower Quartile (Q1):</b> {q1:.2f}<br>"  # Q1
                    f"<b>Lower Fence:</b> {lower_fence:.2f}<br>"  # Lower Fence
                    f"<b>Min:</b> {min_val:.2f}<br>"  # Min value
                    "<extra></extra>"
                )
            ))

    else:
        # For Temperature or Humidity, plot a single boxplot
        q1 = df[param].quantile(0.25)
        q3 = df[param].quantile(0.75)
        median = df[param].median()
        min_val = df[param].min()
        max_val = df[param].max()
        mean_val = df[param].mean()
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        lower_fence = q1 - 1.5 * iqr
        if lower_fence < 0:
            lower_fence = min_val

        fig.add_trace(go.Box(
            y=df[param],
            name=custom_hover_text[selected_feature],  # Custom label (e.g., Temperature (°C))
            boxmean=True,  # Display mean value as well
            hovertemplate=(
                f"<b>{custom_hover_text[selected_feature]}:</b><br>"
                f"<b>Max:</b> {max_val:.2f}<br>"  # Max value
                f"<b>Upper Fence:</b> {upper_fence:.2f}<br>"  # Upper Fence
                f"<b>Upper Quartile (Q3):</b> {q3:.2f}<br>"  # Q3
                f"<b>Mean:</b> {mean_val:.2f}<br>"  # Mean value
                f"<b>Median (Q2):</b> {median:.2f}<br>"  # Median
                f"<b>Lower Quartile (Q1):</b> {q1:.2f}<br>"  # Q1
                f"<b>Lower Fence:</b> {lower_fence:.2f}<br>"  # Lower Fence
                f"<b>Min:</b> {min_val:.2f}<br>"  # Min value
                "<extra></extra>"
            )
        ))

    # Update layout to adjust aesthetics
    fig.update_layout(
        title=f"Distribution of {selected_feature} Data",
        yaxis_title=f"{selected_feature}",   # Y-axis title
        boxmode='group',                  # Group boxes together
        width=800,                        # Set width
        height=500                        # Set height
    )

    # Display the box plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# -- DASHBOARD PAGE --
st.markdown('<p class="dashboard_title">Soil Dashboard</p>', unsafe_allow_html = True)
tab1, tab2, tab3 = st.tabs(["Analytics", "Data Overview", "Advanced Analytics"])

if 'uploaded_df' in st.session_state:
    df = st.session_state['uploaded_df']  # Get the uploaded DataFrame

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
        st.write("")


    # -- DATA OVERVIEW TAB --
    with tab2:
        # Placeholder for displaying the total number of records dynamically
        records_placeholder = st.empty()
        records_placeholder.markdown(f'<p class="records_text">Total number of soil records:<p class="records">{len(st.session_state.get("df2_filtered", df))} 🌱</p>', unsafe_allow_html=True)

        cols = st.columns((4, 1), gap="medium")

        with cols[0].container():
            df_placeholder = st.empty()

            # Use the filtered DataFrame in session_state if available, otherwise use the original df2
            df2 = df.drop(columns=['datetime'])  # Different DataFrame for tab2
            df2_filtered = st.session_state.get('df2_filtered', df2)  # Default to df2 if no filtered DataFrame is set
            df_placeholder.dataframe(df2_filtered, use_container_width=True)

        with cols[1]:
            st.markdown(f'<p class="params_text">Parameter Filters</p>', unsafe_allow_html=True)
            st.divider()

            # If df2_filtered is not in session_state, initialize it with the original df2
            if 'df2_filtered' not in st.session_state:
                st.session_state.df2_filtered = df2.copy()  # Store the unfiltered DataFrame for tab2
                st.session_state.df2_filtered_copy = df2.copy()  # Backup for reset

            # Convert date and time to datetime format
            df2['Date'] = pd.to_datetime(df2['Date'], format='%d/%m/%Y').dt.date
            df2['Time'] = pd.to_datetime(df2['Time'], format='%H:%M:%S').dt.time

            # Capture slider inputs but don't apply filtering immediately
            # Date filter
            date_range = st.date_input(
                "Select a date range:",
                min_value=df2['Date'].min(),
                max_value=df2['Date'].max(),
                value=(df2['Date'].min(), df2['Date'].max())
            )

            # Time filter
            start_time, end_time = st.slider(
                "Select a time range:",
                value=(df2['Time'].min(), df2['Time'].max()),
                format="HH:mm:ss"
            )

            # Numeric filters
            N_min, N_max = st.slider(
                "Nitrogen (N) Range",
                min_value=float(df2['N'].min()), 
                max_value=float(df2['N'].max()), 
                value=(float(df2['N'].min()), float(df2['N'].max()))
            )

            P_min, P_max = st.slider(
                "Phosphorus (P) Range",
                min_value=float(df2['P'].min()), 
                max_value=float(df2['P'].max()), 
                value=(float(df2['P'].min()), float(df2['P'].max()))
            )

            K_min, K_max = st.slider(
                "Potassium (K) Range",
                min_value=float(df2['K'].min()), 
                max_value=float(df2['K'].max()), 
                value=(float(df2['K'].min()), float(df2['K'].max()))
            )

            Temp_min, Temp_max = st.slider(
                "Temperature Range",
                min_value=float(df2['Temperature'].min()), 
                max_value=float(df2['Temperature'].max()), 
                value=(float(df2['Temperature'].min()), float(df2['Temperature'].max()))
            )

            Humi_min, Humi_max = st.slider(
                "Humidity Range",
                min_value=float(df2['Humidity'].min()), 
                max_value=float(df2['Humidity'].max()), 
                value=(float(df2['Humidity'].min()), float(df2['Humidity'].max()))
            )

            EC_min, EC_max = st.slider(
                "Electrical Conductivity Range",
                min_value=float(df2['EC'].min()), 
                max_value=float(df2['EC'].max()), 
                value=(float(df2['EC'].min()), float(df2['EC'].max()))
            )

            PH_min, PH_max = st.slider(
                "pH level Range",
                min_value=float(df2['PH'].min()), 
                max_value=float(df2['PH'].max()), 
                value=(float(df2['PH'].min()), float(df2['PH'].max()))
            )

            # Filtering only happens when the button is pressed
            if st.button("Submit Filter"):
                # Apply all filters when the button is pressed
                df2_filtered = df2[
                    (df2['Date'] >= date_range[0]) & (df2['Date'] <= date_range[1]) &
                    (df2['Time'] >= start_time) & (df2['Time'] <= end_time) &
                    (df2['N'] >= N_min) & (df2['N'] <= N_max) &
                    (df2['P'] >= P_min) & (df2['P'] <= P_max) &
                    (df2['K'] >= K_min) & (df2['K'] <= K_max) &
                    (df2['Temperature'] >= Temp_min) & (df2['Temperature'] <= Temp_max) &
                    (df2['Humidity'] >= Humi_min) & (df2['Humidity'] <= Humi_max) &
                    (df2['EC'] >= EC_min) & (df2['EC'] <= EC_max) &
                    (df2['PH'] >= PH_min) & (df2['PH'] <= PH_max)
                ]
                st.session_state.df2_filtered = df2_filtered.copy()  # Update the filtered DataFrame for tab2
                df_placeholder.dataframe(df2_filtered, use_container_width=True)  # Display the filtered DataFrame
                records_placeholder.markdown(f'<p class="records_text">Total number of soil records:<p class="records">{len(df2_filtered)} 🌱</p>', unsafe_allow_html=True)

            # Add a Reset button to reset the DataFrame to its original state
            if st.button("Reset Filter"):
                st.session_state.df2_filtered = st.session_state.df2_filtered_copy.copy()  # Reset to the original DataFrame
                df2_filtered = st.session_state.df2_filtered  # Update the filtered DataFrame for tab2
                df_placeholder.dataframe(df2_filtered, use_container_width=True)  # Display the original DataFrame
                records_placeholder.markdown(f'<p class="records_text">Total number of soil records:<p class="records">{len(df2_filtered)} 🌱</p>', unsafe_allow_html=True)

                


    # -- ADVANCED ANALYTICS TAB --
    with tab3:
        st.caption("Analytics based on selected dataset:")
        st.success(selected_feature)
        cols = st.columns(2, gap="medium")
        with cols[0]:
            # Distribution of data
            display_boxplot(filtered_df, selected_feature)

        with cols[1]:
            # Calculate correlation matrix
            corr_matrix = filtered_df.drop(columns=['datetime','Date','Time']).corr()

            # Create a heatmap using Plotly
            fig = px.imshow(corr_matrix, 
                            text_auto=True, 
                            aspect="auto", 
                            color_continuous_scale="Viridis", 
                            title="Correlation Map of Soil Properties")
            
            fig.update_traces(
                hovertemplate=(
                    "<b>Variable 1:</b> %{x}<br>"  # Row (variable 1)
                    "<b>Variable 2:</b> %{y}<br>"  # Column (variable 2)
                    "<b>Correlation Value:</b> %{z:.2f}<br>"  # Correlation value (z)
                    "<extra></extra>"
                )
            )

            # Display the heatmap in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        display_scatterplot(filtered_df, selected_feature)
else:
    st.write("Please upload a CSV file in the Home page to proceed.")
