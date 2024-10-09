import streamlit as st 

# -- PAGE SETUP --
st.set_page_config(page_title="My App",layout='wide')
home_page = st.Page(
    page="views/home.py",
    title="Home",
    icon=":material/home:",
    default=True
)
dashboard_page = st.Page(
    page="views/dashboard.py",
    title="Dashboard",
    icon=":material/bar_chart:"
)
insights_page = st.Page(
    page="views/insights.py",
    title="Crop Insights",
    icon=":material/search:"
)
chatbot_page = st.Page(
    page="views/chatbot.py",
    title="Chat bot",
    icon=":material/smart_toy:"
)

# -- NAVIGATION SETUP --
pg = st.navigation(pages=[home_page, dashboard_page, insights_page, chatbot_page])

# -- SHARED ON ALL PAGES --
logo_url = "https://i.imgur.com/NYTwH3h.png"
st.logo(logo_url)
#st.sidebar.text("Made by MDS13/nChen Kai Xuan 32712960/nChua Qiu Xia 33207089/nRichelle Yong 33203547\nMonash University")

st.sidebar.text("""
Made by MDS13 (MONASH UNIVERSITY)
Chen Kai Xuan 32712960
Chua Qiu Xia 33207089
Richelle Yong 33203547

""")


# Adding names with center alignment in the sidebar
# st.sidebar.markdown(
#     "<p class='centered-text'>Made by MDS13<br>Chen Kai Xuan 32712960<br>Chua Qiu Xia 33207089<br>Richelle Yong 33203547<br>Monash University</p>",
#     unsafe_allow_html=True
# )

pg.run()

