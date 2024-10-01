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
tst_page = st.Page(
    page="views/tst.py",
    title="test",
    icon=":material/smart_toy:"
)


# -- NAVIGATION SETUP --
pg = st.navigation(pages=[home_page, dashboard_page, insights_page, chatbot_page, tst_page])

# -- SHARED ON ALL PAGES --
st.logo("assets/logo.png")
st.sidebar.text("Made by MSSD13")

pg.run()

