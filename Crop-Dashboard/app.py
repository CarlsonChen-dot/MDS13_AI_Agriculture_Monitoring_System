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

st.sidebar.text("""Made by MDS13
@ Monash University Malaysia""")

# """)
names_code = """Chen Kai Xuan 32712960
Chua Qiu Xia 33207089
Richelle Yong 33203547"""

st.sidebar.code(names_code, language="python")
pg.run()


