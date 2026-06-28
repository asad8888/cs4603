# %%
import streamlit as st
import pandas as pd

"""
Brief Streamlit tutorial app.

Run:
    uv pip install streamlit pandas
    streamlit run streamlit_tutorial.py
"""


# 1) Page setup
st.set_page_config(page_title="Streamlit Quick Tutorial", page_icon="🚀", layout="centered")
st.title("🚀 Streamlit Quick Tutorial")
st.write("This app shows the core Streamlit patterns in one file.")

# 2) Inputs
st.header("1) User Input")
name = st.text_input("What is your name?", value="Student")
level = st.selectbox("Pick your experience level:", ["Beginner", "Intermediate", "Advanced"])
hours = st.slider("How many hours/week can you practice?", min_value=1, max_value=20, value=5)

# 3) React to input
st.header("2) Dynamic Output")
st.success(f"Hi {name}! Level: {level}. Suggested weekly practice: {hours} hours.")

# 4) Display data
st.header("3) Data Table + Chart")
df = pd.DataFrame(
    {
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "study_hours": [1, 1, 2, 1, 2],
    }
)
st.dataframe(df, use_container_width=True)
st.line_chart(df.set_index("day"))

# 5) Session state (persist values across reruns)
st.header("4) Session State")
if "click_count" not in st.session_state:
    st.session_state.click_count = 0

if st.button("Click me"):
    st.session_state.click_count += 1

st.write(f"Button clicks: {st.session_state.click_count}")

# 6) Layout
st.header("5) Simple Layout")
col1, col2 = st.columns(2)
with col1:
    st.metric("Focus Score", 78, "+3")
with col2:
    st.metric("Consistency", "4/5 days", "+1 day")

st.caption("Next step: add file upload, API calls, or model inference.")


