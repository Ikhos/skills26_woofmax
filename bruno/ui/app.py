import streamlit as st
import requests

st.set_page_config(page_title="BRUNO", page_icon="🐶", layout="centered")

st.title("🐶 BRUNO")
st.caption("Baymax-style scanning pipeline (Mac dev build).")

API_URL = st.text_input("Backend URL", "http://127.0.0.1:8000")

st.subheader("Upload an image to scan")
uploaded = st.file_uploader("Choose a JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Input image", use_container_width=True)

    if st.button("Scan"):
        with st.spinner("Scanning..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            r = requests.post(f"{API_URL}/scan", files=files, timeout=60)
        st.subheader("Scan results (raw JSON)")
        st.json(r.json())
else:
    st.info("Upload an image to start.")