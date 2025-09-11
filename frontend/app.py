import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="IDS Demo", layout="wide")

st.title("Intrusion Detection System - Demo")
st.write("لطفاً فایل CSV دیتاست خود را آپلود کنید تا رکوردها پردازش شوند.")

api_url = st.text_input("API URL", value="http://127.0.0.1:8000")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, encoding="latin1")
    st.write("Preview:")
    st.dataframe(df.head(10))

    if st.button("Start"):
        results = []
        with st.spinner("Processing records..."):
            for idx, row in df.iterrows():
                try:
                    resp = requests.post(f"{api_url}/predict-json", json=row.to_dict(), timeout=30)
                    r = resp.json()
                    results.append({
                        **row.to_dict(),
                        "pred_label": r.get("label"),
                        "confidence": r.get("confidence")
                    })
                except Exception as e:
                    results.append({**row.to_dict(), "pred_label": "error", "confidence": None})

        res_df = pd.DataFrame(results)
        def highlight(row):
            color = "#d4f7d4" if row["pred_label"] == "normal" else ("#f7d4d4" if row["pred_label"] == "attack" else "#fff1d4")
            return [f"background-color: {color}"] * len(row)

        st.write("Results:")
        st.dataframe(res_df.style.apply(highlight, axis=1))



