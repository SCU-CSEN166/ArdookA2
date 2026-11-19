import pandas as pd
import streamlit as st

from src.app_predict import ModelUnavailable, load_model, predict, results_csv
from src.grant_parser import grant_from_text, parse_grant_file

st.set_page_config(page_title="Ardook", layout="centered")


@st.cache_resource(show_spinner=False)
def load_classifier():
    return load_model()


def result_text(match: bool) -> str:
    return "Match" if match else "Not a match"


mode = st.segmented_control("Input", ["Upload table", "Paste grant"], default="Upload table",
                            required=True, label_visibility="collapsed")

grants = None
try:
    if mode == "Upload table":
        upload = st.file_uploader("Grant table", type=["csv", "tsv", "xlsx"],
                                  label_visibility="collapsed")
        if upload is not None:
            grants = parse_grant_file(upload.getvalue(), upload.name)
    else:
        text = st.text_area("Grant", height=160, label_visibility="collapsed",
                            placeholder="Paste a grant description")
        if text.strip():
            grants = grant_from_text(text)
except ValueError as exc:
    st.error(str(exc))
except Exception as exc:  # unreadable or corrupt upload
    st.error(f"Could not read this file: {exc}. Upload a .csv, .tsv or .xlsx grant table.")

if grants is not None:
    with st.spinner("Checking"):
        try:
            model = load_classifier()
        except ModelUnavailable as exc:
            st.error(str(exc))
            st.stop()
        try:
            predictions = predict(model, grants)
        except ValueError as exc:
            st.error(f"Could not score these grants: {exc}")
            st.stop()
    st.space("medium")

    if mode == "Paste grant":
        result = predictions.iloc[0]
        st.subheader(result_text(result["match"]))
        st.caption(f"Model score {result['p_match']:.2f} · match at {model.threshold:.2f} or above")
    else:
        table = pd.DataFrame({
            "Grant": predictions["program"],
            "Result": predictions["match"].map(result_text),
            "Score": predictions["p_match"].map("{:.2f}".format),
        })
        st.table(table, hide_index=True, border="horizontal", alt="Grant results")
        st.caption(f"Model score from the saved match classifier · match at "
                   f"{model.threshold:.2f} or above")
        st.download_button("Download CSV", results_csv(predictions, model.threshold),
                           file_name="hidden_test_outputs.csv", mime="text/csv",
                           type="tertiary", icon=":material/download:")
