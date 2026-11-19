import pandas as pd
import streamlit as st

from src.grant_parser import grant_from_text, parse_grant_file
from src.pytorch_classifier import PairingGrantClassifier, organization_text
from src.run_hidden_test import PROFILES

DESCRIPTION, _, FACTS = PROFILES["ardook_data"]
ORGANIZATION = organization_text(DESCRIPTION, FACTS)

st.set_page_config(page_title="Ardook", layout="centered")


@st.cache_resource(show_spinner=False)
def load_classifier() -> PairingGrantClassifier:
    return PairingGrantClassifier()


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

if grants is not None:
    with st.spinner("Checking"):
        predictions = load_classifier().predict(ORGANIZATION, grants.to_dict("records"))
    st.space("medium")

    if mode == "Paste grant":
        result = predictions[0]
        st.subheader(result["label"])
        st.caption(f"{result['match_probability']:.0%} match")
    else:
        table = pd.DataFrame({
            "Grant": grants["program"],
            "Result": [p["label"] for p in predictions],
            "Match": [f"{p['match_probability']:.0%}" for p in predictions],
        })
        st.table(table, hide_index=True, border="horizontal", alt="Grant results")
        csv = pd.DataFrame({"grant_id": grants["grant_id"],
                            "label": [p["label"] for p in predictions],
                            "reason": [p["reason"] for p in predictions]}).to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="hidden_test_outputs.csv",
                           mime="text/csv", type="tertiary", icon=":material/download:")
