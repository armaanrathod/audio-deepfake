import streamlit as st
import requests
import tempfile
import os

BACKEND_URL = "http://localhost:5000/predict"

st.set_page_config(page_title="Audio Deepfake Detector", layout="centered")

st.title("Audio Deepfake Detector")
st.caption("Upload any voice clip — real or AI generated. Works on clips of any length.")

uploaded = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "flac", "m4a", "ogg"]
)

if uploaded:
    st.audio(uploaded)

    if st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):

            ext = uploaded.name.rsplit(".", 1)[-1].lower()

            # Write to temp file then close immediately — Windows needs this
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            # tmp is now fully closed before we do anything else

            try:
                # Open separately for the request — also closed by with block
                with open(tmp_path, "rb") as f:
                    resp = requests.post(
                        BACKEND_URL,
                        files={"file": f},
                        timeout=120
                    )
                # file handle closed here — safe to delete on Windows

                if not resp.ok:
                    st.error(f"Backend error {resp.status_code}: {resp.text[:200]}")
                    st.stop()

                result = resp.json()

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    label    = result["prediction"].upper()
                    conf     = result["confidence"]
                    slabs    = result["slabs_analysed"]
                    duration = result["audio_duration_seconds"]

                    if label == "FAKE":
                        st.error("FAKE — AI-generated voice")
                    else:
                        st.success("REAL — Human voice")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Confidence", f"{conf * 100:.1f}%")
                    col2.metric("Duration", f"{duration}s")
                    col3.metric("Slabs analysed", slabs)

                    st.progress(conf)
                    st.caption(
                        f"Audio was split into {slabs} × 4-second segment(s) and averaged."
                    )

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure app.py is running on port 5000.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)