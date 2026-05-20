import streamlit as st
import requests
import tempfile
import os

BACKEND_URL = "http://localhost:5000/predict"

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Audio Deepfake Detector",
    page_icon="🎙️",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 2rem;
}

.upload-box {
    padding: 1rem;
    border-radius: 16px;
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 14px;
    text-align: center;
    backdrop-filter: blur(8px);
}

.result-real {
    padding: 1rem;
    border-radius: 14px;
    background: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    color: #bbf7d0;
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
}

.result-fake {
    padding: 1rem;
    border-radius: 14px;
    background: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #fecaca;
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    font-size: 1rem;
    font-weight: 600;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="main-title">🎙️ Audio Deepfake Detector</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Upload any voice clip — real or AI generated</p>',
    unsafe_allow_html=True
)

# ---------- FILE UPLOAD ----------
uploaded = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "flac", "m4a", "ogg"]
)

if uploaded:

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    st.audio(uploaded)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    if st.button("Analyze Audio"):

        with st.spinner("Analyzing audio..."):

            ext = uploaded.name.rsplit(".", 1)[-1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:

                with open(tmp_path, "rb") as f:
                    resp = requests.post(
                        BACKEND_URL,
                        files={"file": f},
                        timeout=120
                    )

                if not resp.ok:
                    st.error(f"Backend error {resp.status_code}: {resp.text[:200]}")
                    st.stop()

                result = resp.json()

                if "error" in result:
                    st.error(f"Error: {result['error']}")

                else:

                    label = result["prediction"].upper()
                    conf = result["confidence"]
                    slabs = result["slabs_analysed"]
                    duration = result["audio_duration_seconds"]

                    st.write("")

                    # ---------- RESULT ----------
                    if label == "FAKE":
                        st.markdown(
                            '<div class="result-fake">🚨 FAKE — AI Generated Voice</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="result-real">✅ REAL — Human Voice</div>',
                            unsafe_allow_html=True
                        )

                    st.write("")

                    # ---------- METRICS ----------
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h4>Confidence</h4>
                                <h2>{conf * 100:.1f}%</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h4>Duration</h4>
                                <h2>{duration}s</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with col3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h4>Segments</h4>
                                <h2>{slabs}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.write("")

                    # ---------- CONFIDENCE BAR ----------
                    st.progress(conf)

                    st.caption(
                        f"Audio split into {slabs} × 4-second segments and averaged for prediction."
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to backend. Make sure app.py is running on port 5000."
                )

            except Exception as e:
                st.error(f"Unexpected error: {e}")

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
