# app.py
import streamlit as st
from PIL import Image
import numpy as np

# local modules (must be in same folder)
from predict import predict, model as loaded_model  # predict returns preds & img_array
from gradcam import generate_gradcam



st.markdown("""
<style>
img:hover{
    transform: scale(1.03);
    transition: 0.3s ease;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# ---- BANNER SECTION ----
#banner = Image.open("banner.jpg")
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent
banner = Image.open(APP_DIR / "banner.jpg")


# Make it nicely span the page width
st.image(banner, use_container_width=True)

st.title("🌿 Plant Disease Detector Prototype")
st.write("Upload a tomato leaf image to get a disease prediction and Grad‑CAM heatmap.")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        raise

    # show original (centered in the wide layout columns)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img.resize((300, 300)), caption="Uploaded Image")

    # Predict: returns predicted_class, confidence, preds (raw vector), img_array (preprocessed)
    with st.spinner("Predicting..."):
        predicted_class, confidence, preds, img_array = predict(img)

    st.subheader("Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Generate Grad-CAM
    with st.spinner("Generating Grad‑CAM..."):
        class_index = int(np.argmax(preds))
        # pass the preprocessed img_array and the original PIL for overlay
        heatmap_overlay = generate_gradcam(loaded_model, img_array, class_index=class_index, original_image_pil=img)

    st.subheader("Grad‑CAM Heatmap")
    st.write(
        "This heatmap highlights the areas of the leaf that the model considered most important "
        "when predicting the disease. Bright regions indicate where the model 'looked' to make its decision."
    )
    colA, colB = st.columns(2)
    with colA:
        st.image(img.resize((350, 350)), caption="Original Image")
    with colB:
        st.image(heatmap_overlay.resize((350, 350)), caption="Grad‑CAM Heatmap (overlay)")
# --- Disease Dataset Exploration Section ---
    st.subheader("Disease Dataset Exploration")

    from pathlib import Path
    from PIL import Image
    import io, base64

    # Use predicted_class directly
    class_name = predicted_class

    # Path to the class folder
    class_folder = Path("samples") / class_name

    # Get images (jpg/png, case-insensitive)
    sample_images = [f for f in class_folder.iterdir() if f.suffix.lower() in [".jpg", ".png"]] if class_folder.exists() else []

    if sample_images:
        st.markdown(f"The images below show how **{predicted_class}** can appear in different forms on plants, reflecting natural variation in symptoms due to plant age, environment, and disease stage:")

        # 2 rows x 3 columns layout
        rows = 2
        cols = 3
        for r in range(rows):
            cols_streamlit = st.columns(cols)
            for c in range(cols):
                idx = r * cols + c
                if idx < len(sample_images):
                    img_sample = Image.open(sample_images[idx])

                    # Convert PIL image to base64 for HTML display
                    buffer = io.BytesIO()
                    img_sample.save(buffer, format="PNG")
                    img_str = base64.b64encode(buffer.getvalue()).decode()

                    # HTML with styling
                    html_code = f"""
                    <div style="
                        background-color: transparent;
                        border: 2px solid green; 
                        border-radius: 10px;
                        padding: 1px;
                        box-shadow: 3px 3px 8px rgba(180, 220, 180, 0.35);
                        text-align:center;
                        background-color:#1a1a1a;
                        transition: transform 0.3s ease;
                    " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                        <img src="data:image/png;base64,{img_str}" width="250"/>
                        <p style="color:white; font-size:12px; margin-top:5px;">{sample_images[idx].name}</p>
                    </div>
                    """
                    cols_streamlit[c].markdown(html_code, unsafe_allow_html=True)
    else:
        st.warning(f"No sample images found for the predicted disease: {class_name}.")


    # ------------------ GEMINI DISEASE INFORMATION-----------------
    from openai import OpenAI
    import json
    from disease_info_fallback import disease_info_fallback  # your fallback dictionary

    st.markdown("""
    <style>
    .disease-card {
        background-color: #f9fafb;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
        min-height: 280px;
        margin-bottom: 26px;
    }
    .disease-card h4 {
        margin-bottom: 14px;
        color: #1b5e20;
        font-weight: 600;
    }
    .disease-card p {
        margin-bottom: 8px;
        line-height: 1.55;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("📖 Disease Information")

    # ------------------ GEMINI CLIENT ------------------
    client = OpenAI(
        api_key="YOUR_GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # ------------------ PROMPT ------------------
    prompt = f"""
    You are a plant disease expert. I will give you the name of a plant disease.
    Return ONLY valid JSON with exactly these keys: symptoms, causes, prevention, treatment.

    Rules:
    1. Each key contains a maximum of 3 short, concise sentences.
    2. Each value must be a list of short sentences.
    3. No Markdown, bold, underscores, or code fences.
    4. Do not add any extra text outside the JSON.

    Disease name: "{predicted_class}"
    """

    disease_info = None

    # ------------------ API CALL WITH SILENT FALLBACK ------------------
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful plant disease expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )

        content = response.choices[0].message.content.strip()

        # Remove code fences if present
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1])

        disease_info = json.loads(content)

        # Normalize structure
        for key in ["symptoms", "causes", "prevention", "treatment"]:
            if key in disease_info and isinstance(disease_info[key], str):
                disease_info[key] = [disease_info[key]]
            if key not in disease_info:
                disease_info[key] = []

    except Exception:
        # SILENT fallback to offline dictionary
        disease_info = disease_info_fallback.get(predicted_class, 
                                                {"symptoms": [], "causes": [], "prevention": [], "treatment": []})

    # ------------------ DISPLAY 2x2 FIXED CARDS ------------------
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            f"""
            <div class="disease-card">
                <h4>🦠 Symptoms</h4>
                {''.join([f'<p>• {s}</p>' for s in disease_info["symptoms"]])}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="disease-card">
                <h4>⚠️ Causes</h4>
                {''.join([f'<p>• {c}</p>' for c in disease_info["causes"]])}
            </div>
            """,
            unsafe_allow_html=True
        )

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown(
            f"""
            <div class="disease-card">
                <h4>🌱 Prevention</h4>
                {''.join([f'<p>• {p}</p>' for p in disease_info["prevention"]])}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="disease-card">
                <h4>💊 Treatment</h4>
                {''.join([f'<p>• {t}</p>' for t in disease_info["treatment"]])}
            </div>
            """,
            unsafe_allow_html=True
        )



st.markdown("""
<hr>
<p style='text-align:center; color:gray; font-size:14px'>
Prototype by Dzifa Esi Dotse, Enock Daniel Essoun • Powered by TensorFlow + Streamlit
</p>
""", unsafe_allow_html=True)
