from pathlib import Path
import base64
import html

import numpy as np
import streamlit as st
from PIL import Image

import predict as predictor
from gradcam import generate_gradcam
from disease_info_fallback import disease_info_fallback

try:
    from disease_info_fallback import DISEASE_INFO_ALIASES
except ImportError:
    DISEASE_INFO_ALIASES = {
        "Cherry_(including_sour)___Powdery_mildew": "Cherry___Powdery_mildew",
        "Cherry_(including_sour)___healthy": "Cherry___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_": "Corn___Common_rust",
        "Corn_(maize)___Northern_Leaf_Blight": "Corn___Northern_Leaf_Blight",
        "Corn_(maize)___healthy": "Corn___healthy",
    }

st.set_page_config(page_title="LeafLens AI", page_icon="🌿", layout="wide")
APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = APP_DIR / "samples"
BANNER_PATH = APP_DIR / "banner.jpg"


def image_data_uri(path):
    """Return a local image as a browser-safe data URI."""
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


BANNER_DATA_URI = image_data_uri(BANNER_PATH)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def reset_analysis():
    st.session_state.uploader_key += 1
    st.rerun()


def lookup_name(class_name):
    return DISEASE_INFO_ALIASES.get(class_name, class_name)


def pretty_name(class_name):
    text = " ".join(class_name.replace("___", " - ").replace("_", " ").split())
    text = text.replace("Cercospora leaf spot Gray leaf spot", "Cercospora leaf spot / Gray leaf spot")
    text = text.replace("Spider mites Two-spotted spider mite", "Spider mites / Two-spotted spider mite")
    return text.replace("Haunglongbing", "Huanglongbing")


def get_sample_files(class_name, limit=6):
    candidates = [class_name]
    alias = lookup_name(class_name)
    if alias != class_name:
        candidates.append(alias)
    for candidate in candidates:
        folder = SAMPLES_DIR / candidate
        if folder.exists() and folder.is_dir():
            files = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if files:
                return files[:limit]
    return []


def info_card(icon, title, entries, delay):
    items = "".join(f"<li>{html.escape(str(item))}</li>" for item in entries)
    if not items:
        items = "<li>No information is currently available.</li>"
    return (
        f'<div class="info-card reveal" style="animation-delay:{delay}ms">'
        f'<h4><span class="card-icon">{icon}</span>{html.escape(title)}</h4>'
        f"<ul>{items}</ul></div>"
    )


def show_disease_information(class_name, display_name):
    info = disease_info_fallback.get(lookup_name(class_name))
    st.markdown('<div class="section-tag">Plant health guidance</div>', unsafe_allow_html=True)
    st.subheader("Disease intelligence")
    st.caption(f"General information associated with {display_name}.")
    if not info:
        st.markdown('<div class="empty reveal">No explanatory notes are available for this class.</div>', unsafe_allow_html=True)
        return
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(info_card("🦠", "Symptoms", info.get("symptoms", []), 40), unsafe_allow_html=True)
        st.markdown(info_card("🌱", "Prevention", info.get("prevention", []), 160), unsafe_allow_html=True)
    with right:
        st.markdown(info_card("⚠️", "Causes", info.get("causes", []), 100), unsafe_allow_html=True)
        st.markdown(info_card("💊", "Treatment", info.get("treatment", []), 220), unsafe_allow_html=True)


def show_samples(class_name, display_name):
    files = get_sample_files(class_name)
    st.markdown('<div class="section-tag">Visual reference</div>', unsafe_allow_html=True)
    st.subheader("Similar dataset examples")
    if not files:
        st.markdown(f'<div class="empty reveal">No reference samples are available for {html.escape(display_name)}.</div>', unsafe_allow_html=True)
        return
    columns = st.columns(3, gap="medium")
    for index, path in enumerate(files):
        try:
            columns[index % 3].image(
                Image.open(path).convert("RGB"),
                caption=f"Reference {index + 1}",
                use_container_width=True,
            )
        except Exception as exc:
            columns[index % 3].warning(f"Could not load {path.name}: {exc}")


# Permanent light theme
palette = {
    "page": "#f2faf5", "page2": "#fbfdfb", "card": "rgba(255,255,255,.92)",
    "card2": "rgba(245,252,247,.98)", "line": "rgba(24,105,63,.17)",
    "text": "#10271b", "muted": "#526b5c", "accent": "#168a50",
    "accent2": "#78a923", "hero1": "rgba(235,250,241,.99)",
    "hero2": "rgba(250,253,251,.99)", "shadow": "rgba(39,94,62,.13)",
    "soft": "rgba(25,130,70,.075)", "glow": "rgba(30,170,83,.22)",
}

st.markdown(
    f"""
<style>
:root{{--page:{palette['page']};--page2:{palette['page2']};--card:{palette['card']};--card2:{palette['card2']};--line:{palette['line']};--text:{palette['text']};--muted:{palette['muted']};--accent:{palette['accent']};--accent2:{palette['accent2']};--hero1:{palette['hero1']};--hero2:{palette['hero2']};--shadow:{palette['shadow']};--soft:{palette['soft']};--glow:{palette['glow']}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(22px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes floatOrb{{0%,100%{{transform:translate3d(0,0,0) scale(1)}}50%{{transform:translate3d(-20px,18px,0) scale(1.08)}}}}
@keyframes pulseDot{{0%,100%{{box-shadow:0 0 0 0 var(--glow)}}50%{{box-shadow:0 0 0 9px transparent}}}}
@keyframes shimmer{{0%{{background-position:200% center}}100%{{background-position:-200% center}}}}
@keyframes softPulse{{0%,100%{{transform:scale(1);filter:brightness(1)}}50%{{transform:scale(1.018);filter:brightness(1.05)}}}}
@keyframes progressGlow{{0%,100%{{filter:drop-shadow(0 0 2px var(--glow))}}50%{{filter:drop-shadow(0 0 7px var(--glow))}}}}
@keyframes heroPop{{0%{{opacity:0;transform:perspective(900px) rotateX(9deg) translateY(48px) scale(.96)}}70%{{opacity:1;transform:perspective(900px) rotateX(-1.5deg) translateY(-5px) scale(1.006)}}100%{{transform:perspective(900px) rotateX(0) translateY(0) scale(1)}}}}
@keyframes leafDrift{{0%{{opacity:0;transform:translate3d(0,34px,0) rotate(0deg) scale(.65)}}20%{{opacity:.7}}50%{{transform:translate3d(24px,-22px,0) rotate(140deg) scale(1)}}80%{{opacity:.55}}100%{{opacity:0;transform:translate3d(-12px,-88px,0) rotate(320deg) scale(.7)}}}}
@keyframes scanMove{{0%{{transform:translateY(-120%);opacity:0}}20%{{opacity:.35}}80%{{opacity:.18}}100%{{transform:translateY(820%);opacity:0}}}}
@keyframes uploaderGlow{{0%,100%{{box-shadow:0 16px 40px var(--shadow),0 0 0 0 transparent}}50%{{box-shadow:0 20px 55px var(--shadow),0 0 0 4px var(--glow)}}}}
@keyframes cardPop{{0%{{opacity:0;transform:scale(.86) translateY(28px)}}70%{{opacity:1;transform:scale(1.025) translateY(-4px)}}100%{{transform:scale(1) translateY(0)}}}}
@keyframes togglePulse{{0%,100%{{box-shadow:0 0 0 0 var(--glow)}}50%{{box-shadow:0 0 0 8px transparent}}}}
html,body,[class*=css]{{font-family:Inter,Segoe UI,sans-serif}}.stApp{{color:var(--text);background:radial-gradient(circle at 12% 7%,var(--glow),transparent 29%),radial-gradient(circle at 88% 16%,var(--soft),transparent 25%),linear-gradient(145deg,var(--page),var(--page2) 48%,var(--page));transition:background .45s ease,color .35s ease}}
[data-testid=stHeader]{{background:transparent}}.block-container{{max-width:1240px;padding-top:.8rem;padding-bottom:4rem}}h1,h2,h3,h4{{color:var(--text);letter-spacing:-.025em}}p,label,.stCaption{{color:var(--muted)}}
.hero{{position:relative;overflow:hidden;padding:44px;margin:8px 0 26px;border:1px solid var(--line);border-radius:30px;background:linear-gradient(120deg,var(--hero1),var(--hero2));box-shadow:0 24px 70px var(--shadow);animation:heroPop .9s cubic-bezier(.16,1,.3,1) both}}.hero:after{{content:'';position:absolute;width:390px;height:390px;right:-100px;top:-160px;border-radius:50%;background:radial-gradient(circle,var(--glow),transparent 68%);animation:floatOrb 8s ease-in-out infinite}}.hero:before{{content:'';position:absolute;inset:0;background:linear-gradient(110deg,transparent 20%,rgba(255,255,255,.06) 45%,transparent 70%);background-size:220% 100%;animation:shimmer 7s linear infinite;pointer-events:none}}
.eyebrow{{display:inline-flex;align-items:center;padding:7px 12px;border:1px solid var(--line);border-radius:999px;background:var(--soft);color:var(--accent);font-size:.76rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;position:relative;z-index:1}}.live-dot{{width:8px;height:8px;margin-right:9px;border-radius:50%;background:var(--accent);animation:pulseDot 2s ease-out infinite}}.hero-grid{{position:relative;z-index:1;display:grid;grid-template-columns:minmax(0,1.25fr) minmax(280px,.75fr);gap:34px;align-items:center}}.hero-copy{{min-width:0}}.hero-visual{{position:relative;min-height:300px;border:1px solid var(--line);border-radius:24px;overflow:hidden;background:var(--soft);box-shadow:0 22px 55px var(--shadow);transform:rotate(1.5deg);transition:transform .4s cubic-bezier(.2,.8,.2,1),box-shadow .4s ease}}.hero-visual:hover{{transform:rotate(0deg) translateY(-7px) scale(1.015);box-shadow:0 30px 70px var(--shadow)}}.hero-visual img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;animation:heroImageZoom 12s ease-in-out infinite alternate}}.hero-visual:after{{content:'';position:absolute;inset:0;background:linear-gradient(180deg,transparent 48%,rgba(5,42,22,.74) 100%)}}.visual-badge{{position:absolute;z-index:2;left:16px;bottom:16px;padding:9px 12px;border:1px solid rgba(255,255,255,.28);border-radius:999px;background:rgba(9,47,28,.68);backdrop-filter:blur(12px);color:#fff;font-size:.78rem;font-weight:800;letter-spacing:.04em}}@keyframes heroImageZoom{{from{{transform:scale(1.02)}}to{{transform:scale(1.11)}}}}.hero h1{{position:relative;z-index:1;margin:18px 0 12px;font-size:clamp(2.3rem,5vw,4.4rem);line-height:1.02;font-weight:800}}.hero h1 span{{background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent));background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 5s linear infinite}}.hero p,.chips{{position:relative;z-index:1}}.hero p{{max-width:730px;font-size:1.07rem;line-height:1.7}}.chips{{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}}.chip{{padding:9px 13px;border:1px solid var(--line);border-radius:13px;background:var(--soft);color:var(--text);font-size:.84rem;font-weight:600;transition:transform .2s ease,border-color .2s ease}}.chip:hover{{transform:translateY(-3px);border-color:var(--accent)}}
.section-tag{{margin:28px 0 10px;color:var(--accent);font-size:.78rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;animation:fadeUp .5s ease both}}.soft-card{{padding:22px;border:1px solid var(--line);border-radius:21px;background:var(--card);box-shadow:0 16px 40px var(--shadow);animation:fadeUp .55s ease both}}
[data-testid=stFileUploader]{{padding:14px;border:1px solid var(--line);border-radius:22px;background:var(--card);box-shadow:0 16px 40px var(--shadow);animation:fadeUp .65s .12s ease both,uploaderGlow 3.2s 1.1s ease-in-out infinite;transition:transform .2s ease,border-color .2s ease}}[data-testid=stFileUploader]:hover{{transform:translateY(-2px);border-color:var(--accent)}}[data-testid=stFileUploaderDropzone]{{min-height:150px;border:1px dashed var(--accent);border-radius:17px;background:var(--soft);animation:softPulse 5s ease-in-out infinite}}
.stButton>button{{min-height:47px;border:0;border-radius:14px;background:linear-gradient(100deg,var(--accent),var(--accent2));color:#052012;font-weight:800;box-shadow:0 12px 28px var(--shadow);transition:transform .18s ease,box-shadow .18s ease}}.stButton>button:hover{{color:#02110a;transform:translateY(-3px) scale(1.01);box-shadow:0 17px 35px var(--glow)}}.stButton>button:active{{transform:translateY(0) scale(.98)}}
[data-testid=stImage]{{animation:fadeUp .55s ease both}}[data-testid=stImage] img{{border:1px solid var(--line);border-radius:20px;box-shadow:0 16px 45px var(--shadow);transition:transform .35s cubic-bezier(.2,.8,.2,1),box-shadow .35s ease}}[data-testid=stImage] img:hover{{transform:translateY(-5px) scale(1.012);box-shadow:0 24px 60px var(--shadow)}}
.condition-card{{min-height:130px;padding:21px;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,var(--card),var(--card2));box-shadow:0 15px 38px var(--shadow);animation:cardPop .72s cubic-bezier(.16,1,.3,1) both;transition:transform .2s ease,border-color .2s ease;display:flex;flex-direction:column;justify-content:center}}.condition-card:hover{{transform:translateY(-4px);border-color:var(--accent)}}.condition-label{{color:var(--muted);font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.07em;margin-bottom:12px}}.condition-value{{color:var(--text);font-size:clamp(1.15rem,2.15vw,1.85rem);font-weight:800;line-height:1.18;white-space:normal;overflow-wrap:anywhere;word-break:normal}}.result-card-spacing{{width:100%;height:20px;display:block;clear:both}}[data-testid=stMetric]{{min-height:130px;padding:21px;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,var(--card),var(--card2));box-shadow:0 15px 38px var(--shadow);animation:cardPop .72s cubic-bezier(.16,1,.3,1) both;transition:transform .2s ease,border-color .2s ease}}[data-testid=stMetric]:hover{{transform:translateY(-4px);border-color:var(--accent)}}[data-testid=stMetricLabel]{{color:var(--muted);font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.07em}}[data-testid=stMetricValue]{{color:var(--text);font-size:clamp(1.2rem,2.6vw,2rem);font-weight:800}}
[data-testid=stAlert]{{border:1px solid var(--line);border-radius:15px;background:var(--card);color:var(--text);animation:fadeUp .45s ease both}}[data-testid=stProgress]>div>div{{background:linear-gradient(90deg,var(--accent),var(--accent2));animation:progressGlow 2.4s ease-in-out infinite}}.match-row{{display:flex;justify-content:space-between;gap:14px;margin:11px 0 6px;color:var(--text);font-weight:600;animation:fadeUp .45s ease both}}.rank{{display:inline-grid;place-items:center;width:25px;height:25px;margin-right:8px;border-radius:8px;background:var(--soft);color:var(--accent);font-size:.75rem;font-weight:800}}
.reveal{{opacity:0;animation:fadeUp .58s cubic-bezier(.2,.8,.2,1) forwards}}.info-card{{min-height:245px;padding:22px;margin-bottom:16px;border:1px solid var(--line);border-radius:20px;background:var(--card);box-shadow:0 12px 30px var(--shadow);transition:transform .25s ease,border-color .25s ease}}.info-card:hover{{transform:translateY(-5px);border-color:var(--accent)}}.info-card h4{{margin:0 0 14px;color:var(--text)}}.card-icon{{display:inline-grid;place-items:center;width:35px;height:35px;margin-right:8px;border-radius:11px;background:var(--soft)}}.info-card li{{margin:9px 0;color:var(--muted);line-height:1.55}}.empty{{padding:17px;border:1px solid var(--line);border-radius:15px;background:var(--card);color:var(--muted)}}hr{{margin:30px 0;border-color:var(--line)}}footer,#MainMenu{{visibility:hidden}}
.leaf-particle{{position:absolute;z-index:0;color:var(--accent);font-size:18px;filter:drop-shadow(0 0 10px var(--glow));pointer-events:none;animation:leafDrift 6.5s ease-in-out infinite}}.leaf-a{{right:12%;bottom:4%;animation-delay:0s}}.leaf-b{{right:25%;bottom:-2%;font-size:12px;animation-delay:1.6s}}.leaf-c{{right:6%;bottom:18%;font-size:24px;animation-delay:3.1s}}.leaf-d{{right:34%;bottom:3%;font-size:10px;animation-delay:4.5s}}.scan-line{{position:absolute;z-index:0;left:0;right:0;top:0;height:70px;background:linear-gradient(180deg,transparent,rgba(255,255,255,.06),transparent);animation:scanMove 7s 1s ease-in-out infinite;pointer-events:none}}
@media(max-width:860px){{.hero-grid{{grid-template-columns:1fr}}.hero-visual{{min-height:240px;transform:none}}}}@media(max-width:760px){{.block-container{{padding:1rem 1rem 3rem}}.hero{{padding:30px 24px;border-radius:24px}}.hero h1{{font-size:2.35rem}}}}
@media(prefers-reduced-motion:reduce){{*,*::before,*::after{{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}}}
</style>
""",
    unsafe_allow_html=True,
)

predictor.load_assets()

hero_visual = (
    f'<div class="hero-visual"><img src="{BANNER_DATA_URI}" alt="Plant disease detection banner"><div class="visual-badge">Visual crop intelligence</div></div>'
    if BANNER_DATA_URI
    else '<div class="hero-visual"><div class="visual-badge">Visual crop intelligence</div></div>'
)

st.markdown(
    f"""
<section class="hero">
<div class="leaf-particle leaf-a">◆</div>
<div class="leaf-particle leaf-b">◆</div>
<div class="leaf-particle leaf-c">◆</div>
<div class="leaf-particle leaf-d">◆</div>
<div class="scan-line"></div>
<div class="hero-grid">
<div class="hero-copy">
<div class="eyebrow"><span class="live-dot"></span>AI-powered crop intelligence</div>
<h1>See plant health<br><span>more clearly.</span></h1>
<p>Upload a leaf photograph for rapid disease classification, confidence scoring, explainable AI insights and practical plant-care guidance.</p>
<div class="chips"><div class="chip">38 diagnostic classes</div><div class="chip">EfficientNetB0 engine</div><div class="chip">Grad-CAM explainability</div></div>
</div>
{hero_visual}
</div>
</section>
""",
    unsafe_allow_html=True,
)

if not predictor.MODEL_AVAILABLE:
    st.error("The production model could not be loaded.")
    st.code(predictor.MODEL_LOAD_ERROR or "Unknown model-loading error")
    st.stop()

st.markdown('<div class="section-tag">Start a diagnosis</div>', unsafe_allow_html=True)
upload_column, action_column = st.columns([4, 1], vertical_alignment="bottom", gap="medium")
with upload_column:
    uploaded = st.file_uploader(
        "Drag and drop a clear leaf image",
        type=["jpg", "jpeg", "png"],
        key=f"leaf_{st.session_state.uploader_key}",
        help="Use a well-lit image containing one visible leaf.",
    )
with action_column:
    if uploaded is not None and st.button("↻ New analysis", type="primary", use_container_width=True, key="new_top"):
        reset_analysis()

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open image: {exc}")
        st.stop()

    try:
        with st.spinner("Analysing visual patterns..."):
            predicted, confidence, scores, image_array, top = predictor.predict(image, top_k=3)
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.stop()

    display_name = pretty_name(predicted)
    margin = top[0]["confidence"] - top[1]["confidence"] if len(top) > 1 else top[0]["confidence"]
    uncertain = confidence < 0.60 or margin < 0.15

    st.markdown('<div class="section-tag">Diagnostic result</div>', unsafe_allow_html=True)
    preview, result = st.columns([1.05, 1.35], gap="large")
    with preview:
        st.image(image, caption="Uploaded leaf", use_container_width=True)
    with result:
        metric_one, metric_two = st.columns(2, gap="medium")
        with metric_one:
            st.markdown(
                '<div class="condition-card">'
                '<div class="condition-label">Detected condition</div>'
                f'<div class="condition-value">{html.escape(display_name)}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        metric_two.metric("Model confidence", f"{confidence:.1%}")

        # Keep the status alert visually separate from result cards,
        # especially when a long condition name makes the card taller.
        st.markdown(
            '<div class="result-card-spacing"></div>',
            unsafe_allow_html=True,
        )

        if uncertain:
            st.warning("Low-confidence result. Try a brighter photograph with one leaf filling the frame.")
        else:
            st.success("Strong visual match detected by the model.")
        st.markdown("#### Alternative matches")
        for rank, item in enumerate(top, start=1):
            score = float(item["confidence"])
            st.markdown(
                f'<div class="match-row"><span><span class="rank">{rank}</span>{html.escape(pretty_name(item["class_name"]))}</span><strong>{score:.1%}</strong></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(score, 0.0), 1.0))

    st.markdown('<div class="section-tag">Explainable AI</div>', unsafe_allow_html=True)
    st.subheader("What influenced the prediction")
    st.caption("Warmer areas indicate regions that contributed more strongly to the selected class.")
    try:
        with st.spinner("Building attention map..."):
            overlay = generate_gradcam(
                predictor.model,
                image_array,
                class_index=int(np.argmax(scores)),
                original_image_pil=image,
            )
        heat_one, heat_two = st.columns(2, gap="large")
        heat_one.image(image, caption="Original image", use_container_width=True)
        heat_two.image(overlay, caption="Grad-CAM attention map", use_container_width=True)
    except Exception as exc:
        st.warning(f"Prediction succeeded, but Grad-CAM could not be generated: {exc}")

    st.divider()
    show_disease_information(predicted, display_name)
    st.divider()
    show_samples(predicted, display_name)
    st.divider()

    _, bottom_button, _ = st.columns([1, 1.4, 1])
    with bottom_button:
        if st.button("↻ Analyse another leaf", type="primary", use_container_width=True, key="new_bottom"):
            reset_analysis()
else:
    st.markdown(
        '<div class="soft-card"><strong>Photography tips</strong>'
        '<p style="margin:9px 0 0">Use one leaf, natural lighting and a simple background. '
        'Keep the leaf sharp, centred and large enough for symptoms to be visible.</p></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center;margin-top:36px;color:var(--muted);font-size:.82rem">'
    'LeafLens AI · TensorFlow · EfficientNetB0 · Grad-CAM<br>'
    'Results are advisory and should be confirmed by a qualified agricultural professional.'
    '</div>',
    unsafe_allow_html=True,
)
