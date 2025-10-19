# app.py — RPS Vision Dashboard (Futuristic • Gradient • Poppins)
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from collections import Counter

st.set_page_config(
    page_title="Rock–Paper–Scissors (RPS) Vision Dashboard",
    page_icon="🧠",
    layout="wide",
)

# =========================
# THEME (gradient + Poppins + futuristic network)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap');

:root{
  --bg1:#010030; --bg2:#160078; --bg3:#7226FF;
  --panel:#12122A; --panel-2:#1A1A34;
  --text:#FFFFFF; --muted:#BBC0E6;
}

/* Typography */
* { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
h1{ font-weight:800; line-height:1.12; }
h2,h3,h4{ font-weight:700; }
p,li,div,span,label{ font-weight:400; color:var(--text); }

/* Hide default header, widen container, naikkan posisi utama */
header[data-testid="stHeader"]{ display:none; }
.block-container{
  padding-top:0.1rem!important;  /* sebelumnya 3.2rem */
  padding-bottom:2rem;
  max-width:1300px;
}

/* Tambahkan sedikit perataan vertikal agar ikon sejajar */
.st-emotion-cache-ocqkz7, .st-emotion-cache-1y4p8pa{
  align-items:flex-start !important;
}


/* Futuristic gradient + network grid */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1000px 600px at 15% -10%, rgba(114,38,255,.28), transparent 65%),
    radial-gradient(900px 500px at 90% 10%, rgba(1,0,48,.30), transparent 60%),
    linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%) fixed;
}
[data-testid="stAppViewContainer"]::before{
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.25;
  background:
    linear-gradient(to right, rgba(255,255,255,.06) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255,255,255,.06) 1px, transparent 1px);
  background-size: 60px 60px, 60px 60px;
}
[data-testid="stAppViewContainer"]::after{
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.12;
  background:
    radial-gradient(3px 3px at 20% 30%, #fff, transparent 40%),
    radial-gradient(3px 3px at 70% 20%, #fff, transparent 40%),
    radial-gradient(3px 3px at 85% 65%, #fff, transparent 40%);
}

/* Cards (glass + subtle neon) */
.card{
  position:relative;
  background:
    linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,0)) padding-box,
    linear-gradient(90deg, rgba(114,38,255,.35), rgba(1,0,48,.35)) border-box;
  border:1px solid transparent; border-radius:18px; padding:22px 22px;
  box-shadow: 0 16px 44px rgba(0,0,0,.42);
  transition: box-shadow .25s ease, transform .25s ease;
}
.card:hover{ box-shadow:0 26px 60px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,.06) inset; transform: translateY(-1px); }
.card-title{ font-weight:700; font-size:1.35rem; margin-bottom:.7rem; color:#fff; }
.caption{ color: var(--muted); font-size:1rem; }

/* Tabs — white, non-bold */
.stTabs [role="tablist"]{ gap:1rem; }
.stTabs [role="tab"]{ color:#FFFFFF !important; font-weight:400; border-bottom:2px solid transparent; }
.stTabs [role="tab"][aria-selected="true"]{
  border-bottom:2px solid; border-image: linear-gradient(90deg,#010030,#7226FF) 1;
}

/* File uploader text color */
[data-testid="stFileUploader"] section div{ color:#D9DCF6 !important; }

/* Progress bars (classification & evaluation) */
.prog{ width:100%; height:12px; border-radius:999px; background:#23234a; overflow:hidden; }
.prog > span{ display:block; height:100%; width:0%; background:linear-gradient(90deg,#160078,#7226FF); animation: loadWidth 1s ease-out forwards; }
@keyframes loadWidth { from{ width:0% } to{ width:var(--w,0%) } }
.prog-wrap{ display:flex; align-items:center; gap:.8rem; margin:.55rem 0; }
.prog-wrap .lbl{ min-width:160px; font-weight:700; font-size:1.02rem; color:#fff; }
.prog-wrap .val{ width:78px; text-align:right; color:#fff; font-weight:700; font-variant-numeric: tabular-nums; }

/* Dataset counter icons */
.icon-bubble{ width:86px; height:86px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  border:2px solid rgba(255,255,255,.85); box-shadow:0 0 18px rgba(255,255,255,.25), inset 0 0 10px rgba(255,255,255,.12);}
.icon-bubble svg{ width:60px; height:60px; }

/* Architecture flow (aligned perfectly) */
.flow{ position:relative; padding-left:46px; }
.flow:before{ content:""; position:absolute; left:26px; top:6px; bottom:6px; width:4px; background:linear-gradient(#160078,#7226FF); border-radius:4px; }
.flow .node{ position:relative; margin:18px 0; padding-left:0; color:#fff; font-weight:700; font-size:1.05rem;}
.flow .node:before{ content:""; position:absolute; left:-36px; top:2px; width:22px; height:22px; border-radius:50%; border:3px solid rgba(255,255,255,.92); background:rgba(255,255,255,.12); box-shadow:0 0 8px rgba(255,255,255,.35); }

/* Big result title */
.big-result{ font-size:2.2rem; font-weight:800; letter-spacing:.3px; margin:.6rem 0 0 0; color:#fff; }

/* Header right image (no box) */
.header-rps-img{ width:100%; max-width:360px; height:auto;
  filter: drop-shadow(0 0 18px rgba(255,255,255,.28)) drop-shadow(0 0 6px rgba(255,255,255,.25)); }

/* Force select label & generic labels to white */
label, .stSelectbox label{ color:#FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Anisa Nabila_Laporan 4.pt")                    # YOLOv8 detector
    clf  = tf.keras.models.load_model("model/Anisa Nabila_Laporan 2.h5")  # CNN classifier
    return yolo, clf

yolo_model, classifier = load_models()

# =========================
# ====== HERO + NAV =======
# =========================

# ---------- CSS tambahan untuk hero, pill tabs, hint ----------
st.markdown("""
<style>
/* Profile mini di kiri atas */
.profile-mini{display:flex;align-items:center;gap:12px;margin:6px 0 18px 4px}
.profile-mini .ava{width:34px;height:34px;border:1.6px solid rgba(255,255,255,.65);
  border-radius:999px;display:flex;align-items:center;justify-content:center}
.profile-mini .ava svg{width:20px;height:20px}
.profile-mini .nm{font-weight:700}
.profile-mini .em{color:var(--muted);font-size:.94rem}

/* Hero */
.hero{display:grid;grid-template-columns:1.2fr .9fr;gap:24px;align-items:center}
.hero-card{
  background:linear-gradient(180deg,rgba(255,255,255,.07),rgba(255,255,255,0)) padding-box,
             linear-gradient(90deg,rgba(114,38,255,.35),rgba(1,0,48,.35)) border-box;
  border:1px solid transparent;border-radius:28px;padding:28px 30px;box-shadow:0 20px 60px rgba(0,0,0,.45);
}
.hero h1{font-size:2.4rem;line-height:1.15;margin:2px 0 10px 0}
.hero p{color:var(--muted);font-size:1.05rem}

/* Ilustrasi kanan */
.hero-ill{display:flex;justify-content:center}
.hero-ill img{max-width:520px;width:100%;height:auto;filter:drop-shadow(0 0 22px rgba(255,255,255,.28))}

/* Pill navigation (tabs) */
.pillbar{
  margin:18px 0 12px 0;padding:8px;
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.10);
  border-radius:999px;box-shadow:0 14px 40px rgba(0,0,0,.35);
}
.pillbar .stRadio>div{display:flex;gap:8px;flex-wrap:wrap}
.pillbar .stRadio input{display:none}
.pillbar .stRadio label{
  display:flex;align-items:center;gap:10px;padding:12px 16px;border-radius:999px;
  color:#fff;cursor:pointer;position:relative;transition:transform .12s ease,background .15s ease;
}
.pillbar .stRadio label:hover{transform:translateY(-1px)}
.pillbar .stRadio [aria-checked="true"] label{
  background:linear-gradient(90deg,#27116d 0%, #5a2fe3 100%);
}
.pillbar svg{width:18px;height:18px;stroke:currentColor}

/* Hint bar */
.hint{
  display:flex;align-items:flex-start;gap:12px;margin:10px 0 8px 4px;
  color:#fff;background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,0));
  border:1px solid rgba(255,255,255,.10);border-radius:16px;padding:12px 14px;
}
.hint svg{width:18px;height:18px;margin-top:2px}
</style>
""", unsafe_allow_html=True)

# ---------- Profile mini ----------
PROFILE_NAME  = "Anisa Nabila"
PROFILE_EMAIL = "anisanbilaa@gmail.com"
st.markdown(
    f"""
    <div class="profile-mini">
      <div class="ava">
        <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.7">
          <path d="M20 21a8 8 0 0 0-16 0"/><circle cx="12" cy="7" r="4"/>
        </svg>
      </div>
      <div>
        <div class="nm">{PROFILE_NAME}</div>
        <div class="em">{PROFILE_EMAIL}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Hero section ----------
HERO_IMG = "rps_outline.png"   # pakai file ikonmu (neon outline)
left, right = st.columns([1.25, .95])
with left:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-card">
            <h1>DETEKSI DAN KLASIFIKASI<br/>GAMBAR BATU, GUNTING, DAN KERTAS</h1>
            <p><b>Ayo coba unggah gambar tanganmu!</b> Sistem ini akan mengidentifikasi bentuknya sebagai
            batu, gunting, atau kertas.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with right:
    try:
        st.markdown('<div class="hero-ill">', unsafe_allow_html=True)
        st.image(Image.open(HERO_IMG).convert("RGBA"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception:
        pass

# ---------- Pill navigation (native radio) ----------
if "page" not in st.session_state: st.session_state.page = "Deteksi Gambar"

pill_icons = {
    "Deteksi Gambar": """
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 15l4-4 3 3 5-5 4 4"/>
        </svg>""",
    "Klasifikasi Gambar": """
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M4 7h16M4 12h10M4 17h6"/>
        </svg>""",
    "Penjelasan Model": """
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <circle cx="12" cy="12" r="9"/><path d="M12 8v4m0 4h.01"/>
        </svg>"""
}

st.markdown('<div class="pillbar">', unsafe_allow_html=True)
choice = st.radio(
    "Menu",
    ["Deteksi Gambar", "Klasifikasi Gambar", "Penjelasan Model"],
    index=["Deteksi Gambar","Klasifikasi Gambar","Penjelasan Model"].index(st.session_state.page),
    format_func=lambda x: f"""{x}""",
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.page = choice
st.markdown('</div>', unsafe_allow_html=True)

# inject ikon ke label radio (menyisipkan HTML ikon di depan teks)
for label in pill_icons:
    st.markdown(
        f"""
        <script>
        const radios = Array.from(parent.document.querySelectorAll('.pillbar [role="radio"] > label'));
        const map = {{"Deteksi Gambar": `{pill_icons["Deteksi Gambar"]}`,
                      "Klasifikasi Gambar": `{pill_icons["Klasifikasi Gambar"]}`,
                      "Penjelasan Model": `{pill_icons["Penjelasan Model"]}`}};
        radios.forEach(l=>{{ const t=l.innerText.trim(); if(map[t] && !l.dataset.icon){{
          l.dataset.icon="1"; const s=document.createElement('span'); s.innerHTML=map[t]; s.style.display='inline-flex';
          s.style.alignItems='center'; s.style.marginRight='6px'; l.prepend(s); }} }});
        </script>
        """,
        unsafe_allow_html=True,
    )

# ---------- Hint di bawah pills ----------
st.markdown(
    """
    <div class="hint">
      <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <path d="M12 9v4m0 4h.01"/>
      </svg>
      <div>Pastikan tangan terlihat jelas pada gambar yang diunggah dan gunakan <b>background polos</b> agar sistem mengenali dengan tepat.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# ======= CONTENTS ========
# =========================

def uploader_card(key_label:str, title="Unggah Gambar"):
    st.markdown(f"<div class='card'><div class='card-title' style='font-size:1.18rem'>{title}</div>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Gunakan <b>latar belakang polos</b> & pencahayaan cukup untuk hasil terbaik.</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f

# ---- DETEKSI ----
if st.session_state.page == "Deteksi Gambar":
    c1, c2 = st.columns([1.04,1])
    with c1:
        f = uploader_card("up_yolo", "Unggah Gambar • Deteksi (RPS)")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='card-title'>Hasil Deteksi</div>", unsafe_allow_html=True)
        if not f:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan deteksi.</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Menjalankan YOLOv8..."):
                res = yolo_model.predict(img, verbose=False)
                plotted = cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB)
            st.image(plotted, use_container_width=True, caption="Bounding boxes")
            names=res[0].names; boxes=res[0].boxes
            if boxes is not None and len(boxes)>0:
                cls_ids=[int(c) for c in boxes.cls.tolist()]
                dominant=Counter(cls_ids).most_common(1)[0][0]
                st.markdown(f"<div class='big-result'>Prediksi Utama ⮕ {names[dominant].capitalize()}</div>", unsafe_allow_html=True)
            else:
                st.info("Tidak ada objek terdeteksi pada gambar ini.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---- KLASIFIKASI ----
elif st.session_state.page == "Klasifikasi Gambar":
    c1, c2 = st.columns([1.04,1])
    with c1:
        g = uploader_card("up_cls", "Unggah Gambar • Klasifikasi (RPS)")
        if g:
            img2 = Image.open(g).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='card-title'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
        if not g:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            arr = np.expand_dims(image.img_to_array(img2.resize((224,224))),0)/255.0
            with st.spinner("Mengklasifikasikan..."):
                probs = classifier.predict(arr)[0].astype(float)
            labels = ["paper","rock","scissors"] if len(probs)==3 else [f"class_{i}" for i in range(len(probs))]
            top_idx=int(np.argmax(probs)); top=labels[top_idx]; p=float(probs[top_idx])
            st.markdown(f"<div class='big-result'>Prediksi Utama ⮕ {top.capitalize()}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='caption' style='margin:.2rem 0 1rem 0;'>Skor keyakinan: <b>{p:.4f}</b></p>", unsafe_allow_html=True)
            for name, pv in zip(labels, probs):
                st.markdown(
                    f"<div class='prog-wrap'><span class='lbl'>{name.capitalize()}</span>"
                    f"<div class='prog'><span style='--w:{pv*100:.2f}%;'></span></div>"
                    f"<span class='val'>{pv*100:.1f}%</span></div>", unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

# ---- PENJELASAN ----
else:
    model_choice = st.selectbox("Pilih model yang ingin dijelaskan", ["YOLOv8","CNN"], index=0)
    # (lanjutkan blok dokumentasi/evaluasi kamu seperti sebelumnya)

    def metric_bar(label:str, value:float):
        pct = max(0.0, min(1.0, float(value))) * 100
        st.markdown(
            f"<div class='prog-wrap'><span class='lbl'>{label}</span>"
            f"<div class='prog'><span style='--w:{pct:.2f}%;'></span></div>"
            f"<span class='val'>{pct:.1f}%</span></div>",
            unsafe_allow_html=True
        )

    # ---- Dataset (with per-class counters)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Dataset</div>", unsafe_allow_html=True)
    if model_choice == "YOLOv8":
        st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (RPS) – Dicoding** dengan anotasi **bounding box** (Roboflow).  
**Split & Ukuran.** Semua citra **640×640**; split **80%** latih, **10%** validasi, **10%** uji.  
**Format.** Label kompatibel **YOLOv8** (anchor-free).
        """)
    else:
        st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (RPS) – Dicoding** untuk klasifikasi.  
**Split & Prapemrosesan.** **70/20/10** (latih/validasi/uji), **224×224** RGB, normalisasi **0–1**, augmentasi ringan.
        """)

    counts = {"Rock":726, "Paper":712, "Scissors":750}
    colc = st.columns(3)
    icons = {
      "Rock": """<path d="M18,30 c-4,0 -8,-3 -8,-7 v-8 c0-6 16-6 16,2 v6 c0,4 -4,7 -8,7z" stroke="white" stroke-width="3" fill="none"/>""",
      "Paper": """<path d="M14,30 c-3,-10 2,-18 8,-18 5,0 6,5 6,10 v8" stroke="white" stroke-width="3" fill="none"/><path d="M10,26 c-2,-7 1,-12 6,-12" stroke="white" stroke-width="3" fill="none"/>""",
      "Scissors": """<path d="M10,12 l8,12 M22,12 l-6,10 M12,26 c4,4 10,4 12,0" stroke="white" stroke-width="3" fill="none"/>"""
    }
    for (k,v), col in zip(counts.items(), colc):
        col.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-top:10px;">
          <div class="icon-bubble">
            <svg viewBox="0 0 36 36">{icons[k]}</svg>
          </div>
          <div>
            <div style="font-weight:700;font-size:1.05rem">{k}</div>
            <div style="font-weight:800;font-size:1.6rem">{v:,}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Arsitektur + Evaluasi
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Arsitektur</div>", unsafe_allow_html=True)
        if model_choice == "CNN":
            st.markdown(
                "<div class='flow'>"
                "<div class='node'>Conv2D(32, 3×3, ReLU) → MaxPool(2×2)</div>"
                "<div class='node'>Conv2D(64, 3×3, ReLU) → MaxPool(2×2)</div>"
                "<div class='node'>Conv2D(128, 3×3, ReLU) → MaxPool(2×2)</div>"
                "<div class='node'>Flatten</div>"
                "<div class='node'>Dense(128, ReLU) → Dropout(0.5)</div>"
                "<div class='node'>Dense(3, Softmax)</div>"
                "</div>", unsafe_allow_html=True)
            st.markdown("Optimizer **Adam**, loss **categorical_crossentropy**, **EarlyStopping** + **ModelCheckpoint**.")
        else:
            st.markdown(
                "<div class='flow'>"
                "<div class='node'>Backbone (SiLU, C2f, SPPF)</div>"
                "<div class='node'>Neck (FPN/PAN, multi-scale fusion)</div>"
                "<div class='node'>Head (stride 8/16/32, cls+box, anchor-free)</div>"
                "</div>", unsafe_allow_html=True)
            st.markdown("Inferensi UI menggunakan nilai default internal.")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Evaluasi</div>", unsafe_allow_html=True)
        if model_choice == "CNN":
            metric_bar("Accuracy", 0.94)
            metric_bar("Precision (macro)", 0.94)
            metric_bar("Recall (macro)", 0.94)
            metric_bar("F1-score (macro)", 0.94)
            metric_bar("Val Loss (↓ skala)", 1-0.94)
            st.markdown("Performa merata di tiga kelas; tidak tampak bias dominan.")
        else:
            metric_bar("Precision", 0.996)
            metric_bar("Recall", 1.00)
            metric_bar("mAP@50", 0.995)
            metric_bar("mAP@50–95", 0.925)
            metric_bar("Latency (skala cepat)", 1-0.017)  # 17ms ~ cepat
            st.markdown("Akurat & cepat — layak untuk **real-time**.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Kesimpulan
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Kesimpulan</div>", unsafe_allow_html=True)
    if model_choice == "CNN":
        st.markdown("CNN ringkas (3 blok konvolusi + **Dropout 0.5**) dengan **EarlyStopping/Checkpoint** memberi akurasi ~**94%** pada **RPS**. Cocok untuk pengklasifikasi akhir.")
    else:
        st.markdown("**YOLOv8n** presisi tinggi (**mAP@50 ≈ 0.995**) dengan latensi ~**17 ms/gambar**. FPN/PAN + head anchor-free efektif untuk deteksi **RPS** real-time.")
    st.markdown("</div>", unsafe_allow_html=True)
