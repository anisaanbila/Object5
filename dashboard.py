# app.py — Vision Dashboard (RPS) — Gradient UI + Model Docs (per-box)
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Vision Dashboard — RPS", page_icon="🧠", layout="wide")

# -----------------------------
# THEME (gradient + clean)
# -----------------------------
st.markdown("""
<style>
:root{
  /* base colors */
  --bg1:#0e0f1a;         /* top gradient */
  --bg2:#151527;         /* bottom gradient */
  --panel:#151524;
  --panel-2:#1B1B2C;
  --text:#E8E9F5;
  --muted:#A0A2B8;
  /* gradient accents */
  --g1:#7C3AED;          /* purple */
  --g2:#5B7CFF;          /* indigo/blue */
  --g3:#06B6D4;          /* cyan */
}

header[data-testid="stHeader"]{ display:none; }
.block-container{ padding-top:3.6rem!important; padding-bottom:2rem; max-width:1300px; }

/* app background = gradient */
html, body, [data-testid="stAppViewContainer"]{
  background: linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 70%) fixed;
  color: var(--text);
}
a{ color:#A7B6FF!important; }
h1,h2,h3,h4{ color: var(--text); }

/* cards */
.card{
  background:
    linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0)) padding-box,
    linear-gradient(90deg, rgba(124,58,237,.35), rgba(6,182,212,.35)) border-box;
  border:1px solid transparent;
  border-radius:16px;
  padding:18px 20px;
  box-shadow: 0 10px 28px rgba(0,0,0,.35);
}
.card.compact{ padding:14px 16px; }
.card-title{
  font-weight:700; font-size:1rem; color:#CFD2FF; margin-bottom:.5rem;
}
.caption{ color: var(--muted); font-size:.86rem; }
hr{ border-color: #2b2c44; }

/* header pills + action */
.pill{
  background: linear-gradient(90deg, rgba(124,58,237,.18), rgba(6,182,212,.18));
  border:1px solid rgba(140,140,210,.35);
  color:#C9CCFF; padding:.35rem .7rem; border-radius:999px; font-weight:600; font-size:.82rem;
}
.action{
  background: linear-gradient(90deg, var(--g1), var(--g3));
  color:#fff; font-weight:700; padding:.55rem 1rem; border-radius:12px;
  display:inline-flex; gap:.5rem; text-decoration:none;
}

/* KPI */
.kpi{ display:flex; gap:.6rem; align-items:center; padding:.6rem .9rem;
     background:#101020; border:1px solid #2b2c44; border-radius:12px; }
.kpi .big{ font-weight:800; font-size:1.1rem; }

/* tabs underline */
.stTabs [role="tablist"]{ gap:1rem; }
.stTabs [role="tab"]{ border-bottom:2px solid transparent; }
.stTabs [role="tab"][aria-selected="true"]{
  border-bottom:2px solid; border-image: linear-gradient(90deg,var(--g1),var(--g3)) 1;
}

/* file uploader text color */
[data-testid="stFileUploader"] section div{ color:#A0A2B8!important; }

/* RADIO — circular chips (CNN / YOLOv8) */
.stRadio > div{ flex-direction: row !important; gap: 1rem; }
.stRadio label{
  position: relative;
  display: inline-flex; align-items:center; justify-content:center;
  min-width:78px; height:42px;
  padding:0 14px; border-radius:999px; cursor:pointer;
  color:#DADCF9; font-weight:700; letter-spacing:.2px;
  background: radial-gradient(120% 140% at 0% 0%, rgba(124,58,237,.20), rgba(6,182,212,.08));
  border:1px solid #343556;
  box-shadow: inset 0 0 0 2px rgba(255,255,255,.02);
}
.stRadio label:hover{
  box-shadow: 0 0 0 3px rgba(124,58,237,.20);
}
/* selected state = gradient ring + bright text */
.stRadio [role="radio"][aria-checked="true"] label{
  color:white;
  background: linear-gradient(120deg, rgba(124,58,237,.55), rgba(6,182,212,.55));
  border-color: transparent;
  box-shadow: inset 0 0 0 2px rgba(255,255,255,.08), 0 0 0 2px rgba(124,58,237,.25);
}
/* hide the default radio dot */
.stRadio input[type="radio"]{ display:none; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Anisa Nabila_Laporan 4.pt")                   # YOLOv8 detector
    clf  = tf.keras.models.load_model("model/Anisa Nabila_Laporan 2.h5")  # CNN classifier
    return yolo, clf

yolo_model, classifier = load_models()

# -----------------------------
# HEADER
# -----------------------------
c1, c2, c3 = st.columns([1.6,1,1])
with c1:
    st.markdown(
        "<div class='card'>"
        "<div class='card-title'>Dashboard</div>"
        "<h1 style='margin:0 0 .3rem 0;'>Welcome Back, Anisa!</h1>"
        "<div class='caption'>UI untuk <b>Rock–Paper–Scissors</b>: deteksi (YOLOv8) & klasifikasi (CNN)</div>"
        "</div>", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class='card compact'>
      <div class='card-title'>Model Status</div>
      <div class='kpi'>✅<span class='big'>Ready</span><span class='caption'>YOLO & Classifier</span></div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class='card compact' style='display:flex;justify-content:space-between;align-items:center;gap:.8rem'>
      <span class='pill'>RPS Vision</span>
      <a class='action' href='#' onclick='return false;'>＋ Create Session</a>
    </div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab_det, tab_cls, tab_profile, tab_docs = st.tabs([
    "Deteksi Objek (YOLO)", "Klasifikasi Gambar", "Profil Developer", "Penjelasan Model"
])

def uploader_card(key_label:str, title="Unggah Gambar"):
    st.markdown(f"<div class='card'><div class='card-title'>{title}</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f

# -----------------------------
# TAB: DETEKSI
# -----------------------------
with tab_det:
    left, right = st.columns([1.04,1])
    with left:
        f = uploader_card("up_yolo", "Unggah Gambar • Deteksi")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil Deteksi</div>", unsafe_allow_html=True)
        if not f:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan deteksi.</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Menjalankan YOLO..."):
                res = yolo_model(img); plotted = res[0].plot()
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, use_container_width=True, caption="Deteksi (bounding boxes)")
            names, boxes = res[0].names, res[0].boxes
            if boxes is not None and len(boxes)>0:
                st.markdown("<hr>", unsafe_allow_html=True); st.markdown("**Ringkasan:**")
                for cid, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    st.write(f"• {names[int(cid)]} — conf: {conf:.2f}")
            else:
                st.info("Tidak ada objek terdeteksi.")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB: KLASIFIKASI
# -----------------------------
with tab_cls:
    left, right = st.columns([1.04,1])
    with left:
        g = uploader_card("up_cls", "Unggah Gambar • Klasifikasi")
        if g:
            img2 = Image.open(g).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
        if not g:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            img_resized = img2.resize((224,224))
            arr = image.img_to_array(img_resized); arr = np.expand_dims(arr,0)/255.0
            with st.spinner("Mengklasifikasikan..."): pred = classifier.predict(arr)
            prob = float(np.max(pred)); idx = int(np.argmax(pred))
            labels = ["paper","rock","scissors"]  # sesuaikan dengan urutan output model Anda
            name = labels[idx] if idx < len(labels) else str(idx)
            st.markdown(f"**Label Prediksi:** `{name}`")
            st.markdown(f"**Probabilitas:** `{prob:.4f}`")
            st.markdown("<div class='caption'>Catatan: sesuaikan daftar <code>labels</code> dengan kelas model Anda.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB: PROFIL DEVELOPER (pertanyaan saja)
# -----------------------------
with tab_profile:
    st.markdown("<div class='card'><div class='card-title'>Profil Developer — Mohon jawab di chat</div>", unsafe_allow_html=True)
    st.markdown("""
• **Nama yang ditampilkan** (dan panggilan)  
• **Peran/role utama**  
• **Tagline singkat** (1–2 kalimat)  
• **Skill inti (5–8)**  
• **Proyek unggulan (≤3)**  
• **Kontak & tautan** (email, GitHub, LinkedIn/Portofolio)  
• **Riwayat pendidikan (opsional)** dalam format timeline  
• **Preferensi warna/aksen tambahan** (kalau mau selain gradasi saat ini)
""")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB: PENJELASAN MODEL (per-box + radio pilihan CNN/YOLOv8)
# -----------------------------
with tab_docs:
    st.markdown("<div class='card'><div class='card-title'>Penjelasan Model</div>", unsafe_allow_html=True)

    choice = st.radio("Pilih model:", ["CNN","YOLOv8"], horizontal=True, label_visibility="collapsed", key="doc_choice")

    def box_dataset(kind:str):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Dataset")
        if kind=="CNN":
            st.markdown(
                "- Rock–Paper–Scissors (Dicoding). 3 kelas: **paper (712)**, **rock (726)**, **scissors (750)** → total **2.188**.\n"
                "- Split **70/20/10** (latih/val/uji), **resize 224×224**, **normalisasi 0–1**, **augmentasi** (rotasi ≤10°, zoom ≤10%, horizontal flip)."
            )
        else:
            st.markdown(
                "- Rock–Paper–Scissors (Dicoding) dengan **labeling Roboflow**.\n"
                "- Semua gambar **640×640**, split **80/10/10**, siap untuk deteksi objek."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    def box_arch_eval(kind:str):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Arsitektur")
            if kind=="CNN":
                st.markdown(
                    "3×(Conv2D + MaxPool) → Flatten → Dense(128, ReLU) + **Dropout 0.5** → Dense(3, Softmax). "
                    "Callbacks: **EarlyStopping** & **ModelCheckpoint**."
                )
            else:
                st.markdown(
                    "**YOLOv8n (anchor-free)**: Backbone → Neck (**FPN/PAN**) → Head (**Detect**) pada 3 skala (stride 8/16/32)."
                )
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Evaluasi (ringkas)")
            if kind=="CNN":
                st.markdown(
                    "- Akurasi validasi ≈ **94%**; precision/recall/F1 **stabil** di tiga kelas.\n"
                    "- Confusion matrix & laporan klasifikasi menunjukkan generalisasi baik."
                )
            else:
                st.markdown(
                    "- 100 epoch; **Precision ≈ 0.996**, **Recall 1.000**, **mAP@50 0.995**, **mAP@50–95 0.925**.\n"
                    "- Kecepatan ~**17 ms/gambar** (pre+infer+post)."
                )
            st.markdown("</div>", unsafe_allow_html=True)

    def box_conclusion(kind:str):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Kesimpulan")
        if kind=="CNN":
            st.markdown(
                "Model CNN memberikan **klasifikasi stabil** untuk 3 kelas RPS dengan akurasi tinggi dan pengendalian overfitting via callbacks."
            )
        else:
            st.markdown(
                "YOLOv8n **akurat dan cepat** untuk deteksi RPS; cocok untuk skenario **realtime** dengan performa metrik yang sangat tinggi."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # RENDER per-box sesuai pilihan
    box_dataset(choice)
    box_arch_eval(choice)
    box_conclusion(choice)

    st.markdown("</div>", unsafe_allow_html=True)
