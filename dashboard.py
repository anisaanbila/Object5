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
h1{ 
  font-weight:900; 
  line-height:1.12; 
  font-size:3.2rem;   /* ⬅️ Tambahkan baris ini */
}
h2,h3,h4{ font-weight:700; }
p,li,div,span,label{ font-weight:400; color:var(--text); }

/* Hide default header, widen container */
header[data-testid="stHeader"]{ display:none; }
.block-container{
  padding-top:0rem!important;
  padding-bottom:2rem;
  max-width:1300px;
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
.card h1{
  font-weight:900;
  margin-top:0.3rem;
  margin-left:0.8rem;
  font-size:3.3rem;
  color:#FFFFFF;
  text-shadow: 0 0 8px rgba(255,255,255,.2);
}
/* Caption khusus di header (biar gak ganggu tab) */
.header-caption{
  position:relative;
  left:0.8rem;        /* ⬅️ geser blok ke kanan sejauh 10rem */
  color:#DDE0FF;
  font-size:1.3rem;
  font-weight:500;
  line-height:1.6;
  margin-top:0.5rem;
  margin-bottom:1.2rem;
}
.card-title{ font-weight:700; font-size:2.2rem; margin-bottom:.7rem; color:#fff; }
.caption{ color: var(--muted); font-size:1.25rem; margin-top:1rem; margin-left:0.8rem}
/* Box versi kecil hanya untuk hasil deteksi & klasifikasi */
.card-small {
  padding: 5px 16px !important;   /* lebih tipis */
  border-radius: 14px !important;  /* opsional: biar tetap halus */
  font-weight:700; font-size:2.2rem; margin-bottom:.2rem; color:#fff; margin-left:0rem
}

/* ===== PROFILE CHIP (atas header) ===== */
.profile-bar{
  display:flex; align-items:center; gap:14px;
  width:fit-content;
  padding:.6rem .9rem; border-radius:999px;
  background:linear-gradient(90deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
  box-shadow:0 8px 26px rgba(0,0,0,.35), inset 0 0 14px rgba(255,255,255,.05);
  margin: 6px 0 6px 0; /* jarak bawah lebih rapat */
}

/* ===== PROFILE CHIP (dengan efek pop-up saat hover) ===== */
/* ===== PROFILE CHIP (dengan efek sama seperti .card:hover) ===== */
.profile-bar{
  display:flex;
  align-items:center;
  gap:14px;
  width:fit-content;
  padding:.6rem .9rem;
  border-radius:999px;
  background:linear-gradient(90deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
  box-shadow:0 8px 26px rgba(0,0,0,.35), inset 0 0 14px rgba(255,255,255,.05);
  margin: 6px 0 6px 0;
  transition: all 0.35s ease;       /* biar halus kayak card */
}

/* Saat diarahkan kursor */
.profile-bar:hover{
  box-shadow:
    0 26px 60px rgba(0,0,0,.55),          /* bayangan lebih dalam */
    0 0 0 1px rgba(255,255,255,.06) inset;/* garis halus dalam */
  transform: translateY(-1px);             /* naik dikit */
  background:linear-gradient(90deg, rgba(255,255,255,.08), rgba(255,255,255,.05));
}

/* Avatar tetap punya efek lembut */
.profile-avatar{
  width:36px;
  height:36px;
  border-radius:50%;
  border:2px solid rgba(255,255,255,.85);
  display:flex;
  align-items:center;
  justify-content:center;
  box-shadow:0 0 10px rgba(255,255,255,.18), inset 0 0 8px rgba(255,255,255,.12);
  transition: all 0.3s ease;
}

/* Tambahan: avatar ikut hidup dikit pas hover */
.profile-bar:hover .profile-avatar{
  transform: scale(1.08);
  box-shadow:0 0 14px rgba(255,255,255,.25), 0 0 8px rgba(141,63,255,.5);
}

/* Nama dan email tetap clean */
.profile-name{ font-weight:700; color:#FFFFFF; line-height:1.05; }
.profile-email{ color:#BBC0E6; font-size:.92rem; margin-top:2px; }


/* ===== FUTURISTIC LUCIDE TABS (Versi Stabil + 32px Ikon) ===== */
.stTabs [role="tablist"] {
  display: flex;
  justify-content: flex-start;
  align-items: center;
  gap: 1rem;
  padding: 0.8rem 1.2rem;
  border-radius: 40px;
  background: linear-gradient(90deg, #1a0066, #4b00c7);
  box-shadow: inset 0 0 20px rgba(255,255,255,0.06);
}

/* ===== Tab button ===== */
.stTabs [role="tab"] {
  display: flex;
  align-items: center;
  gap: 0.1rem;                        /* jarak ikon ↔ teks */
  color: #E0E2FF !important;
  font-weight: 500 !important;
  font-size: 1.2rem !important;       /* ukuran teks tab */
  line-height: 1.25 !important;
  padding: 0.8rem 1.6rem;
  border: none;
  border-radius: 40px;
  background: transparent;
  transition: all 0.25s ease;
}

/* pastikan semua elemen di dalam tab ikut ukuran teks */
.stTabs [role="tab"] * {
  font-size: inherit !important;
  font-weight: inherit !important;
  line-height: inherit !important;
}

/* hover & active state */
.stTabs [role="tab"]:hover {
  background: rgba(255,255,255,0.08);
  color: #FFFFFF !important;
}
.stTabs [role="tab"][aria-selected="true"] {
  background: linear-gradient(90deg, #602FFF, #8D3FFF);
  box-shadow: 0 0 18px rgba(138,70,255,0.5);
  color: #fff !important;
  transform: translateY(-1px);
}

/* ===== Ikon semua tab (ukuran seragam 32px, sejajar) ===== */
.stTabs [role="tab"]::before{
  content: "";
  display: inline-block;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  margin-right: 0.8rem;             /* jarak antara ikon dan teks */
  background-repeat: no-repeat;
  background-position: center;
  background-size: 32px 32px;
}

/* ===== Ikon per tab (Lucide outline style) ===== */
.stTabs [role="tab"]:nth-child(1)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='%23E0E2FF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 5h2l2-2h10l2 2h2a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z'/><circle cx='12' cy='13' r='3'/></svg>");
}
.stTabs [role="tab"]:nth-child(2)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='%23E0E2FF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='3' width='18' height='18' rx='2' ry='2'/><circle cx='8.5' cy='8.5' r='1.5'/><path d='M21 15l-5-5L5 21'/></svg>");
}
.stTabs [role="tab"]:nth-child(3)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='%23E0E2FF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='3'/><path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'/></svg>");
}

/* Hilangkan underline bawaan Streamlit */
.stTabs [role="tablist"] button { border-bottom: none !important; }



/* Header right image (lebih kecil & rapat) */
.header-rps-wrap{
  display:flex;
  justify-content:center;
  align-items:flex-start;
  margin-top:-36px;
  margin-bottom:0;
}
.header-rps-img{
  width:auto;
  max-width:250px;
  height:auto;
  display:block;
  filter: drop-shadow(0 0 10px rgba(255,255,255,.20))
          drop-shadow(0 0 3px rgba(255,255,255,.18));
}
@media (max-width: 1200px){
  .header-rps-wrap{ margin-top:-24px; }
  .header-rps-img{ max-width:260px; }
}
@media (max-width: 992px){
  .header-rps-wrap{ margin-top:-10px; }
  .header-rps-img{ max-width:220px; }
}

/* ===== Notice / Tips Banner (di atas uploader) ===== */
.notice-banner{
  display:flex; align-items:center; gap:14px;
  padding:12px 16px; margin:0 0 14px 0;
  border-radius:14px;
  background: linear-gradient(90deg, #240070, #4a00c4 55%, #6e2bff);
  border:1px solid rgba(255,255,255,.14);
  box-shadow:0 10px 28px rgba(0,0,0,.35), inset 0 0 18px rgba(255,255,255,.05);
}
.notice-icon{
  width:38px; height:38px; min-width:38px;
  display:flex; align-items:center; justify-content:center;
  border-radius:10px;
  background: rgba(255,255,255,.08);
  box-shadow: inset 0 0 10px rgba(255,255,255,.08);
}
.notice-icon svg{ width:22px; height:22px; color:#FFFFFF; opacity:.95; }
.notice-text{
  color:#EAEAFF; font-size:1.02rem; line-height:1.45; font-weight:500;
}

.notice-text{
  color:#EAEAFF; font-size:1.02rem; line-height:1.45; font-weight:500;
}

/* Tambah jarak antara box notice dan box drag uploader */
[data-testid="stFileUploader"] {
  margin-top: 20px !important;
}
[data-testid="stFileUploadDropzone"] {
  margin-top: 20px !important;
}
/* Tombol 'Browse files' */


/* ===== UBAH WARNA FILE UPLOADED DAN TOMBOL BROWSE FILE ===== */

/* 1️⃣ Warna teks nama file yang diunggah */
[data-testid="stFileUploader"] a, 
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
  color: #545454 !important;            /* putih */
  font-weight: 500 !important;
}

/* 2️⃣ Warna tombol "Browse files" */
[data-testid="stFileUploader"] button {
  color: #545454 !important;            /* teks gelap */
  background: #FFFFFF !important;       /* tombol putih */
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: all 0.25s ease !important;
}



/* ==== 2) HASIL UPLOAD (list di bawah dropzone) — putih ==== */
/* hanya elemen di dalam list hasil (bukan teks drag) */
[data-testid="stFileUploader"] [role="list"] a,
[data-testid="stFileUploader"] [role="list"] span,
[data-testid="stFileUploader"] [role="list"] p{
  color:#FFFFFF !important;          /* nama file & size jadi putih */
  font-weight:500 !important;
}

/* (opsional) ikon & tombol X tetap terlihat */
[data-testid="stFileUploader"] [role="list"] svg{ opacity:.9; }
[data-testid="stFileUploader"] [role="list"] button{ color:#FFFFFF !important; }

/* 3️⃣ Efek hover tombol */
[data-testid="stFileUploader"] button:hover {
  background: linear-gradient(90deg,#ECECEC,#D6D6D6) !important;
  color: #000000 !important;
  box-shadow: 0 0 10px rgba(255,255,255,0.3);
}


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
# PROFILE CHIP (atas header)
# =========================
col_profile, _ = st.columns([1, 3])
with col_profile:
    st.markdown(
        """
        <div class="profile-bar">
          <div class="profile-avatar">
            <!-- Lucide user-round -->
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M18 20a6 6 0 0 0-12 0"/>
              <circle cx="12" cy="10" r="4"/>
            </svg>
          </div>
          <div>
            <div class="profile-name">Anisa Nabila</div>
            <div class="profile-email">anisanbilaa@gmail.com</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# HEADER (left text + PNG icon on right, NO BOX)
# =========================
ICON_PATH = "rps_outline.png"  # file sejajar dengan app.py
c1, c2 = st.columns([1.6, 1.0], vertical_alignment="center")

with c1:
    st.markdown(
        "<div class='card'>"
        "<h1>DETEKSI DAN KLASIFIKASI GAMBAR BATU, GUNTING, DAN KERTAS</h1>"
        "<p class='header-caption'><b>Ayo coba unggah gambar tanganmu! </b>Sistem ini akan mengidentifikasi bentuknya "
        "sebagai batu, gunting, atau kertas.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <style>
          .header-rps-wrap{
              display:flex; justify-content:center; align-items:flex-start;
              margin-top:-72px;
          }
          @media (max-width: 1200px){ .header-rps-wrap{ margin-top:-40px; } }
          @media (max-width: 992px){ .header-rps-wrap{ margin-top:-16px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        rps_icon = Image.open(ICON_PATH).convert("RGBA")
        st.markdown("<div class='header-rps-wrap'>", unsafe_allow_html=True)
        st.image(rps_icon, caption=None, use_container_width=False, output_format="PNG")
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Ikon header tidak ditemukan di '{ICON_PATH}'. Detil: {e}")

# =========================
# TABS (3 tabs — Lucide capsule)
# =========================
tab_det, tab_cls, tab_docs = st.tabs([
    "Deteksi Gambar", "Klasifikasi Gambar", "Penjelasan Model"
])

def uploader_card(key_label: str, title="UNGGAH GAMBAR"):
    st.markdown(
        f"""
        <div class='card'>
          <div class='card-title' style='font-size:1.5rem'>{title}</div>

          <!-- Notice banner -->
          <div class="notice-banner">
            <div class="notice-icon">
              <!-- warning triangle (SVG) -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
              </svg>
            </div>
            <div class="notice-text">
              Pastikan tangan terlihat jelas pada gambar yang diunggah dan gunakan background polos supaya sistem dapat mengenali dengan tepat.
            </div>
          </div>
          """,
        unsafe_allow_html=True,
    )
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f


# =========================
# TAB: DETEKSI (YOLOv8)
# =========================
with tab_det:
    left, right = st.columns([1.04,1])
    with left:
        f = uploader_card("up_yolo", "UNGGAH GAMBAR")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-small' style='font-size:1.5rem'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='card-small' style='font-size:1.5rem'>HASIL DETEKSI</div>", unsafe_allow_html=True)
        if not f:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan deteksi.</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Menjalankan YOLOv8..."):
                res = yolo_model.predict(img, verbose=False)
                plotted = res[0].plot()
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, use_container_width=True, caption="Bounding boxes")

            names = res[0].names
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                cls_ids = [int(c) for c in boxes.cls.tolist()]
                dominant = Counter(cls_ids).most_common(1)[0][0]
                st.markdown(f"<div class='big-result'>Gambar ini terdeteksi sebagai {names[dominant].capitalize()}</div>", unsafe_allow_html=True)
            else:
                st.info("Tidak ada objek terdeteksi pada gambar ini.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB: KLASIFIKASI (CNN)
# =========================
with tab_cls:
    left, right = st.columns([1.04,1])
    with left:
        g = uploader_card("up_cls", "UNGGAH GAMBAR")
        if g:
            img2 = Image.open(g).convert("RGB")
            st.markdown("<div class='card'><div class='card-small' style='font-size:1.5rem'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='card-small' style='font-size:1.5rem'>HASIL KLASIFIKASI</div>", unsafe_allow_html=True)
        if not g:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            img_resized = img2.resize((224,224))
            arr = image.img_to_array(img_resized); arr = np.expand_dims(arr,0)/255.0
            with st.spinner("Mengklasifikasikan..."):
                pred = classifier.predict(arr)
            probs = pred[0].astype(float)
            labels = ["paper","rock","scissors"] if len(pred[0])==3 else [f"class_{i}" for i in range(len(pred[0]))]
            top_idx = int(np.argmax(probs)); top_name = labels[top_idx]; top_prob = float(probs[top_idx])

            st.markdown(f"<div class='big-result'>Gambar ini terdeteksi sebagai {top_name.capitalize()}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='caption' style='margin:.2rem 0 1rem 0;'>Skor keyakinan: <b>{top_prob:.4f}</b></p>", unsafe_allow_html=True)

            for name, p in zip(labels, probs):
                st.markdown(
                    f"<div class='prog-wrap'><span class='lbl'>{name.capitalize()}</span>"
                    f"<div class='prog'><span style='--w:{p*100:.2f}%;'></span></div>"
                    f"<span class='val'>{p*100:.1f}%</span></div>",
                    unsafe_allow_html=True
                )

            df = pd.DataFrame({"Kelas": [n.capitalize() for n in labels], "Probabilitas (%)": (probs*100).round(2)})
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB: PENJELASAN MODEL (dropdown + per-box rapi)
# =========================
