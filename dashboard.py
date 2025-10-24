# app.py — RPS Vision Dashboard (Futuristic • Gradient • Poppins)
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import time
import pandas as pd
from collections import Counter


st.set_page_config(
    page_title="DETEKSI DAN KLASIFIKASI GAMBAR BATU, GUNTING, DAN KERTAS",
    page_icon="icon.png",
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
  margin-top:0rem;
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
  gap: -6rem;                        /* jarak ikon ↔ teks */
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
  margin-right: 0.5rem;             /* jarak antara ikon dan teks */
  background-repeat: no-repeat;
  background-position: center;
  background-size: 28px 28px;
}

/* ===== Ikon per tab (Lucide outline style) ===== */
.stTabs [role="tab"]:nth-child(1)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='%23E0E2FF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 5h2l2-2h10l2 2h2a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z'/><circle cx='12' cy='13' r='3'/></svg>");
}
.stTabs [role="tab"]:nth-child(2)::before {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='%23E0E2FF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='3' width='18' height='18' rx='2' ry='2'/><circle cx='8.5' cy='8.5' r='1.5'/><path d='M21 15l-5-5L5 21'/></svg>");
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
/* ===== Game Link Button ===== */
.game-link { margin-top: 18px; text-align: center; }
.game-link a {
  display:inline-block; text-decoration:none;
  font-weight:600; font-size:1.05rem; color:#FFFFFF;
  background: linear-gradient(90deg, #5C4DFF, #9B6DFF);
  padding:10px 22px; border-radius:12px;
  box-shadow:0 0 12px rgba(156,122,255,.35);
  transition: all .25s ease;
}
.game-link a:hover {
  box-shadow:0 0 18px rgba(156,122,255,.6);
  transform: translateY(-1px);
}

/* ===== Hasil Prediksi Futuristik ===== */
.pred-result{
  text-align:center;
  margin-top:10px;
  padding:14px 20px;
  border-radius:14px;
  background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.15);
  box-shadow:0 0 18px rgba(114,38,255,0.3), inset 0 0 12px rgba(255,255,255,0.05);
  transition: all 0.3s ease;
}
.pred-result:hover{
  box-shadow:0 0 28px rgba(114,38,255,0.55), 0 0 0 1px rgba(255,255,255,0.08) inset;
  transform: translateY(-1px);
}
.pred-label{
  font-size:2rem;
  font-weight:800;
  color:#FFFFFF;
  letter-spacing:1px;
  text-shadow:0 0 8px rgba(255,255,255,0.25), 0 0 18px rgba(114,38,255,0.45);
}
.pred-acc{
  font-size:1.05rem;
  color:#C7D2FE;
  margin-top:4px;
  font-weight:500;
  opacity:.95;
}

/* Ajakan game */
.game-lead{
  text-align: center;
  text-justify: inter-word;
  color:#EAEAFF;
  font-size:1.05rem;
  font-weight:500;            /* atur di sini: 500/600/700/800 */
  margin-top:25px !important;
  margin-bottom:0px;
}

/* Supaya <b>/<strong> di dalam .game-lead ikut bobot parent */
.game-lead b,
.game-lead strong{
  font-weight: inherit !important;
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
            st.image(plotted, use_container_width=True)

            names = res[0].names
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                cls_ids = [int(c) for c in boxes.cls.tolist()]
                confs = [float(c) for c in boxes.conf.tolist()]   # ambil nilai confidence
                dominant = Counter(cls_ids).most_common(1)[0][0]
                conf_mean = np.mean(confs) * 100                  # rata-rata confidence (persentase)
                conf_top  = max(confs) * 100                      # ambil yang tertinggi (opsional)
                
                st.markdown(f"""
                <div class='pred-result'>
                    <div class='pred-label'>{names[dominant].capitalize()}</div>
                    <div class='pred-acc'>Akurasi Deteksi: {conf_top:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Tidak ada objek terdeteksi pada gambar ini.")
                
            st.markdown("""
            <p class='game-lead'>
            Tertarik untuk mencoba game Rock–Paper–Scissors versi online?
            </p>
            """, unsafe_allow_html=True)


            # Tombol game online (Deteksi)
            st.markdown("""
            <div class='game-link'>
              <a href='https://bloob.io/id/rps' target='_blank' rel='noopener'>
                🎮 Mainkan Rock–Paper–Scissors Online
              </a>
            </div>
            """, unsafe_allow_html=True)

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

            conf_top = top_prob * 100.0
            st.markdown(f"""
            <div class='pred-result'>
              <div class='pred-label'>{top_name.capitalize()}</div>
              <div class='pred-acc'>Akurasi Klasifikasi: {conf_top:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
                
            st.markdown("""
            <p class='game-lead'>
            Tertarik untuk mencoba game Rock–Paper–Scissors versi online?
            </p>
            """, unsafe_allow_html=True)
        
            # Tombol game online (Deteksi)
            st.markdown("""
            <div class='game-link'>
              <a href='https://bloob.io/id/rps' target='_blank' rel='noopener'>
                🎮 Mainkan Rock–Paper–Scissors Online
              </a>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB: PENJELASAN MODEL — dropdown → deskripsi langsung di background utama
# ============================================================
with tab_docs:
    model_choice = st.selectbox(
        "Pilih jenis model yang ingin dijelaskan:",
        ["Model Klasifikasi", "Model Deteksi"],
        index=0
    )
    # ==== ubah warna teks di dropdown ====
    st.markdown("""
    <style>
    /* Hanya teks di dalam dropdown */
    .stSelectbox div[data-baseweb="select"] div[role="combobox"],
    .stSelectbox div[data-baseweb="select"] div[role="combobox"] * {
        color: #0F172A !important;     /* Warna teks di dropdown */
        font-weight: 600 !important;
    }
    
    /* Teks opsi saat dropdown dibuka */
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li * {
        color: #0F172A !important;     /* Warna teks opsi */
    }
    
    /* Ikon panah di sisi kanan */
    .stSelectbox div[data-baseweb="select"] svg {
        color: #0F172A !important;
        opacity: 0.9 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ==== CSS full transparan ====
    st.markdown("""
    <style>
      .model-head{
        display:flex;
        align-items:center;
        gap:12px;
        margin-top:-35px;
        margin-bottom:10px;
      }
      .icon-bubble{
        width:44px; height:44px; min-width:44px;
        display:flex; align-items:center; justify-content:center;
        border-radius:12px;
        background:rgba(255,255,255,.09);
        border:1px solid rgba(255,255,255,.12);
      }
      .icon-bubble svg{ width:24px; height:24px; color:#fff; opacity:.95; }
      .model-title{
        font-weight:800;
        font-size:1.5rem;
        color:#FFFFFF;
        text-shadow:0 0 8px rgba(255,255,255,.2);
      }
      .model-body{
          color:#EAEAFF;
          line-height:1.7;
          font-size:1.05rem;
          white-space:pre-wrap;
          background:transparent !important;
          padding:0 !important;
          margin:0 !important;
          text-align: justify;     /* 🟢 Tambahkan ini */
          text-justify: inter-word; /* opsional biar antar kata rata */
      }
    </style>
    """, unsafe_allow_html=True)
    # === OVERRIDE FINAL: hanya untuk tulisan di selectbox & label ===
    st.markdown("""
    <style>
    /* ===== LABEL selectbox (teks "Pilih jenis model...") ===== */
    [data-testid="stSelectbox"] > label,
    [data-testid="stWidgetLabel"] > p {
      color: #C7D2FE !important;        /* bukan putih */
      font-size: 1.15rem !important;    /* lebih besar */
      font-weight: 800 !important;       /* tebal */
    }
    
    /* ===== TEKS DI DALAM KOTAK (placeholder & nilai terpilih) ===== */
    /* ‘palu godam’: paksa SEMUA teks di dalam combobox jadi gelap */
    [data-testid="stSelectbox"] [data-baseweb="select"] * {
      color: #0F172A !important;        /* gelap, bukan putih */
      fill:  #0F172A !important;        /* untuk ikon svg */
      opacity: 1 !important;
      -webkit-text-fill-color: #0F172A !important; /* antisipasi Safari/Chromium */
    }
    
    /* nilai terpilih sering ada di span/value container */
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stSelectbox"] [role="combobox"] *,
    [data-testid="stSelectbox"] [class*="ValueContainer"],
    [data-testid="stSelectbox"] [class*="Placeholder"] {
      color: #0F172A !important;
      -webkit-text-fill-color: #0F172A !important;
      font-weight: 600 !important;
    }
    
    /* ikon caret di kanan */
    [data-testid="stSelectbox"] [data-baseweb="select"] svg {
      color: #0F172A !important;
      opacity: .9 !important;
    }
    
    /* ===== TEKS OPSI SAAT DIBUKA ===== */
    div[data-baseweb="popover"] [role="listbox"] li,
    div[data-baseweb="popover"] [role="listbox"] li * {
      color: #0F172A !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    /* Hilangkan jarak default di bawah dropdown */
    [data-testid="stSelectbox"] {
        margin-bottom: 0 !important;     /* hapus jarak eksternal */
        padding-bottom: 0 !important;    /* hapus jarak internal */
    }
    
    /* kadang wrapper-nya punya padding tambahan */
    [data-testid="stSelectbox"] > div:first-child {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ==== ikon lucide ====
    ICON_BRAIN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10 7a2 2 0 1 1 4 0"/><path d="M7.5 11.5A2.5 2.5 0 0 1 5 9V8a3 3 0 0 1 3-3"/>
    <path d="M16 5a3 3 0 0 1 3 3v1a2.5 2.5 0 0 1-2.5 2.5"/>
    <path d="M8 14a2 2 0 0 0-2 2v1a3 3 0 0 0 3 3h1"/><path d="M16 14a2 2 0 0 1 2 2v1a3 3 0 0 1-3 3h-1"/>
    <path d="M12 5v14"/></svg>"""

    ICON_TARGET = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
    <path d="M22 12h-4"/><path d="M6 12H2"/><path d="M12 6V2"/><path d="M12 22v-4"/></svg>"""

    ICON_LIGHT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 2a7 7 0 0 0-7 7c0 3.09 1.64 5.84 4 7.23V19a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-2.77c2.36-1.39 4-4.14 4-7.23a7 7 0 0 0-7-7z"/>
    <path d="M9 22h6"/></svg>"""

    # ==== render function ====
    def render_model(title, icon, body):
        # pecah teks jika ada “Analogi Sederhana:”
        parts = body.split("Analogi Sederhana:")
        main_part = parts[0]
        illus_part = parts[1] if len(parts) > 1 else ""

        # judul utama
        st.markdown(f"""
        <div class='model-head'>
          <div class='icon-bubble'>{icon}</div>
          <div class='model-title'>{title}</div>
        </div>
        <div class='model-body'>{main_part}</div>
        """, unsafe_allow_html=True)

        # jika ada ilustrasi, tambahkan subjudul dengan ikon lampu
        if illus_part.strip():
            st.markdown(f"""
            <div class='model-head' style='margin-top:15px;'>
              <div class='icon-bubble'>{ICON_LIGHT}</div>
              <div class='model-title'>Analogi Sederhana</div>
            </div>
            <div class='model-body' style='margin-bottom:15px;'>{illus_part}</div>
            """, unsafe_allow_html=True)

    # ==== teks deskripsi ====
    cnn_text = """
Model klasifikasi ini digunakan untuk mengenali jenis tangan pada gambar dan menentukan apakah termasuk kategori batu <i>(rock)</i>, gunting <i>(scissors)</i>, atau kertas <i>(paper)</i>. Model ini bekerja dengan prinsip pengenalan pola visual melalui jaringan saraf tiruan <b>(Convolutional Neural Network/CNN)</b> yang meniru cara kerja otak manusia dalam mengenali bentuk dan pola.<br>
Proses kerjanya dapat dijelaskan sebagai berikut:
• <b>Input gambar:</b> setiap gambar tangan diubah menjadi susunan angka berdasarkan nilai warna dan kecerahan piksel. 
• <b>Ekstraksi pola:</b> lapisan konvolusi menganalisis bagian-bagian kecil dari gambar seperti tepi jari dan lekukan tangan untuk menemukan pola visual yang khas.
• <b>Pembelajaran fitur:</b> semakin dalam lapisan jaringan, semakin kompleks pula pola yang dipelajari — dari garis sederhana hingga keseluruhan bentuk tangan.
• <b>Klasifikasi akhir:</b> hasil pembelajaran dikirim ke lapisan akhir untuk menghitung peluang setiap kelas, kemudian model memilih kategori dengan nilai probabilitas tertinggi.<br>

Dengan tahapan ini, CNN mampu mengenali bentuk tangan secara otomatis tanpa perlu diberi tahu secara eksplisit bagaimana bentuk “batu”, “gunting”, atau “kertas”.

Analogi Sederhana:
Cara kerja CNN dapat dianalogikan seperti seseorang yang belajar mengenali teman-temannya dari foto. Awalnya ia hanya mengenali ciri umum seperti warna rambut atau bentuk wajah, lalu seiring waktu ia mengingat detail seperti mata atau ekspresi. Begitu juga CNN, model ini mempelajari pola sederhana hingga kompleks sehingga mampu mengenali bentuk tangan yang berbeda secara akurat.
"""

    yolo_text = """
Model deteksi digunakan untuk mengenali sekaligus menentukan posisi objek tangan di dalam gambar. Model ini menggunakan algoritma YOLOv8n (<i>You Only Look Once</i> versi 8 – nano), yang dirancang untuk melakukan deteksi secara cepat dan efisien pada berbagai ukuran gambar.<br>

Secara garis besar, cara kerjanya adalah sebagai berikut:
• <b>Pemindaian gambar:</b> model membagi gambar menjadi banyak area kecil, masing-masing dianggap sebagai wilayah kandidat objek. 
• <b>Analisis fitur:</b> setiap area diperiksa untuk melihat apakah pola visualnya menyerupai bentuk tangan.
• <b>Prediksi posisi dan kelas:</b> jika ditemukan kecocokan, model menggambar kotak deteksi <i>(bounding box)</i> dan memberi label seperti “Rock” atau “Scissors”.
• <b>Perhitungan keyakinan:</b> setiap hasil prediksi disertai nilai confidence yang menunjukkan tingkat keyakinan model terhadap deteksi tersebut.<br>

YOLOv8 dapat memproses gambar dalam waktu sangat singkat (hanya beberapa milidetik per gambar) sehingga memungkinkan penggunaan pada sistem real-time seperti kamera interaktif.

Analogi Sederhana:
Cara kerja YOLOv8 dapat dianalogikan seperti seseorang yang sedang mencari wajah temannya di tengah kerumunan. Ia memindai seluruh area pandang dengan cepat, mengenali ciri-ciri yang cocok, lalu menunjuk posisi orang yang dimaksud. Demikian pula YOLOv8, model ini melihat seluruh gambar sekaligus, lalu menandai area yang sesuai dengan pola tangan yang telah ia pelajari.
"""


    # ==== render ====
    if model_choice == "Model Klasifikasi":
        render_model("Model Klasifikasi (CNN)", ICON_BRAIN, cnn_text)
    else:
        render_model("Model Deteksi (YOLOv8n)", ICON_TARGET, yolo_text)
