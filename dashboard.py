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
# HEADER (left text + PNG icon on right, NO BOX)
# =========================
ICON_PATH = "rps_outline.png"  # file sejajar dengan dashboard.py

c1, c2 = st.columns([1.6, 1.0], vertical_alignment="center")

with c1:
    st.markdown(
        "<div class='card'>"
        "<div class='card-title'>RPS Vision Dashboard</div>"
        "<h1>Detection & Classification for<br/>Rock–Paper–Scissors (RPS)</h1>"
        "<p class='caption'>Dashboard futuristik untuk <b>deteksi objek</b> (YOLOv8) dan "
        "<b>klasifikasi gambar</b> (CNN) pada gestur tangan RPS.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with c2:
    # Naikkan posisi ikon header biar sejajar: align ke atas + margin-top negatif
    st.markdown(
        """
        <style>
          .header-rps-wrap{
              display:flex;
              justify-content:center;
              align-items:flex-start;   /* ratakan ke atas kolom */
              margin-top:-72px;         /* ⬅️ naikkan; sesuaikan -56 / -64 / -80 kalau perlu */
          }
          .header-rps-img{
              max-width:360px;
              width:100%;
              height:auto;
              filter:drop-shadow(0 0 18px rgba(255,255,255,.28))
                     drop-shadow(0 0 6px rgba(255,255,255,.25));
          }
          /* Responsif: di layar kecil jangan terlalu naik */
          @media (max-width: 1200px){
            .header-rps-wrap{ margin-top:-40px; }
          }
          @media (max-width: 992px){
            .header-rps-wrap{ margin-top:-16px; }
          }
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
# TABS (white titles, non-bold)
# =========================
tab_det, tab_cls, tab_profile, tab_docs = st.tabs([
    "Deteksi Objek (YOLOv8)", "Klasifikasi Gambar (CNN)", "Profil Developer", "Penjelasan Model"
])

def uploader_card(key_label:str, title="Unggah Gambar"):
    st.markdown(f"<div class='card'><div class='card-title' style='font-size:1.35rem'>{title}</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f

# =========================
# TAB: DETEKSI (clean, elegant: no sliders/summary/table)
# =========================
with tab_det:
    left, right = st.columns([1.04,1])
    with left:
        f = uploader_card("up_yolo", "Unggah Gambar • Deteksi (RPS)")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-title' style='font-size:1.35rem'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='card-title' style='font-size:1.35rem'>Hasil Deteksi</div>", unsafe_allow_html=True)
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
                st.markdown(f"<div class='big-result'>Prediksi Utama ⮕ {names[dominant].capitalize()}</div>", unsafe_allow_html=True)
            else:
                st.info("Tidak ada objek terdeteksi pada gambar ini.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB: KLASIFIKASI (progress + tabel)
# =========================
with tab_cls:
    left, right = st.columns([1.04,1])
    with left:
        g = uploader_card("up_cls", "Unggah Gambar • Klasifikasi (RPS)")
        if g:
            img2 = Image.open(g).convert("RGB")
            st.markdown("<div class='card'><div class='card-title' style='font-size:1.35rem'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='card-title' style='font-size:1.35rem'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
        if not g:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            img_resized = img2.resize((224,224))
            arr = image.img_to_array(img_resized); arr = np.expand_dims(arr,0)/255.0
            with st.spinner("Mengklasifikasikan..."): pred = classifier.predict(arr)
            probs = pred[0].astype(float)
            labels = ["paper","rock","scissors"] if len(pred[0])==3 else [f"class_{i}" for i in range(len(pred[0]))]
            top_idx = int(np.argmax(probs)); top_name = labels[top_idx]; top_prob = float(probs[top_idx])

            st.markdown(f"<div class='big-result'>Prediksi Utama ⮕ {top_name.capitalize()}</div>", unsafe_allow_html=True)
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
# TAB: PROFIL DEVELOPER (prompt)
# =========================
with tab_profile:
    st.markdown("<div class='card'><div class='card-title'>Profil Developer — Mohon jawab di chat</div>", unsafe_allow_html=True)
    st.markdown("""
• **Nama yang ditampilkan** & panggilan  
• **Peran/role utama**  
• **Tagline singkat** (1–2 kalimat)  
• **Skill inti (5–8)**  
• **Proyek unggulan (≤3)**  
• **Kontak & tautan** (email, GitHub, LinkedIn/Portofolio)  
• **Riwayat pendidikan** (opsional) dalam format timeline  
• **Preferensi warna/aksen tambahan** (bila ada)
""")
    st.markdown("</div>", unsafe_allow_html=True)

