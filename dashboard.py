# =========================================================
# app.py — RPS Vision Dashboard
# Layout: Top Navigation (no sidebar, no searchbox)
# Theme: Futuristic • Gradient • Poppins
# =========================================================

# == 0. Imports ==
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from collections import Counter

# == 1. App Config ==
st.set_page_config(
    page_title="Rock–Paper–Scissors (RPS) Vision Dashboard",
    page_icon="🧠",
    layout="wide",
)

# == 2. THEME (CSS) ==
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
h1{ font-weight:800; line-height:1.12; color:var(--text); }
h2,h3,h4{ font-weight:700; color:var(--text); }
p,li,div,span,label{ font-weight:400; color:var(--text); }

/* Hide default header & container width */
header[data-testid="stHeader"]{ display:none; }
.block-container{ padding-top:0.1rem!important; padding-bottom:2rem; max-width:1300px; }

/* Futuristic background */
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

/* File uploader text color */
[data-testid="stFileUploader"] section div{ color:#D9DCF6 !important; }

/* Progress bars */
.prog{ width:100%; height:12px; border-radius:999px; background:#23234a; overflow:hidden; }
.prog > span{ display:block; height:100%; width:0%; background:linear-gradient(90deg,#160078,#7226FF); animation: loadWidth 1s ease-out forwards; }
@keyframes loadWidth { from{ width:0% } to{ width:var(--w,0%) } }
.prog-wrap{ display:flex; align-items:center; gap:.8rem; margin:.55rem 0; }
.prog-wrap .lbl{ min-width:160px; font-weight:700; font-size:1.02rem; color:#fff; }
.prog-wrap .val{ width:78px; text-align:right; color:#fff; font-weight:700; font-variant-numeric: tabular-nums; }

/* Architecture flow */
.flow{ position:relative; padding-left:46px; }
.flow:before{ content:""; position:absolute; left:26px; top:6px; bottom:6px; width:4px; background:linear-gradient(#160078,#7226FF); border-radius:4px; }
.flow .node{ position:relative; margin:18px 0; padding-left:0; color:#fff; font-weight:700; font-size:1.05rem;}
.flow .node:before{ content:""; position:absolute; left:-36px; top:2px; width:22px; height:22px; border-radius:50%; border:3px solid rgba(255,255,255,.92); background:rgba(255,255,255,.12); box-shadow:0 0 8px rgba(255,255,255,.35); }

/* Big result title */
.big-result{ font-size:2.2rem; font-weight:800; letter-spacing:.3px; margin:.6rem 0 0 0; color:#fff; }

/* ===== TOP NAV (radio horizontal + profile right) ===== */
.topnav{
  position:sticky; top:0; z-index:5;
  background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,0));
  border:1px solid rgba(255,255,255,.08);
  border-radius:14px; padding:8px 12px; margin:0 0 12px 0;
  box-shadow:0 8px 28px rgba(0,0,0,.28);
}
.topnav-row{ display:flex; align-items:center; justify-content:space-between; gap:12px; }

/* radio → tab elegan */
.topnav .stRadio > div{ display:flex; gap:6px; flex-wrap:wrap; }
.topnav .stRadio label{
  background: transparent; border: none; cursor: pointer;
  color:#fff; font-weight:500; letter-spacing:.1px;
  padding:10px 12px; border-radius:12px; position:relative;
  transition: transform .15s ease;
}
.topnav .stRadio label:hover{ transform: translateY(-1px); }
.topnav .stRadio label::after{
  content:""; position:absolute; left:12px; right:12px; bottom:6px; height:2px;
  background:linear-gradient(90deg, var(--bg1), var(--bg3));
  border-radius:2px; transform:scaleX(0); transform-origin:left center;
  transition: transform .2s ease-in-out;
}
/* item aktif: underline tampil */
.topnav .stRadio [aria-checked="true"] label::after{ transform:scaleX(1); }
/* sembunyikan bullet radio */
.topnav .stRadio input{ display:none !important; }

/* profil (ikon bulat) */
.profile-btn .stButton>button{
  background:var(--panel-2); border:1px solid rgba(255,255,255,.10);
  width:38px; height:38px; border-radius:11px; color:#fff;
  display:flex; align-items:center; justify-content:center;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
}
.profile-btn .stButton>button:hover{ transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# == 3. Load Models ==
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Anisa Nabila_Laporan 4.pt")                    # YOLOv8 detector
    clf  = tf.keras.models.load_model("model/Anisa Nabila_Laporan 2.h5")  # CNN classifier
    return yolo, clf

yolo_model, classifier = load_models()

# == 4. Top Navigation (native, no JS) ==
#   - Kiri: radio horizontal (tabs)
#   - Kanan: tombol profil (ikon)
PAGES = ["Deteksi Objek (YOLOv8)", "Klasifikasi Gambar (CNN)", "Penjelasan Model"]
PROFILE_LABEL = "Profil Developer"

# default pilihan
if "nav_page" not in st.session_state:
    st.session_state.nav_page = PAGES[0]

with st.container():
    st.markdown('<div class="topnav"><div class="topnav-row">', unsafe_allow_html=True)
    col_left, col_right = st.columns([0.86, 0.14])
    with col_left:
        choice = st.radio(
            "Menu",
            PAGES,
            index=PAGES.index(st.session_state.nav_page) if st.session_state.nav_page in PAGES else 0,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.nav_page = choice
    with col_right:
        # ikon user ala lucide (outline)
        user_svg = "👤"  # bisa diganti inline SVG kalau mau persis Lucide
        clicked = st.button(user_svg, key="profile_btn", help="Profil", use_container_width=False)
        if clicked:
            st.session_state.nav_page = PROFILE_LABEL
    st.markdown('</div></div>', unsafe_allow_html=True)

# Judul dinamis
title_map = {
    "Deteksi Objek (YOLOv8)": "Deteksi Objek • YOLOv8",
    "Klasifikasi Gambar (CNN)": "Klasifikasi Gambar • CNN",
    "Penjelasan Model": "Penjelasan Model",
    PROFILE_LABEL: "Profil Developer",
}
st.markdown(f"<h1>{title_map.get(st.session_state.nav_page, 'RPS Vision Dashboard')}</h1>", unsafe_allow_html=True)

# == 5. Small Components / Helpers ==
def uploader_card(key_label:str, title="Unggah Gambar", hint=True):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title' style='font-size:1.35rem'>{title}</div>", unsafe_allow_html=True)
    if hint:
        st.markdown("<div class='caption'>Gunakan <b>latar belakang polos</b> & pencahayaan cukup agar akurasi lebih baik.</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f

def metric_bar(label:str, value:float):
    pct = max(0.0, min(1.0, float(value))) * 100
    st.markdown(
        f"<div class='prog-wrap'><span class='lbl'>{label}</span>"
        f"<div class='prog'><span style='--w:{pct:.2f}%;'></span></div>"
        f"<span class='val'>{pct:.1f}%</span></div>",
        unsafe_allow_html=True
    )

# == 6. Pages ==
page = st.session_state.nav_page

# ---- Page: Deteksi (YOLOv8) ----
if page == "Deteksi Objek (YOLOv8)":
    left, right = st.columns([1.04, 1])
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

# ---- Page: Klasifikasi (CNN) ----
elif page == "Klasifikasi Gambar (CNN)":
    left, right = st.columns([1.04, 1])
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
            with st.spinner("Mengklasifikasikan..."):
                pred = classifier.predict(arr)
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

# ---- Page: Penjelasan Model ----
elif page == "Penjelasan Model":
    model_choice = st.selectbox("Pilih model yang ingin dijelaskan", ["YOLOv8", "CNN"], index=0)

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
    icons_small = {
      "Rock": """<path d="M18,30 c-4,0 -8,-3 -8,-7 v-8 c0-6 16-6 16,2 v6 c0,4 -4,7 -8,7z" stroke="white" stroke-width="3" fill="none"/>""",
      "Paper": """<path d="M14,30 c-3,-10 2,-18 8,-18 5,0 6,5 6,10 v8" stroke="white" stroke-width="3" fill="none"/><path d="M10,26 c-2,-7 1,-12 6,-12" stroke="white" stroke-width="3" fill="none"/>""",
      "Scissors": """<path d="M10,12 l8,12 M22,12 l-6,10 M12,26 c4,4 10,4 12,0" stroke="white" stroke-width="3" fill="none"/>"""
    }
    for (k,v), col in zip(counts.items(), colc):
        col.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-top:10px;">
          <div class="icon-bubble"><svg viewBox="0 0 36 36">{icons_small[k]}</svg></div>
          <div><div style="font-weight:700;font-size:1.05rem">{k}</div>
               <div style="font-weight:800;font-size:1.6rem">{v:,}</div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='card'><div class='card-title'>Arsitektur</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='card'><div class='card-title'>Evaluasi</div>", unsafe_allow_html=True)
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
            metric_bar("Latency (skala cepat)", 1-0.017)
            st.markdown("Akurat & cepat — layak untuk **real-time**.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>Kesimpulan</div>", unsafe_allow_html=True)
    st.markdown("**YOLOv8n** presisi tinggi (**mAP@50 ≈ 0.995**) + CNN akurasi ~**94%**. Cocok untuk RPS real-time.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Page: Profil ----
elif page == "Profil Developer":
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
