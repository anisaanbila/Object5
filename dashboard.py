# app.py — RPS Vision Dashboard (Sidebar Logo • Elegant Menu • Topbar • Hints)
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from collections import Counter

st.set_page_config(page_title="RPS Vision Dashboard", page_icon="🧠", layout="wide")

# =========================
# GLOBAL STYLE (semua CSS di dalam satu blok!)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap');

:root{
  --bg1:#010030; --bg2:#160078; --bg3:#7226FF;
  --panel:#12122A; --panel-2:#1A1A34;
  --text:#FFFFFF; --muted:#BBC0E6;
}

/* Typography & base */
*{font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;}
h1{font-weight:800;line-height:1.12;color:var(--text)}
h2,h3,h4{font-weight:700;color:var(--text)}
p,li,div,span,label{font-weight:400;color:var(--text)}
header[data-testid="stHeader"]{display:none;}
.block-container{padding-top:0.1rem!important;max-width:1300px;}

/* Background */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1000px 600px at 15% -10%, rgba(114,38,255,.28), transparent 65%),
    radial-gradient(900px 500px at 90% 10%, rgba(1,0,48,.30), transparent 60%),
    linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%) fixed;
}

/* ==== SIDEBAR (logo kiri atas + judul besar + menu elegan) ==== */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,0));
  border-right:1px solid rgba(255,255,255,.06);
}
.sb-header{
  display:flex; align-items:center; gap:10px;
  padding:14px 10px 0 10px; /* pojok kiri atas */
}
.sb-logo{
  width:64px; height:64px; object-fit:contain; border-radius:12px;
  filter: drop-shadow(0 2px 8px rgba(0,0,0,.35));
}
.sidebar-title{
  font-weight:800;
  font-size:1.45rem;       /* ► diperbesar */
  letter-spacing:.2px;
  margin:8px 10px 0 10px;
  color:#fff;
}
.sb-spacer{ height:14px; } /* jarak antara judul & menu */

.sb-menu [role="radiogroup"] > label{
  padding:6px 10px; border-radius:12px;
  color:#fff; font-weight:500; letter-spacing:.1px;
}
.sb-menu [role="radiogroup"] > label:hover{
  background: rgba(255,255,255,.05);
}
.sb-menu [role="radiogroup"] input{ accent-color:#7226FF; } /* dot pilihan */

/* Top bar */
.topbar{
  position:sticky; top:0; z-index:5; margin:-6px 0 14px 0;
  display:flex; gap:12px; align-items:center; justify-content:space-between;
  background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,0));
  border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:10px 14px;
  box-shadow:0 8px 28px rgba(0,0,0,.28);
}
.tb-left{display:flex; align-items:center; gap:10px}
.search{
  display:flex; align-items:center; gap:8px; min-width:340px;
  background:var(--panel-2); border:1px solid rgba(255,255,255,.08);
  border-radius:999px; padding:8px 12px; color:var(--muted);
}
.search input{all:unset; width:100%; color:#fff}
.tb-right{display:flex; align-items:center; gap:10px}
.badge{background:var(--panel-2); border:1px solid rgba(255,255,255,.1); padding:6px 10px; border-radius:10px; color:#fff}

/* Cards */
.card{
  background:
    linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,0)) padding-box,
    linear-gradient(90deg, rgba(114,38,255,.35), rgba(1,0,48,.35)) border-box;
  border:1px solid transparent; border-radius:18px; padding:22px;
  box-shadow:0 16px 44px rgba(0,0,0,.42);
}
.card-title{font-weight:700;font-size:1.25rem;margin-bottom:.6rem;color:#fff}
.caption{color:var(--muted);font-size:.98rem}

/* Progress bars */
.prog{width:100%;height:12px;border-radius:999px;background:#23234a;overflow:hidden}
.prog>span{display:block;height:100%;width:0%;background:linear-gradient(90deg,#160078,#7226FF);animation:loadWidth 1s ease-out forwards}
@keyframes loadWidth{from{width:0%}to{width:var(--w,0%)}}
.prog-wrap{display:flex;align-items:center;gap:.8rem;margin:.55rem 0}
.prog-wrap .lbl{min-width:160px;font-weight:700;font-size:1.02rem;color:#fff}
.prog-wrap .val{width:78px;text-align:right;color:#fff;font-weight:700;font-variant-numeric:tabular-nums}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Anisa Nabila_Laporan 4.pt")
    clf  = tf.keras.models.load_model("model/Anisa Nabila_Laporan 2.h5")
    return yolo, clf

yolo_model, classifier = load_models()

# =========================
# SIDEBAR (logo kiri atas + judul besar + menu elegan)
# =========================
ICON_PATH = "rps_outline.png"  # logo kecil pojok kiri atas

# header sidebar: logo (ikon saja)
try:
    st.sidebar.markdown("<div class='sb-header'>", unsafe_allow_html=True)
    st.sidebar.image(ICON_PATH, use_container_width=False, width=64, caption=None)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
except Exception:
    pass

# judul besar + jarak
st.sidebar.markdown("<div class='sidebar-title'>RPS Vision</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sb-spacer'></div>", unsafe_allow_html=True)

# menu elegan (tanpa “button look”)
st.sidebar.markdown("<div class='sb-menu'>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Deteksi (YOLOv8)", "Klasifikasi (CNN)", "Penjelasan Model", "Profil Developer"],
    index=0,
    label_visibility="collapsed",
)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.caption("Tip: Gunakan latar gelap untuk konsistensi tampilan.")

# =========================
# TOP BAR (search kiri + ikon)
# =========================
st.markdown("""
<div class="topbar">
  <div class="tb-left">
    <div class="search">
      <svg width="18" height="18" viewBox="0 0 24 24">
        <path d="M21 21l-4.3-4.3M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15z"
              fill="none" stroke="white" stroke-width="1.6" opacity=".9"/>
      </svg>
      <input placeholder="Cari apapun… (opsional)" />
    </div>
  </div>
  <div class="tb-right">
    <div class="badge">🔔</div>
    <div class="badge">👤</div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1>RPS Vision Dashboard</h1>", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def uploader_card(key_label:str, title:str, hint:str):
    """Card uploader dengan hint petunjuk."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='caption'>{hint}</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f

# =========================
# PAGES
# =========================
if page == "Dashboard":
    c1, c2, c3 = st.columns([1.1,1,1])
    with c1:
        st.markdown("<div class='card'><div class='card-title'>Ringkas Deteksi</div><p class='caption'>Objek RPS terdeteksi terakhir & kelas dominan.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='card-title'>Ringkas Klasifikasi</div><p class='caption'>Prediksi utama & skor keyakinan.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><div class='card-title'>Dataset</div><p class='caption'>Total citra & pembagian kelas.</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='card-title'>Preview / Grafik</div><p class='caption'>Tampilkan pratinjau hasil atau grafik performa di sini.</p></div>", unsafe_allow_html=True)

elif page == "Deteksi (YOLOv8)":
    left, right = st.columns([1.04,1])
    with left:
        f = uploader_card(
            "up_yolo",
            "Unggah Gambar • Deteksi (RPS)",
            "Gunakan **latar belakang polos** & pencahayaan cukup agar akurasi bounding box lebih baik."
        )
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil Deteksi</div>", unsafe_allow_html=True)
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
                st.markdown(
                    f"<div class='card-title' style='margin-top:8px'>Prediksi Utama</div>"
                    f"<div class='caption' style='font-size:1.6rem;font-weight:800;color:#fff'>{names[dominant].capitalize()}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("Tidak ada objek terdeteksi pada gambar ini.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Klasifikasi (CNN)":
    left, right = st.columns([1.04,1])
    with left:
        g = uploader_card(
            "up_cls",
            "Unggah Gambar • Klasifikasi (RPS)",
            "Disarankan **background polos** dan tangan memenuhi frame untuk meningkatkan akurasi."
        )
        if g:
            img2 = Image.open(g).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
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

            st.markdown(
                f"<div class='card-title' style='margin-bottom:.2rem'>Prediksi Utama</div>"
                f"<div class='caption' style='font-size:1.6rem;font-weight:800;color:#fff'>{top_name.capitalize()} — {top_prob:.4f}</div>",
                unsafe_allow_html=True
            )

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

elif page == "Penjelasan Model":
    st.markdown("<div class='card'><div class='card-title'>Dokumentasi Singkat</div><p class='caption'>Ringkasan dataset, arsitektur, dan metrik—sesuai versi kamu sebelumnya.</p></div>", unsafe_allow_html=True)

else:  # Profil Developer
    st.markdown("<div class='card'><div class='card-title'>Profil Developer</div>", unsafe_allow_html=True)
    st.markdown("""
• **Nama tampil & panggilan**  
• **Peran** — (contoh: Data Enthusiast & ML Engineer)  
• **Tagline** — 1–2 kalimat  
• **Skill inti (5–8)**  
• **Proyek unggulan (≤3)**  
• **Kontak & tautan** (email, GitHub, LinkedIn/Portofolio)  
• **Riwayat pendidikan** (opsional)
""")
    st.markdown("</div>", unsafe_allow_html=True)
