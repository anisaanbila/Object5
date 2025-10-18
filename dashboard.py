# app.py — RPS Vision Dashboard (Top Navbar • Lucide-like Icons • No Sidebar)
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
# CSS (single block) — warna tetap pakai variabel kamu
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap');

:root{
  --bg1:#010030; --bg2:#160078; --bg3:#7226FF;
  --panel:#12122A; --panel-2:#1A1A34;
  --text:#FFFFFF; --muted:#BBC0E6;
}

*{font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;}
h1{font-weight:800;line-height:1.12;color:var(--text); margin:.8rem 0 .6rem;}
h2,h3,h4{font-weight:700;color:var(--text)}
p,li,div,span,label{font-weight:400;color:var(--text)}
header[data-testid="stHeader"]{display:none;}
.block-container{padding-top:0.1rem!important;max-width:1300px;}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1000px 600px at 15% -10%, rgba(114,38,255,.28), transparent 65%),
    radial-gradient(900px 500px at 90% 10%, rgba(1,0,48,.30), transparent 60%),
    linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%) fixed;
}

/* ============ NAVBAR ============ */
.navbar{
  position:sticky; top:0; z-index:5;
  display:flex; align-items:center; justify-content:space-between;
  gap:12px; padding:10px 16px; margin:0 0 10px 0;
  background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,0));
  border:1px solid rgba(255,255,255,.08);
  border-radius:14px; box-shadow:0 8px 28px rgba(0,0,0,.28);
}

/* Left nav (tabs) */
.nav-left{ display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.nav-item{
  display:flex; align-items:center; gap:8px;
  padding:10px 12px; border-radius:12px;
  color:#fff; text-decoration:none; position:relative;
  transition: transform .15s ease;
}
.nav-item:hover{ transform: translateY(-1px); }
.nav-item svg{ width:18px; height:18px; stroke:#fff; opacity:.9; }

/* Underline gradient on hover/active */
.nav-item::after{
  content:""; position:absolute; left:12px; right:12px; bottom:6px; height:2px;
  background:linear-gradient(90deg, var(--bg1), var(--bg3));
  border-radius:2px; transform:scaleX(0); transform-origin:left center;
  transition: transform .2s ease-in-out;
}
.nav-item:hover::after{ transform:scaleX(1); }
.nav-item.active::after{ transform:scaleX(1); }

/* Right (profile) */
.nav-right{ display:flex; align-items:center; gap:10px; }
.nav-icon{
  display:flex; align-items:center; justify-content:center;
  width:38px; height:38px; border-radius:11px;
  background:var(--panel-2); border:1px solid rgba(255,255,255,.10);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
}
.nav-icon svg{ width:20px; height:20px; stroke:#fff; }

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
# Helpers — query param (no button) untuk navigasi
# =========================
def get_query_page():
    # Streamlit API baru
    page = None
    try:
        page = st.query_params.get("page", None)
    except Exception:
        pass
    if not page:
        # fallback API lama
        q = st.experimental_get_query_params()
        page = q.get("page", ["Dashboard"])[0] if q else "Dashboard"
    return page

# =========================
# QUERY PARAM (pakai API baru saja, aman & tanpa error)
# =========================
def get_page():
    try:
        p = st.query_params.get("page", "Dashboard")
    except Exception:
        p = "Dashboard"
    # kadang return list → ambil elemen pertama
    if isinstance(p, (list, tuple)):
        p = p[0] if p else "Dashboard"
    return p

page = get_page()



# =========================
# NAVBAR HTML (ikon outline ala Lucide via inline SVG)
# =========================
def nav_item(href, label, icon_svg, active=False):
    cls = "nav-item active" if active else "nav-item"
    return f"""
      <a class="{cls}" href="?page={href}">
        {icon_svg}<span>{label}</span>
      </a>
    """

icons = {
  "home":   '<svg viewBox="0 0 24 24" fill="none"><path d="M3 10.5L12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1v-9.5Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  "camera": '<svg viewBox="0 0 24 24" fill="none"><path d="M4 8h4l2-3h4l2 3h4v10H4V8Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="12" cy="13" r="3.5" stroke="currentColor" stroke-width="1.8"/></svg>',
  "image":  '<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" stroke-width="1.8"/><path d="M7 15l3-3 3 3 4-4 2 2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="8" cy="9" r="1.5" stroke="currentColor" stroke-width="1.6"/></svg>',
  "book":   '<svg viewBox="0 0 24 24" fill="none"><path d="M5 4h10a3 3 0 0 1 3 3v13H8a3 3 0 0 0-3 3V4Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 18h13" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  "user":   '<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="3.5" stroke="currentColor" stroke-width="1.8"/><path d="M5 20a7 7 0 0 1 14 0" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>'
}

st.markdown(f"""
<div class="navbar">
  <div class="nav-left">
    {nav_item("Dashboard","Dashboard", icons['home'],   active=(page=="Dashboard"))}
    {nav_item("Deteksi","Deteksi (YOLOv8)", icons['camera'], active=(page=="Deteksi"))}
    {nav_item("Klasifikasi","Klasifikasi (CNN)", icons['image'], active=(page=="Klasifikasi"))}
    {nav_item("Penjelasan","Penjelasan Model", icons['book'], active=(page=="Penjelasan"))}
  </div>
  <div class="nav-right">
    <a class="nav-icon" href="?page=Profil" title="Profil">
      {icons['user']}
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
title_map = {
  "Dashboard":  "RPS Vision Dashboard",
  "Deteksi":    "Deteksi Objek • YOLOv8",
  "Klasifikasi":"Klasifikasi Gambar • CNN",
  "Penjelasan": "Penjelasan Model",
  "Profil":     "Profil Developer",
}
st.markdown(f"<h1>{title_map.get(page,'RPS Vision Dashboard')}</h1>", unsafe_allow_html=True)

# =========================
# Load Models
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Anisa Nabila_Laporan 4.pt")
    clf  = tf.keras.models.load_model("model/Anisa Nabila_Laporan 2.h5")
    return yolo, clf

yolo_model, classifier = load_models()

# =========================
# Helpers
# =========================
def uploader_card(key_label:str, title:str, hint:str):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='caption'>{hint}</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f

# =========================
# Routing (berdasarkan ?page=...)
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

elif page == "Deteksi":
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

elif page == "Klasifikasi":
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

elif page == "Penjelasan":
    st.markdown("<div class='card'><div class='card-title'>Dokumentasi Singkat</div><p class='caption'>Ringkasan dataset, arsitektur, dan metrik—sesuai versi kamu sebelumnya.</p></div>", unsafe_allow_html=True)

elif page == "Profil":
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
