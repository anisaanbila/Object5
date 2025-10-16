# app.py — RPS Vision Dashboard (Futuristic • Gradient • Poppins) — FINAL
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Rock–Paper–Scissors (RPS) Vision Dashboard", page_icon="🧠", layout="wide")

# =========================
# THEME (gradient + Poppins)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root{
  --bg1:#010030; --bg2:#160078; --bg3:#7226FF;
  --panel:#12122A; --panel-2:#1A1A34;
  --text:#E9E9F6; --muted:#A3A6C2;
  --neon:#FFFFFF;
}
* { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
h1{ font-weight:800; line-height:1.12; } 
h2,h3,h4{ font-weight:800; }
p,li,div,span{ font-weight:400; }

header[data-testid="stHeader"]{ display:none; }
.block-container{ padding-top:3.6rem!important; padding-bottom:2rem; max-width:1300px; }

/* Futuristic gradient background + subtle grid */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 20% -10%, rgba(114,38,255,.22), transparent 65%),
    radial-gradient(900px 500px at 90% 10%, rgba(1,0,48,.28), transparent 60%),
    linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%) fixed;
  color: var(--text);
}

/* Cards with neon-glass effect + hover glow */
.card{
  position:relative;
  background:
    linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0)) padding-box,
    linear-gradient(90deg, rgba(114,38,255,.35), rgba(1,0,48,.35)) border-box;
  border:1px solid transparent; border-radius:18px; padding:22px 22px;
  box-shadow: 0 16px 44px rgba(0,0,0,.42);
  overflow:hidden; transition: box-shadow .25s ease, transform .25s ease;
}
.card:hover{ box-shadow:0 26px 60px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,.06) inset; transform: translateY(-1px); }
.card.compact{ padding:16px 18px; }
.card-title{ font-weight:800; font-size:1.2rem; color:#FFFFFF; margin-bottom:.7rem; letter-spacing:.15px; }
.caption{ color: var(--muted); font-size:.95rem; }
hr{ border-color:#2b2c44; }

/* Tabs — text white normal (no bold) */
.stTabs [role="tablist"]{ gap:1rem; }
.stTabs [role="tab"]{
  color:#FFFFFF !important; font-weight:500; letter-spacing:.2px;
  border-bottom:2px solid transparent;
}
.stTabs [role="tab"][aria-selected="true"]{
  border-bottom:2px solid; border-image: linear-gradient(90deg,#010030,#7226FF) 1;
}

/* File uploader text color */
[data-testid="stFileUploader"] section div{ color:#A3A6C2!important; }

/* Progress bars (classification & evaluation) */
.prog{ width:100%; height:12px; border-radius:999px; background:#23234a; overflow:hidden; }
.prog > span{
  display:block; height:100%; width:0%;
  background:linear-gradient(90deg,#160078,#7226FF);
  animation: loadWidth 1s ease-out forwards;
}
@keyframes loadWidth { from{ width:0% } to{ width:var(--w,0%) } }
.prog-wrap{ display:flex; align-items:center; gap:.8rem; margin:.5rem 0; }
.prog-wrap .lbl{ min-width:150px; font-weight:700; font-size:1rem; color:#FFFFFF; }
.prog-wrap .val{ width:72px; text-align:right; color:#FFFFFF; font-weight:600; font-variant-numeric: tabular-nums; }

/* Chips (for detections) */
.chips{ display:flex; gap:.5rem; flex-wrap:wrap; }
.chip{
  padding:.3rem .65rem; border-radius:999px;
  background:rgba(114,38,255,.18); border:1px solid rgba(114,38,255,.35);
  font-weight:600; font-size:.9rem; color:#E9E9F6;
}

/* Number badge (dataset count) */
.dataset-count{ display:flex; align-items:center; gap:1rem; }
.dataset-icon{
  width:68px; height:68px; border-radius:50%;
  background: radial-gradient(circle at 50% 50%, rgba(255,255,255,.12), rgba(255,255,255,.02));
  border:2px solid rgba(255,255,255,.65);
  box-shadow: 0 0 18px rgba(255,255,255,.25), inset 0 0 10px rgba(255,255,255,.12);
  display:flex; align-items:center; justify-content:center;
}
.dataset-icon:before{
  content:""; width:36px; height:26px; border:2px solid rgba(255,255,255,.8);
  border-radius:6px; box-shadow: inset 0 0 6px rgba(255,255,255,.25);
}
/* Vertical flow (architecture) */
.flow{ position:relative; padding-left:34px; }
.flow:before{ content:""; position:absolute; left:13px; top:0; bottom:0; width:4px; background:linear-gradient(#160078,#7226FF); border-radius:4px; }
.flow .node{
  position:relative; margin:16px 0; padding-left:10px; 
  font-weight:600; color:#FFFFFF;
}
.flow .node:before{
  content:""; position:absolute; left:-25px; top:2px; width:18px; height:18px; border-radius:50%;
  border:2px solid rgba(255,255,255,.85); background:rgba(255,255,255,.15); box-shadow:0 0 8px rgba(255,255,255,.35);
}

/* SVG RPS circle (header) responsive wrapper */
.rps-wrap{ display:flex; justify-content:center; align-items:center; }
.rps-svg{ width:100%; max-width:340px; height:auto; filter: drop-shadow(0 0 14px rgba(255,255,255,.18)); }

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
# HEADER (judul kiri + SVG RPS outline kanan)
# =========================
c1, c2 = st.columns([1.6,1.0], vertical_alignment="center")
with c1:
    st.markdown(
        "<div class='card'>"
        "<div class='card-title'>RPS Vision Dashboard</div>"
        "<h1>Detection & Classification for<br/>Rock–Paper–Scissors (RPS)</h1>"
        "<p class='caption'>Dashboard futuristik untuk <b>deteksi objek</b> (YOLOv8) dan <b>klasifikasi gambar</b> (CNN) pada gestur tangan RPS.</p>"
        "</div>", unsafe_allow_html=True)

with c2:
    # SVG: neon white outline circle with 3 nodes R P S
    st.markdown("""
    <div class="card compact rps-wrap">
      <svg class="rps-svg" viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <circle cx="150" cy="150" r="100" stroke="white" stroke-opacity="0.9" stroke-width="3" fill="none" filter="url(#glow)"/>
        <!-- nodes -->
        <circle cx="150" cy="45" r="16" stroke="white" stroke-width="3" fill="rgba(255,255,255,0.12)"/>
        <circle cx="245" cy="180" r="16" stroke="white" stroke-width="3" fill="rgba(255,255,255,0.12)"/>
        <circle cx="55"  cy="180" r="16" stroke="white" stroke-width="3" fill="rgba(255,255,255,0.12)"/>
        <!-- labels -->
        <text x="150" y="50" text-anchor="middle" fill="#FFFFFF" font-size="14" font-weight="700">R</text>
        <text x="245" y="185" text-anchor="middle" fill="#FFFFFF" font-size="14" font-weight="700">P</text>
        <text x="55"  y="185" text-anchor="middle" fill="#FFFFFF" font-size="14" font-weight="700">S</text>
      </svg>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# TABS (white, non-bold)
# =========================
tab_det, tab_cls, tab_profile, tab_docs = st.tabs([
    "Deteksi Objek (YOLOv8)", "Klasifikasi Gambar (CNN)", "Profil Developer", "Penjelasan Model"
])

def uploader_card(key_label:str, title="Unggah Gambar"):
    st.markdown(f"<div class='card'><div class='card-title' style='font-size:1.25rem'>{title}</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f

# =========================
# TAB: DETEKSI (ringkasan elegan)
# =========================
with tab_det:
    left, right = st.columns([1.04,1])
    with left:
        f = uploader_card("up_yolo", "Unggah Gambar • Deteksi (RPS)")
        conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.01)
        iou  = st.slider("IoU NMS", 0.1, 0.9, 0.45, 0.01)
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-title' style='font-size:1.25rem'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='card-title' style='font-size:1.25rem'>Hasil Deteksi</div>", unsafe_allow_html=True)
        if not f:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan deteksi.</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Menjalankan YOLOv8..."):
                res = yolo_model.predict(img, conf=conf, iou=iou, verbose=False)
                plotted = res[0].plot()
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, use_container_width=True, caption="Bounding boxes")

            # Ringkasan elegan
            names = res[0].names
            boxes = res[0].boxes
            if boxes is not None and len(boxes)>0:
                cls_ids = [int(c) for c in boxes.cls.tolist()]
                confs   = boxes.conf.tolist()
                st.markdown("**Ringkasan Deteksi**", unsafe_allow_html=True)
                st.markdown("<div class='chips'>" + "".join([f"<span class='chip'>{names[c]}</span>" for c in cls_ids]) + "</div>", unsafe_allow_html=True)
                counts = Counter(cls_ids)
                rows = []
                for cid, cnt in counts.items():
                    avg_conf = np.mean([c for i,c in enumerate(confs) if cls_ids[i]==cid])
                    rows.append([names[cid].capitalize(), int(cnt), float(avg_conf)])
                df = pd.DataFrame(rows, columns=["Kelas", "Jumlah", "Rata-rata Confidence"])
                df["Rata-rata Confidence (%)"] = (df["Rata-rata Confidence"]*100).round(2)
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df[["Kelas","Jumlah","Rata-rata Confidence (%)"]], hide_index=True, use_container_width=True)
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
            st.markdown("<div class='card'><div class='card-title' style='font-size:1.25rem'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='card-title' style='font-size:1.25rem'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
        if not g:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            img_resized = img2.resize((224,224))
            arr = image.img_to_array(img_resized); arr = np.expand_dims(arr,0)/255.0
            with st.spinner("Mengklasifikasikan..."): pred = classifier.predict(arr)
            probs = pred[0].astype(float)
            labels = ["paper","rock","scissors"]
            if len(probs) != len(labels):
                labels = [f"class_{i}" for i in range(len(probs))]
            top_idx = int(np.argmax(probs)); top_name = labels[top_idx]; top_prob = float(probs[top_idx])

            st.markdown(f"### Prediksi Utama: **{top_name.capitalize()}** — Rock–Paper–Scissors (RPS)")
            st.markdown(f"Skor keyakinan: **{top_prob:.4f}**")

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
    st.markdown("<div class='card'><div class='card-title' style='font-size:1.25rem'>Profil Developer — Mohon jawab di chat</div>", unsafe_allow_html=True)
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

# =========================
# TAB: PENJELASAN MODEL (dropdown + per-box)
# =========================
with tab_docs:
    model_choice = st.selectbox("Pilih model yang ingin dijelaskan", ["CNN", "YOLOv8"], index=0)

    def metric_bar(label:str, value:float):
        pct = max(0.0, min(1.0, float(value))) * 100
        st.markdown(
            f"<div class='prog-wrap'><span class='lbl'>{label}</span>"
            f"<div class='prog'><span style='--w:{pct:.2f}%;'></span></div>"
            f"<span class='val'>{pct:.1f}%</span></div>",
            unsafe_allow_html=True
        )

    # ---- Box 1: Dataset (judul di dalam box + ikon & jumlah)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Dataset</div>", unsafe_allow_html=True)
    if model_choice == "CNN":
        st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (RPS) – Dicoding**. Tiga kelas: *paper* (712), *rock* (726), *scissors* (750); total **2.188** gambar.

**Split & Prapemrosesan.**
- **70%** latih, **20%** validasi, **10%** uji  
- **Resize** ke **224×224** (RGB), **normalisasi** 0–1  
- **Augmentasi** (latih saja): rotasi ≤10°, zoom ≤10%, horizontal flip
        """)
        total_images = 2188
    else:
        st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (RPS) – Dicoding** dengan anotasi **bounding box** via **Roboflow**.

**Split & Ukuran.**
- Semua citra diubah ke **640×640**  
- Split: **80%** latih, **10%** validasi, **10%** uji  
- Struktur anotasi kompatibel format **YOLOv8** (anchor-free)
        """)
        total_images = 2188

    # ikon gambar + angka
    st.markdown(f"""
    <div class="dataset-count" style="margin-top:.6rem">
      <div class="dataset-icon"></div>
      <div>
        <div style="font-weight:700;font-size:1.1rem">Total Gambar</div>
        <div style="font-weight:800;font-size:1.6rem">{total_images:,}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Box 2 & 3: Arsitektur + Evaluasi
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Arsitektur</div>", unsafe_allow_html=True)
        if model_choice == "CNN":
            st.markdown("""
Berikut alur layer dari input hingga prediksi (diagram alur vertikal):
            """)
            st.markdown("""
<div class="flow">
  <div class="node">Conv2D(32, 3×3, ReLU) → MaxPool(2×2)</div>
  <div class="node">Conv2D(64, 3×3, ReLU) → MaxPool(2×2)</div>
  <div class="node">Conv2D(128, 3×3, ReLU) → MaxPool(2×2)</div>
  <div class="node">Flatten</div>
  <div class="node">Dense(128, ReLU) → Dropout(0.5)</div>
  <div class="node">Dense(3, Softmax)</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Pelatihan**
- Optimizer **Adam**, loss **categorical_crossentropy**  
- **EarlyStopping** (monitor *val_loss*) dan **ModelCheckpoint** (bobot terbaik)
            """)
        else:
            st.markdown("""
Struktur inti **YOLOv8n (anchor-free)** ditunjukkan sebagai alur dari Backbone ➜ Neck ➜ Head:
            """)
            st.markdown("""
<div class="flow">
  <div class="node">Backbone (SiLU, C2f, SPPF)</div>
  <div class="node">Neck (FPN/PAN, multi-scale fusion)</div>
  <div class="node">Head (stride 8/16/32, cls+box)</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Inferensi (UI)** — default **confidence 0.25**, **IoU NMS 0.45** (bisa diatur di tab Deteksi).
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Evaluasi</div>", unsafe_allow_html=True)
        if model_choice == "CNN":
            st.markdown("Metrik penting (validasi terbaik):")
            metric_bar("Accuracy", 0.94)
            metric_bar("Precision (macro)", 0.94)
            metric_bar("Recall (macro)", 0.94)
            metric_bar("F1-score (macro)", 0.94)
            metric_bar("Val Loss (↓)", 1-0.94)  # presentasi relatif, untuk visual saja
            st.markdown("Model konsisten di seluruh kelas; tidak ada bias dominan.")
        else:
            st.markdown("Metrik validasi & performa:")
            metric_bar("Precision", 0.996)
            metric_bar("Recall", 1.00)
            metric_bar("mAP@50", 0.995)
            metric_bar("mAP@50–95", 0.925)
            metric_bar("Latency (kecepatan) — skala", 1-0.017)  # 17ms → semakin kecil semakin cepat
            st.markdown("Deteksi sangat akurat sekaligus cepat — cocok untuk **real-time**.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Box 4: Kesimpulan
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Kesimpulan</div>", unsafe_allow_html=True)
    if model_choice == "CNN":
        st.markdown("""
CNN yang ringkas (tiga blok konvolusi + **Dropout 0.5**) dengan **EarlyStopping** dan **ModelCheckpoint**
memberi akurasi ~**94%** pada **Rock–Paper–Scissors (RPS)**. Cocok sebagai **pengklasifikasi akhir** (mis. setelah ROI dari detektor) atau aplikasi edukasi.
        """)
    else:
        st.markdown("""
**YOLOv8n** mencapai presisi tinggi (mAP@50 ~**0.995**) dengan latensi rendah (~**17 ms/gambar**).
Arsitektur **FPN/PAN + head anchor-free** efektif untuk variasi ukuran tangan, ideal untuk **deteksi real-time** di kamera/stream.
        """)
    st.markdown("</div>", unsafe_allow_html=True)
