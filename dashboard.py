# app.py — RPS Vision Dashboard (Futuristic · Gradient · Poppins)
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd

# -----------------------------
# PAGE
# -----------------------------
st.set_page_config(
    page_title="RPS Vision Dashboard — Detection & Classification",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# THEME & CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root{
  --bg1:#010030;  --bg2:#160078;  --bg3:#7226FF; /* gradient */
  --panel:#12122A; --panel-2:#1A1A34;
  --text:#E9E9F6; --muted:#A3A6C2;
  --border:#2b2c44;
}

*{ font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; }
h1{ font-weight:800; font-size:2.1rem; letter-spacing:.2px; }
h2{ font-weight:700; font-size:1.4rem; margin:.2rem 0 .6rem 0; }
h3{ font-weight:700; font-size:1.2rem; margin:.2rem 0 .5rem 0; }
p,li,div{ font-weight:400; }

header[data-testid="stHeader"]{ display:none; }
.block-container{ padding-top:3.8rem!important; padding-bottom:2rem; max-width:1300px; }

/* Futuristic gradient + subtle grid background */
[data-testid="stAppViewContainer"]{
  position:relative;
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(114,38,255,.18), transparent 60%),
    linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%),
    #0b0b1c;
  color:var(--text);
}
[data-testid="stAppViewContainer"]::before{
  content:"";
  position:fixed; inset:0; pointer-events:none; opacity:.28;
  background-image:
    linear-gradient(to right, rgba(255,255,255,.08) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255,255,255,.06) 1px, transparent 1px);
  background-size: 40px 40px; /* subtle grid */
  mask-image: radial-gradient(70% 70% at 50% 30%, #000 60%, transparent 100%);
}

/* Cards (glass + gradient border) */
.card{
  background:
    linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01)) padding-box,
    linear-gradient(90deg, rgba(1,0,48,.5), rgba(114,38,255,.5)) border-box;
  border:1px solid transparent; border-radius:16px; padding:18px 20px;
  box-shadow: 0 12px 28px rgba(0,0,0,.35);
  backdrop-filter: saturate(130%) blur(4px);
}
.card.compact{ padding:14px 16px; }
.card-title{ font-weight:700; font-size:1.05rem; color:#D7DAFF; margin-bottom:.55rem; }
.caption{ color:var(--muted); font-size:.9rem; }
hr{ border-color:var(--border); }

/* Chips */
.pill{
  background: linear-gradient(90deg, rgba(1,0,48,.25), rgba(114,38,255,.25));
  border:1px solid rgba(150,150,220,.35); color:#D7DAFF;
  padding:.35rem .7rem; border-radius:999px; font-weight:600; font-size:.82rem;
}
.kpi{ display:flex; gap:.6rem; align-items:center; padding:.6rem .9rem;
     background:#101026; border:1px solid var(--border); border-radius:12px; }
.kpi .big{ font-weight:800; font-size:1.05rem; }

/* Tabs underline */
.stTabs [role="tablist"]{ gap:1rem; }
.stTabs [role="tab"]{ border-bottom:2px solid transparent; }
.stTabs [role="tab"][aria-selected="true"]{
  border-bottom:2px solid; border-image: linear-gradient(90deg,#010030,#7226FF) 1;
}

/* File uploader */
[data-testid="stFileUploader"] section div{ color:#A3A6C2!important; }

/* Progress bars (classification & evaluation) */
.prog{ width:100%; height:12px; border-radius:999px; background:#20204a; overflow:hidden;
       box-shadow: inset 0 0 0 1px rgba(255,255,255,.04); }
.prog > span{
  display:block; height:100%;
  background:linear-gradient(90deg,#160078,#7226FF);
  filter: drop-shadow(0 0 6px rgba(114,38,255,.45));
  width:0%;
  transition: width .6s ease;
}
.prog-wrap{ display:flex; align-items:center; gap:.7rem; margin:.42rem 0; }
.prog-wrap .lbl{ min-width:120px; font-weight:600; font-size:.95rem; color:#E5E6FF; letter-spacing:.2px; }
.prog-wrap .val{ width:60px; text-align:right; color:#E5E6FF; font-variant-numeric: tabular-nums; }

/* Selectbox weight */
[data-baseweb="select"] div{ font-weight:600; }

/* Section separators */
.section-head{ display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:.4rem; }
.section-head .sub{ color:var(--muted); font-size:.95rem; }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODELS
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Anisa Nabila_Laporan 4.pt")                    # YOLOv8 detector
    clf  = tf.keras.models.load_model("model/Anisa Nabila_Laporan 2.h5")  # CNN classifier
    return yolo, clf

yolo_model, classifier = load_models()

# -----------------------------
# HEADER
# -----------------------------
c1, c2 = st.columns([1.9,1.1])
with c1:
    st.markdown(
        "<div class='card'>"
        "<div class='card-title'>RPS Vision Dashboard</div>"
        "<h1 style='margin:0 0 .4rem 0;'>Detection & Classification for Rock–Paper–Scissors</h1>"
        "<div class='caption'>Antarmuka futuristik untuk deteksi objek (YOLOv8) dan klasifikasi gambar (CNN) pada gestur tangan Rock–Paper–Scissors.</div>"
        "</div>", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class='card compact'>
      <div class='card-title'>Model Status</div>
      <div class='kpi'>🟢<span class='big'>Ready</span><span class='caption'>YOLOv8 & CNN aktif</span></div>
      <div style="margin-top:.6rem;display:flex;gap:.5rem;flex-wrap:wrap;">
        <span class='pill'>Poppins</span><span class='pill'>Gradient UI</span><span class='pill'>RPS • 3 Kelas</span>
      </div>
    </div>""", unsafe_allow_html=True)

# Mini stat cards untuk mengisi ruang & memberi kesan “hidup”
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("<div class='card compact'><div class='card-title'>Pipeline</div><div class='caption'>YOLOv8 (deteksi) → ROI → CNN (klasifikasi)</div></div>", unsafe_allow_html=True)
with m2:
    st.markdown("<div class='card compact'><div class='card-title'>Dataset</div><div class='caption'>Dicoding RPS · 2.188 gambar · 3 kelas</div></div>", unsafe_allow_html=True)
with m3:
    st.markdown("<div class='card compact'><div class='card-title'>Tema</div><div class='caption'>#010030 → #160078 → #7226FF</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab_det, tab_cls, tab_profile, tab_docs = st.tabs([
    "Deteksi Objek (YOLOv8)", "Klasifikasi Gambar (CNN)", "Profil Developer", "Penjelasan Model"
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
        f = uploader_card("up_yolo", "Unggah Gambar • Deteksi (RPS)")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil Deteksi</div>", unsafe_allow_html=True)
        if not f:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan deteksi.</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Menjalankan YOLOv8..."):
                res = yolo_model(img)
                plotted = res[0].plot()
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, use_container_width=True, caption="Bounding boxes")
            names, boxes = res[0].names, res[0].boxes
            if boxes is not None and len(boxes)>0:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("**Ringkasan Deteksi:**")
                for cid, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    st.write(f"• {names[int(cid)]} — confidence: {conf:.2f}")
            else:
                st.info("Tidak ada objek terdeteksi.")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB: KLASIFIKASI
# -----------------------------
with tab_cls:
    left, right = st.columns([1.04,1])
    with left:
        g = uploader_card("up_cls", "Unggah Gambar • Klasifikasi (RPS)")
        if g:
            img2 = Image.open(g).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
        if not g:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            # Prediksi
            img_resized = img2.resize((224,224))
            arr = image.img_to_array(img_resized); arr = np.expand_dims(arr,0)/255.0
            with st.spinner("Mengklasifikasikan..."): pred = classifier.predict(arr)
            probs = pred[0].astype(float)

            labels = ["paper","rock","scissors"]  # sesuaikan urutan output model Anda
            if len(probs) != len(labels):
                labels = [f"class_{i}" for i in range(len(probs))]

            top_idx = int(np.argmax(probs))
            top_name = labels[top_idx].capitalize()
            top_prob = float(probs[top_idx])

            st.markdown(f"### Prediksi Utama: **{top_name}**")
            st.markdown(f"Skor keyakinan: **{top_prob:.4f}**")

            # Progress bar per kelas
            for name, p in zip(labels, probs):
                st.markdown(
                    f"<div class='prog-wrap'><span class='lbl'>{name.capitalize()}</span>"
                    f"<div class='prog'><span style='width:{p*100:.2f}%'></span></div>"
                    f"<span class='val'>{p*100:.1f}%</span></div>",
                    unsafe_allow_html=True
                )

            df = pd.DataFrame({"Kelas": [n.capitalize() for n in labels],
                               "Probabilitas (%)": (probs*100).round(2)})
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB: PROFIL DEVELOPER (pertanyaan saja)
# -----------------------------
with tab_profile:
    st.markdown("<div class='card'><div class='card-title'>Profil Developer — Mohon jawab di chat</div>", unsafe_allow_html=True)
    st.markdown("""
• **Nama yang ditampilkan**  
• **Peran/role utama**  
• **Tagline singkat** (1–2 kalimat)  
• **Skill inti (5–8)**  
• **Proyek unggulan (≤3)**  
• **Kontak & tautan** (email, GitHub, LinkedIn/Portofolio)  
• **Riwayat pendidikan** (opsional, timeline)
""")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB: PENJELASAN MODEL (per-box + dropdown)
# -----------------------------
with tab_docs:
    st.markdown("<div class='card'><div class='card-title'>Penjelasan Model</div>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Pilih model yang ingin dijelaskan",
        ["CNN", "YOLOv8"], index=0
    )

    def metric_bar(label:str, value:float):
        pct = max(0.0, min(1.0, float(value))) * 100
        st.markdown(
            f"<div class='prog-wrap'><span class='lbl'>{label}</span>"
            f"<div class='prog'><span style='width:{pct:.2f}%'></span></div>"
            f"<span class='val'>{pct:.1f}%</span></div>",
            unsafe_allow_html=True
        )

    # --- Box 1: Dataset (full width)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dataset")
    if model_choice == "CNN":
        st.markdown("""
**Sumber & Kelas** — **Rock–Paper–Scissors (RPS) — Dicoding** dengan tiga kelas: *paper* (712), *rock* (726), *scissors* (750); total **2.188** gambar.

**Pembagian** — **70%** latih, **20%** validasi, **10%** uji.

**Prapemrosesan**
- **Resize** ke **224×224** (RGB) dan **normalisasi** 0–1.
- **Augmentasi** (hanya latih): rotasi ≤10°, zoom ≤10%, horizontal flip — menambah variasi dan mengurangi overfitting.
""")
    else:
        st.markdown("""
**Sumber & Kelas** — **Rock–Paper–Scissors (RPS) — Dicoding**, dianotasi dengan **Roboflow** untuk deteksi.

**Pembagian & Ukuran** — Semua gambar diubah ke **640×640**; split **80%** latih, **10%** validasi, **10%** uji.

**Kesiapan Deteksi** — Bounding box tertata untuk tiga kelas (paper/rock/scissors), kompatibel dengan pipeline **YOLOv8** (anchor-free).
""")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Box 2 & 3: Arsitektur & Evaluasi (two columns)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Arsitektur")
        if model_choice == "CNN":
            st.markdown("""
**Rangkaian Layer**  
`[Conv2D(32, 3×3, ReLU) → MaxPool(2×2)]`  
`[Conv2D(64, 3×3, ReLU) → MaxPool(2×2)]`  
`[Conv2D(128,3×3, ReLU) → MaxPool(2×2)]`  
`Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(3, Softmax)`

**Callback**
- **EarlyStopping** (monitor *val_loss*) untuk mencegah overfitting.  
- **ModelCheckpoint** untuk menyimpan bobot terbaik.
""")
        else:
            st.markdown("""
**YOLOv8n (anchor-free)**  
- **Backbone** — ekstraksi fitur (SiLU, C2f, SPPF).  
- **Neck** — **FPN/PAN** untuk agregasi multi-skala.  
- **Head** — prediksi kelas + box pada stride **8/16/32** (tanpa anchor statis) → efisien dan akurat.
""")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Evaluasi")
        if model_choice == "CNN":
            st.markdown("Ringkasan metrik validasi:")
            metric_bar("Accuracy", 0.94)
            metric_bar("Precision (macro)", 0.94)
            metric_bar("Recall (macro)", 0.94)
            metric_bar("F1-score (macro)", 0.94)
            st.markdown("Model stabil di seluruh kelas berdasarkan confusion matrix dan classification report.")
        else:
            st.markdown("Metrik validasi & kecepatan:")
            metric_bar("Precision", 0.996)
            metric_bar("Recall", 1.00)
            metric_bar("mAP@50", 0.995)
            metric_bar("mAP@50–95", 0.925)
            st.markdown("Rata-rata latensi inferensi sekitar **17 ms/gambar** (pre-process + inferensi + post-process) — ideal untuk real-time.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Box 4: Kesimpulan (full width)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Kesimpulan")
    if model_choice == "CNN":
        st.markdown("""
Arsitektur CNN ringkas (tiga blok konvolusi + **Dropout 0.5**) dengan **EarlyStopping**/**ModelCheckpoint** mencapai akurasi **≈94%** dan generalisasi baik pada tiga kelas **Rock–Paper–Scissors**. Cocok sebagai pengklasifikasi akhir (misal setelah ROI dari detektor).
""")
    else:
        st.markdown("""
**YOLOv8n** memberikan presisi sangat tinggi dengan latensi rendah pada **Rock–Paper–Scissors**. Kombinasi **FPN/PAN** dan head **anchor-free** menjadikannya pilihan kuat untuk **deteksi real-time** maupun batch.
""")
    st.markdown("</div>", unsafe_allow_html=True)
