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
}
* { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
h1{ font-weight:800; } h2,h3,h4{ font-weight:800; } p,li,div{ font-weight:400; }

header[data-testid="stHeader"]{ display:none; }
.block-container{ padding-top:3.6rem!important; padding-bottom:2rem; max-width:1300px; }

/* Futuristic gradient background + subtle grid */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 20% -10%, rgba(114,38,255,.25), transparent 65%),
    radial-gradient(900px 500px at 90% 10%, rgba(1,0,48,.35), transparent 60%),
    linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%) fixed;
  color: var(--text);
}

/* Cards with neon-glass effect */
.card{
  position:relative;
  background:
    linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0)) padding-box,
    linear-gradient(90deg, rgba(114,38,255,.35), rgba(1,0,48,.35)) border-box;
  border:1px solid transparent; border-radius:16px; padding:18px 20px;
  box-shadow: 0 14px 36px rgba(0,0,0,.40);
  overflow:hidden;
}
.card:before{
  content:""; position:absolute; inset:-2px; border-radius:18px;
  background: radial-gradient(600px 120px at 10% -10%, rgba(114,38,255,.16), transparent 60%);
  pointer-events:none; filter: blur(6px);
}
.card.compact{ padding:14px 16px; }
.card-title{ font-weight:700; font-size:1rem; color:#D7DAFF; margin-bottom:.6rem; letter-spacing:.15px; }
.caption{ color: var(--muted); font-size:.9rem; }
hr{ border-color:#2b2c44; }

/* Pills */
.pill{
  background: linear-gradient(90deg, rgba(1,0,48,.28), rgba(114,38,255,.28));
  border:1px solid rgba(150,150,220,.38); color:#D7DAFF;
  padding:.35rem .7rem; border-radius:999px; font-weight:600; font-size:.82rem;
}

/* KPI */
.kpi{ display:flex; gap:.6rem; align-items:center; padding:.6rem .9rem;
     background:#101026; border:1px solid #2b2c44; border-radius:12px; }
.kpi .big{ font-weight:800; font-size:1.1rem; }

/* Tabs — title white */
.stTabs [role="tablist"]{ gap:1rem; }
.stTabs [role="tab"]{
  color:#FFFFFF !important; font-weight:700; letter-spacing:.2px;
  border-bottom:2px solid transparent;
}
.stTabs [role="tab"][aria-selected="true"]{
  border-bottom:2px solid; border-image: linear-gradient(90deg,#010030,#7226FF) 1;
}

/* File uploader text color */
[data-testid="stFileUploader"] section div{ color:#A3A6C2!important; }

/* Progress bars (classification & evaluation) */
.prog{ width:100%; height:10px; border-radius:999px; background:#23234a; overflow:hidden; }
.prog > span{
  display:block; height:100%; width:0%;
  background:linear-gradient(90deg,#160078,#7226FF);
  animation: loadWidth 1s ease-out forwards;
}
@keyframes loadWidth { from{ width:0% } to{ width:var(--w,0%) } }
.prog-wrap{ display:flex; align-items:center; gap:.6rem; margin:.42rem 0; }
.prog-wrap .lbl{ min-width:130px; font-weight:700; font-size:.92rem; color:#FFFFFF; }
.prog-wrap .val{ width:64px; text-align:right; color:#FFFFFF; font-weight:600; font-variant-numeric: tabular-nums; }

/* Subheader accent (bold) */
.subttl{
  font-weight:800; font-size:1.05rem; margin:.2rem 0 .6rem 0;
  background: linear-gradient(90deg,#FFFFFF 0%, #D7DAFF 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.subttl-underline{
  height:2px; width:100%; margin:.25rem 0 .8rem 0;
  background: linear-gradient(90deg, #010030, #7226FF);
  border-radius:999px;
}

/* Chips (for detections) */
.chips{ display:flex; gap:.4rem; flex-wrap:wrap; }
.chip{
  padding:.25rem .55rem; border-radius:999px;
  background:rgba(114,38,255,.18); border:1px solid rgba(114,38,255,.35);
  font-weight:600; font-size:.82rem; color:#E9E9F6;
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
# HEADER
# =========================
c1, c2 = st.columns([1.8,1.2])
with c1:
    st.markdown(
        "<div class='card'>"
        "<div class='card-title'>RPS Vision Dashboard</div>"
        "<h1 style='margin:0 0 .3rem 0;'>Rock–Paper–Scissors (RPS) — Detection & Classification</h1>"
        "<div class='caption'>Antarmuka mewah & futuristik untuk deteksi objek (YOLOv8) dan klasifikasi gambar (CNN) gestur tangan RPS.</div>"
        "</div>", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class='card compact'>
      <div class='card-title'>Model Status</div>
      <div class='kpi'>✅<span class='big'>Ready</span><span class='caption'>YOLOv8 & CNN aktif</span></div>
      <div style="margin-top:.6rem"><span class='pill'>Gradient • Poppins</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab_det, tab_cls, tab_profile, tab_docs = st.tabs([
    "Deteksi Objek (YOLOv8)", "Klasifikasi Gambar (CNN)", "Profil Developer", "Penjelasan Model"
])

def uploader_card(key_label:str, title="Unggah Gambar"):
    st.markdown(f"<div class='card'><div class='card-title'>{title}</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return f

# =========================
# TAB: DETEKSI (rapi + ringkasan elegan)
# =========================
with tab_det:
    left, right = st.columns([1.04,1])
    with left:
        f = uploader_card("up_yolo", "Unggah Gambar • Deteksi (RPS)")
        # kontrol threshold agar user bisa eksplor
        conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.01)
        iou  = st.slider("IoU NMS", 0.1, 0.9, 0.45, 0.01)
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
                res = yolo_model.predict(img, conf=conf, iou=iou, verbose=False)
                plotted = res[0].plot()
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

            st.image(plotted, use_container_width=True, caption="Deteksi (bounding boxes)")

            # --- Ringkasan elegan
            names = res[0].names
            boxes = res[0].boxes
            if boxes is not None and len(boxes)>0:
                cls_ids = [int(c) for c in boxes.cls.tolist()]
                confs   = boxes.conf.tolist()
                # chips
                st.markdown("**Deteksi Terdapat:**")
                st.markdown("<div class='chips'>" + "".join([f"<span class='chip'>{names[c]}</span>" for c in cls_ids]) + "</div>", unsafe_allow_html=True)

                # tabel agregat
                counts = Counter(cls_ids)
                rows = []
                for cid, cnt in counts.items():
                    avg_conf = np.mean([c for i,c in enumerate(confs) if cls_ids[i]==cid])
                    rows.append([names[cid], int(cnt), float(avg_conf)])
                df = pd.DataFrame(rows, columns=["Kelas", "Jumlah", "Rata-rata Confidence"])
                df["Rata-rata Confidence (%)"] = (df["Rata-rata Confidence"]*100).round(2)
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df[["Kelas","Jumlah","Rata-rata Confidence (%)"]], hide_index=True, use_container_width=True)
            else:
                st.info("Tidak ada objek terdeteksi pada gambar ini.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB: KLASIFIKASI (rapi + progress)
# =========================
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
            # --- Prediksi
            img_resized = img2.resize((224,224))
            arr = image.img_to_array(img_resized); arr = np.expand_dims(arr,0)/255.0
            with st.spinner("Mengklasifikasikan..."): pred = classifier.predict(arr)
            probs = pred[0].astype(float)
            labels = ["paper","rock","scissors"]
            if len(probs) != len(labels):
                labels = [f"class_{i}" for i in range(len(probs))]
            top_idx = int(np.argmax(probs)); top_name = labels[top_idx]; top_prob = float(probs[top_idx])

            st.markdown(f"### Prediksi Utama: **{top_name.capitalize()}** — Rock–Paper–Scissors (RPS)")
            st.markdown(f"Probabilitas: **{top_prob:.4f}**")

            for name, p in zip(labels, probs):
                pct = f"{p*100:.2f}%"
                st.markdown(
                    f"<div class='prog-wrap'><span class='lbl'>{name.capitalize()}</span>"
                    f"<div class='prog'><span style='--w:{p*100:.2f}%;'></span></div>"
                    f"<span class='val'>{pct}</span></div>",
                    unsafe_allow_html=True
                )

            df = pd.DataFrame({"Kelas": labels, "Probabilitas (%)": (probs*100).round(2)})
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("<div class='caption'>Pastikan urutan <code>labels</code> sesuai output model Anda.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB: PROFIL DEVELOPER (prompt untuk kamu)
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

# =========================
# TAB: PENJELASAN MODEL (lengkap, per-box, dropdown)
# =========================
with tab_docs:
    st.markdown("<div class='card'><div class='card-title'>Penjelasan Model</div>", unsafe_allow_html=True)
    model_choice = st.selectbox("Pilih model yang ingin dijelaskan", ["CNN", "YOLOv8"], index=0)

    def metric_bar(label:str, value:float):
        pct = max(0.0, min(1.0, float(value))) * 100
        st.markdown(
            f"<div class='prog-wrap'><span class='lbl'>{label}</span>"
            f"<div class='prog'><span style='--w:{pct:.2f}%;'></span></div>"
            f"<span class='val'>{pct:.1f}%</span></div>",
            unsafe_allow_html=True
        )

    # ---- Box 1: Dataset
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='subttl'>Dataset</div><div class='subttl-underline'></div>", unsafe_allow_html=True)
    if model_choice == "CNN":
        st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (RPS) – Dicoding**. Tiga kelas seimbang: *paper* (712), *rock* (726), *scissors* (750); total **2.188** gambar.

**Split & Prapemrosesan.**
- Split: **70%** latih, **20%** validasi, **10%** uji.  
- **Resize** ke **224×224** (RGB) dan **normalisasi** 0–1.  
- **Augmentasi** pada data latih: rotasi ≤10°, zoom ≤10%, horizontal flip — untuk memperluas distribusi dan mengurangi overfitting.
        """)
    else:
        st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (RPS) – Dicoding** dengan anotasi **bounding box** via **Roboflow**.

**Split & Ukuran.**
- Semua citra diubah ke **640×640**.  
- Split: **80%** latih, **10%** validasi, **10%** uji.  
- Struktur anotasi kompatibel dengan format **YOLOv8** (anchor-free).
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Box 2 & 3: Arsitektur + Evaluasi
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='subttl'>Arsitektur</div><div class='subttl-underline'></div>", unsafe_allow_html=True)
        if model_choice == "CNN":
            st.markdown("""
**Rangkaian Layer**
- `Conv2D(32, 3×3, ReLU)` → `MaxPool(2×2)`  
- `Conv2D(64, 3×3, ReLU)` → `MaxPool(2×2)`  
- `Conv2D(128, 3×3, ReLU)` → `MaxPool(2×2)`  
- `Flatten` → `Dense(128, ReLU)` → `Dropout(0.5)` → `Dense(3, Softmax)`

**Pelatihan**
- Optimizer **Adam**, loss **categorical_crossentropy**.  
- **EarlyStopping** (monitor *val_loss*) dan **ModelCheckpoint** (bobot terbaik).  
- Epoch menyesuaikan; berhenti dini saat konvergensi tercapai.
            """)
        else:
            st.markdown("""
**YOLOv8n (anchor-free)**
- **Backbone**: ekstraksi fitur (aktivasi **SiLU**, blok **C2f**, modul **SPPF**).  
- **Neck**: **FPN/PAN** untuk penggabungan fitur multi-skala.  
- **Head**: prediksi kelas & box pada stride **8/16/32** tanpa anchor tetap → efisien, cepat, presisi tinggi.

**Inferensi (UI)**
- Batas default: **confidence** 0.25 & **IoU NMS** 0.45 (dapat diubah pada tab Deteksi).
            """)
        st.markdown("</div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='subttl'>Evaluasi</div><div class='subttl-underline'></div>", unsafe_allow_html=True)
        if model_choice == "CNN":
            st.markdown("Metrik validasi (perkiraan dari hasil terbaik):")
            metric_bar("Accuracy", 0.94)
            metric_bar("Precision (macro)", 0.94)
            metric_bar("Recall (macro)", 0.94)
            metric_bar("F1-score (macro)", 0.94)
            st.markdown("""
**Interpretasi.** Performa konsisten di ketiga kelas; confusion matrix & classification report menunjukkan tidak ada kelas yang dominan salah.
            """)
        else:
            st.markdown("Metrik validasi & performa:")
            metric_bar("Precision", 0.996)
            metric_bar("Recall", 1.00)
            metric_bar("mAP@50", 0.995)
            metric_bar("mAP@50–95", 0.925)
            st.markdown("""
**Kecepatan.** Rata-rata latensi ~ **17 ms/gambar** (preprocess + inferensi + postprocess) → layak untuk **real-time**.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Box 4: Kesimpulan
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='subttl'>Kesimpulan</div><div class='subttl-underline'></div>", unsafe_allow_html=True)
    if model_choice == "CNN":
        st.markdown("""
Arsitektur CNN yang ringkas (tiga blok konvolusi + **Dropout 0.5**) dengan **EarlyStopping** dan **ModelCheckpoint**
memberikan akurasi sekitar **94%** pada **Rock–Paper–Scissors (RPS)**. Model ringan dan cocok menjadi **pengklasifikasi akhir**,
misalnya setelah ROI dari detektor atau untuk aplikasi edukasi/latih.
        """)
    else:
        st.markdown("""
**YOLOv8n** menghasilkan deteksi yang **sangat akurat** (Precision & mAP tinggi) sekaligus **cepat** (≈17 ms/gambar) pada **Rock–Paper–Scissors (RPS)**.
Arsitektur **FPN/PAN + head anchor-free** memudahkan generalisasi lintas ukuran objek tangan, sehingga ideal untuk **deteksi real-time** pada kamera/stream.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
