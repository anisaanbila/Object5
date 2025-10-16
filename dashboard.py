# app.py — Vision Dashboard (RPS) — Gradient + Poppins + Clear Docs
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Rock–Paper–Scissors (RPS) Vision Dashboard", page_icon="🧠", layout="wide")

# -----------------------------
# THEME (3-stop gradient + Poppins)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root{
  --bg1:#010030;        /* top */
  --bg2:#160078;        /* middle */
  --bg3:#7226FF;        /* bottom */
  --panel:#12122A;
  --panel-2:#1A1A34;
  --text:#E9E9F6;
  --muted:#A3A6C2;
}

* { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
h1 { font-weight:800; }
h2, h3 { font-weight:700; }
p, li, div { font-weight:400; }

header[data-testid="stHeader"]{ display:none; }
.block-container{ padding-top:3.6rem!important; padding-bottom:2rem; max-width:1300px; }

html, body, [data-testid="stAppViewContainer"]{
  background: linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%) fixed;
  color: var(--text);
}
a{ color:#C8CEFF!important; }

.card{
  background:
    linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0)) padding-box,
    linear-gradient(90deg, rgba(114,38,255,.35), rgba(1,0,48,.35)) border-box;
  border:1px solid transparent; border-radius:16px; padding:18px 20px;
  box-shadow: 0 10px 28px rgba(0,0,0,.35);
}
.card.compact{ padding:14px 16px; }
.card-title{ font-weight:700; font-size:1rem; color:#D7DAFF; margin-bottom:.5rem; }
.caption{ color: var(--muted); font-size:.86rem; }
hr{ border-color:#2b2c44; }

.pill{
  background: linear-gradient(90deg, rgba(1,0,48,.25), rgba(114,38,255,.25));
  border:1px solid rgba(150,150,220,.35); color:#D7DAFF;
  padding:.35rem .7rem; border-radius:999px; font-weight:600; font-size:.82rem;
}

.kpi{ display:flex; gap:.6rem; align-items:center; padding:.6rem .9rem;
     background:#101026; border:1px solid #2b2c44; border-radius:12px; }
.kpi .big{ font-weight:800; font-size:1.1rem; }

/* Tabs underline */
.stTabs [role="tablist"]{ gap:1rem; }
.stTabs [role="tab"]{ border-bottom:2px solid transparent; }
.stTabs [role="tab"][aria-selected="true"]{
  border-bottom:2px solid; border-image: linear-gradient(90deg,#010030,#7226FF) 1;
}

/* File uploader text color */
[data-testid="stFileUploader"] section div{ color:#A3A6C2!important; }

/* Progress bars (classification & evaluation) */
.prog{ width:100%; height:10px; border-radius:999px; background:#23234a; overflow:hidden; }
.prog > span{ display:block; height:100%;
  background:linear-gradient(90deg,#160078,#7226FF); width:0%;
}
.prog-wrap{ display:flex; align-items:center; gap:.6rem; margin:.35rem 0; }
.prog-wrap .lbl{ min-width:110px; font-weight:600; font-size:.9rem; color:#D7DAFF; }
.prog-wrap .val{ width:56px; text-align:right; color:#D7DAFF; font-variant-numeric: tabular-nums; }

/* Selectbox tidy */
[data-baseweb="select"] div{ font-weight:600; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Anisa Nabila_Laporan 4.pt")                    # YOLOv8 detector
    clf  = tf.keras.models.load_model("model/Anisa Nabila_Laporan 2.h5")  # CNN classifier
    return yolo, clf

yolo_model, classifier = load_models()

# -----------------------------
# HEADER (RPS-focused)
# -----------------------------
c1, c2 = st.columns([1.9,1.1])
with c1:
    st.markdown(
        "<div class='card'>"
        "<div class='card-title'>RPS Vision Dashboard</div>"
        "<h1 style='margin:0 0 .3rem 0;'>Rock–Paper–Scissors (RPS) Detection & Classification</h1>"
        "<div class='caption'>Antarmuka untuk deteksi objek (YOLOv8) dan klasifikasi gambar (CNN) pada gestur tangan RPS.</div>"
        "</div>", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class='card compact'>
      <div class='card-title'>Model Status</div>
      <div class='kpi'>✅<span class='big'>Ready</span><span class='caption'>YOLOv8 & CNN aktif</span></div>
      <div style="margin-top:.6rem"><span class='pill'>Gradient UI · Poppins</span></div>
    </div>""", unsafe_allow_html=True)

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
                res = yolo_model(img); plotted = res[0].plot()
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, use_container_width=True, caption="Deteksi (bounding boxes)")
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
# TAB: KLASIFIKASI  (rapi + progress)
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
            # --- Prediksi
            img_resized = img2.resize((224,224))
            arr = image.img_to_array(img_resized); arr = np.expand_dims(arr,0)/255.0
            with st.spinner("Mengklasifikasikan..."): pred = classifier.predict(arr)
            probs = pred[0].astype(float)
            labels = ["paper","rock","scissors"]     # sesuaikan urutan output model Anda
            if len(probs) != len(labels):
                labels = [f"class_{i}" for i in range(len(probs))]
            top_idx = int(np.argmax(probs))
            top_name = labels[top_idx]
            top_prob = float(probs[top_idx])

            st.markdown(f"### Prediksi Utama: **{top_name.capitalize()}** — Rock–Paper–Scissors (RPS)")
            st.markdown(f"Probabilitas: **{top_prob:.4f}**")

            # --- Progress bar per kelas
            for name, p in zip(labels, probs):
                st.markdown(
                    f"<div class='prog-wrap'><span class='lbl'>{name.capitalize()}</span>"
                    f"<div class='prog'><span style='width:{p*100:.2f}%'></span></div>"
                    f"<span class='val'>{p*100:.1f}%</span></div>",
                    unsafe_allow_html=True
                )

            # --- Tabel ringkas
            df = pd.DataFrame({"Kelas": labels, "Probabilitas (%)": (probs*100).round(2)})
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("<div class='caption'>Pastikan urutan <code>labels</code> sesuai output model Anda.</div>", unsafe_allow_html=True)
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
• **Preferensi warna/aksen tambahan** (bila ada)
""")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB: PENJELASAN MODEL (per-box + SELECTBOX)
# -----------------------------
with tab_docs:
    st.markdown("<div class='card'><div class='card-title'>Penjelasan Model</div>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Pilih model yang ingin dijelaskan",
        ["CNN", "YOLOv8"],
        index=0
    )

    def metric_bar(label:str, value:float):
        """Render satu bar evaluasi, value 0..1"""
        pct = max(0.0, min(1.0, float(value))) * 100
        st.markdown(
            f"<div class='prog-wrap'><span class='lbl'>{label}</span>"
            f"<div class='prog'><span style='width:{pct:.2f}%'></span></div>"
            f"<span class='val'>{pct:.1f}%</span></div>",
            unsafe_allow_html=True
        )

    def box_dataset(kind:str):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset")
        if kind=="CNN":
            st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (RPS) dari Dicoding** dengan tiga kelas: *paper* (712), *rock* (726), *scissors* (750) — total **2.188** gambar.

**Pembagian Data.** **70%** latih, **20%** validasi, **10%** uji.

**Prapemrosesan.**
- **Resize** ke **224×224** piksel (RGB).
- **Normalisasi** 0–1 untuk mempercepat konvergensi.
- **Augmentasi** (latih saja): rotasi ≤10°, zoom ≤10%, horizontal flip → menambah variasi dan mengurangi overfitting.
            """)
        else:
            st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (RPS) dari Dicoding**, dianotasi dengan **Roboflow** untuk tugas deteksi.

**Pembagian & Ukuran.** Gambar diubah ke **640×640**; split **80%** latih, **10%** validasi, **10%** uji.

**Kesiapan Deteksi.** Bounding box tertata untuk tiga kelas (*paper*, *rock*, *scissors*), kompatibel dengan pipeline **YOLOv8** (anchor-free).
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    def box_arch_eval(kind:str):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Arsitektur")
            if kind=="CNN":
                st.markdown("""
**Rangkaian Layer**  
`[Conv2D(32, 3×3, ReLU) → MaxPool(2×2)]`  
`[Conv2D(64, 3×3, ReLU) → MaxPool(2×2)]`  
`[Conv2D(128,3×3, ReLU) → MaxPool(2×2)]`  
`Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(3, Softmax)`

**Callback**  
- **EarlyStopping** (monitor *val_loss*) untuk menghentikan training saat tidak membaik.  
- **ModelCheckpoint** menyimpan bobot terbaik.
                """)
            else:
                st.markdown("""
**YOLOv8n (anchor-free)**  
- **Backbone**: ekstraksi fitur (SiLU, C2f, SPPF).  
- **Neck**: **FPN/PAN** untuk penggabungan fitur multi-skala.  
- **Head**: prediksi kelas + box pada stride **8/16/32**, tanpa anchor statis → efisien dan akurat.
                """)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Evaluasi")
            if kind=="CNN":
                st.markdown("Ringkasan metrik validasi (sekitar):")
                metric_bar("Accuracy", 0.94)
                metric_bar("Precision (macro)", 0.94)
                metric_bar("Recall (macro)", 0.94)
                metric_bar("F1-score (macro)", 0.94)
                st.markdown("""
Model menunjukkan performa **stabil di seluruh kelas** (lihat confusion matrix & classification report di laporan), tanpa bias dominan.
                """)
            else:
                st.markdown("Metrik validasi & kecepatan:")
                metric_bar("Precision", 0.996)
                metric_bar("Recall", 1.00)
                metric_bar("mAP@50", 0.995)
                metric_bar("mAP@50–95", 0.925)
                st.markdown("""
Rata-rata latensi inferensi ~ **17 ms/gambar** (pre-process + inferensi + post-process), ideal untuk aplikasi **real-time**.
                """)
            st.markdown("</div>", unsafe_allow_html=True)

    def box_conclusion(kind:str):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Kesimpulan")
        if kind=="CNN":
            st.markdown("""
Arsitektur CNN yang ringkas (tiga blok konvolusi + **Dropout 0.5**) dengan **EarlyStopping** dan **ModelCheckpoint** menghasilkan akurasi **≈94%** serta generalisasi yang baik untuk 3 kelas **Rock–Paper–Scissors (RPS)**. Cocok sebagai pengklasifikasi akhir—misalnya setelah ROI dipotong dari detektor.
            """)
        else:
            st.markdown("""
**YOLOv8n** mencapai presisi tinggi dan latensi rendah pada **Rock–Paper–Scissors (RPS)**. Kombinasi **FPN/PAN** dan head **anchor-free** membuatnya unggul untuk **deteksi real-time** maupun batch processing.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    # RENDER per-box
    box_dataset(model_choice)
    box_arch_eval(model_choice)
    box_conclusion(model_choice)

    st.markdown("</div>", unsafe_allow_html=True)
