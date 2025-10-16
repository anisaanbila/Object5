# app.py — Vision Dashboard (RPS) — Gradient UI + Clear Model Docs
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
st.set_page_config(page_title="Vision Dashboard — RPS", page_icon="🧠", layout="wide")

# -----------------------------
# THEME (3-stop gradient + tidy)
# -----------------------------
st.markdown("""
<style>
:root{
  --bg1:#010030;        /* top */
  --bg2:#160078;        /* middle */
  --bg3:#7226FF;        /* bottom */
  --panel:#12122A;
  --panel-2:#1A1A34;
  --text:#E9E9F6;
  --muted:#A3A6C2;
}
header[data-testid="stHeader"]{ display:none; }
.block-container{ padding-top:3.6rem!important; padding-bottom:2rem; max-width:1300px; }

html, body, [data-testid="stAppViewContainer"]{
  background: linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%) fixed;
  color: var(--text);
}
a{ color:#C8CEFF!important; }
h1,h2,h3,h4{ color: var(--text); }

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
.action{
  background: linear-gradient(90deg, #160078, #7226FF);
  color:#fff; font-weight:700; padding:.55rem 1rem; border-radius:12px;
  display:inline-flex; gap:.5rem; text-decoration:none;
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

/* Progress bars for probabilities */
.prog{ width:100%; height:10px; border-radius:999px; background:#23234a; overflow:hidden; }
.prog > span{ display:block; height:100%;
  background:linear-gradient(90deg,#160078,#7226FF); width:0%;
}
.prog-wrap{ display:flex; align-items:center; gap:.6rem; }
.prog-wrap .lbl{ min-width:90px; font-weight:600; font-size:.88rem; color:#D7DAFF; }
.prog-wrap .val{ width:46px; text-align:right; color:#D7DAFF; font-variant-numeric: tabular-nums; }
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
# HEADER
# -----------------------------
c1, c2, c3 = st.columns([1.6,1,1])
with c1:
    st.markdown(
        "<div class='card'><div class='card-title'>Dashboard</div>"
        "<h1 style='margin:0 0 .3rem 0;'>Welcome Back, Anisa!</h1>"
        "<div class='caption'>UI untuk <b>Rock–Paper–Scissors</b>: deteksi (YOLOv8) & klasifikasi (CNN)</div>"
        "</div>", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class='card compact'>
      <div class='card-title'>Model Status</div>
      <div class='kpi'>✅<span class='big'>Ready</span><span class='caption'>YOLO & Classifier</span></div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class='card compact' style='display:flex;justify-content:space-between;align-items:center;gap:.8rem'>
      <span class='pill'>RPS Vision</span>
      <a class='action' href='#' onclick='return false;'>＋ Create Session</a>
    </div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab_det, tab_cls, tab_profile, tab_docs = st.tabs([
    "Deteksi Objek (YOLO)", "Klasifikasi Gambar", "Profil Developer", "Penjelasan Model"
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
        f = uploader_card("up_yolo", "Unggah Gambar • Deteksi")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil Deteksi</div>", unsafe_allow_html=True)
        if not f:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan deteksi.</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Menjalankan YOLO..."):
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
        g = uploader_card("up_cls", "Unggah Gambar • Klasifikasi")
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
            labels = ["paper","rock","scissors"]     # sesuaikan dengan urutan output model Anda
            # safety jika jumlah kelas berbeda
            if len(probs) != len(labels):
                labels = [f"class_{i}" for i in range(len(probs))]
            # ringkasan utama
            top_idx = int(np.argmax(probs))
            top_name = labels[top_idx]
            top_prob = float(probs[top_idx])

            st.markdown(f"### Prediksi Utama: **{top_name.capitalize()}**")
            st.markdown(f"Probabilitas: **{top_prob:.4f}**")

            # --- Progress bar per kelas (rapi)
            for name, p in zip(labels, probs):
                st.markdown(
                    f"<div class='prog-wrap'><span class='lbl'>{name.capitalize()}</span>"
                    f"<div class='prog'><span style='width:{p*100:.2f}%'></span></div>"
                    f"<span class='val'>{p*100:.1f}%</span></div>",
                    unsafe_allow_html=True
                )

            # --- Tabel probabilitas
            df = pd.DataFrame({"Kelas": labels, "Probabilitas": [float(x) for x in probs]})
            df["Probabilitas (%)"] = (df["Probabilitas"]*100).round(2)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df[["Kelas","Probabilitas (%)"]], use_container_width=True, hide_index=True)

            st.markdown("<div class='caption'>Catatan: sesuaikan urutan <code>labels</code> dengan model Anda.</div>", unsafe_allow_html=True)
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

    def box_dataset(kind:str):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset")
        if kind=="CNN":
            st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (Dicoding)** dengan tiga kelas: *paper* (712), *rock* (726), dan *scissors* (750) — total **2.188** gambar.

**Pembagian Data.** Data dibagi **70% latih / 20% validasi / 10% uji**.

**Prapemrosesan.**
- **Resize** ke **224×224** piksel (RGB).
- **Normalisasi** nilai piksel ke rentang **0–1**.
- **Augmentasi** hanya untuk data latih: rotasi ≤ 10°, zoom ≤ 10%, dan horizontal flip. 
Tujuannya memperkaya variasi sehingga model lebih robust dan tidak overfitting.
            """)
        else:
            st.markdown("""
**Sumber & Kelas.** Dataset **Rock–Paper–Scissors (Dicoding)** yang telah **dilabeli di Roboflow** untuk tugas deteksi.

**Pembagian Data & Ukuran.** Seluruh gambar diubah ke **640×640** piksel dengan split **80% latih / 10% validasi / 10% uji**.

**Kesiapan Deteksi.** Anotasi bounding box tersusun rapi per kelas (*paper/rock/scissors*) sehingga kompatibel dengan pipeline **YOLOv8** (anchor-free).
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    def box_arch_eval(kind:str):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Arsitektur")
            if kind=="CNN":
                st.markdown("""
**Rangkaian Layer.**  
`[Conv2D(32, 3×3, ReLU) → MaxPool(2×2)] × 1`  
`[Conv2D(64, 3×3, ReLU) → MaxPool(2×2)] × 1`  
`[Conv2D(128,3×3, ReLU) → MaxPool(2×2)] × 1`  
`Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(3, Softmax)`

**Callback Penting.**  
- **EarlyStopping** memutus training saat *val_loss* tidak membaik (mencegah overfitting).  
- **ModelCheckpoint** menyimpan bobot terbaik sepanjang training.
                """)
            else:
                st.markdown("""
**Struktur YOLOv8n.**
- **Backbone** mengekstraksi fitur (aktivasi **SiLU**, blok **C2f**, dan **SPPF** untuk receptive field luas).
- **Neck** menggabungkan fitur lintas skala (**FPN** + **PAN**) agar objek kecil/besar tetap terwakili.
- **Head (anchor-free)** langsung memprediksi kelas + box di **3 skala** (stride **8/16/32**), tanpa anchor tetap sehingga efisien.
                """)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Evaluasi")
            if kind=="CNN":
                st.markdown("""
**Hasil Utama.**
- Akurasi validasi sekitar **94%**.
- **Precision/Recall/F1** merata di ketiga kelas (sesuai laporan CM & classification report).  
**Makna.** Model mampu membedakan *paper*, *rock*, dan *scissors* dengan konsisten—tidak bias pada kelas tertentu dan tetap generalizable pada data uji.
                """)
            else:
                st.markdown("""
**Setelan & Hasil.**
- **Training 100 epoch**; bobot terbaik diseleksi otomatis.
- Metrik validasi: **Precision ≈ 0.996**, **Recall 1.000**, **mAP@50 0.995**, **mAP@50–95 0.925**.
- Rata-rata latensi per gambar ~ **17 ms** (pre-process + inferensi + post-process).

**Makna.** Deteksi sangat akurat sekaligus cepat — ideal untuk skenario **real-time** dengan objek tangan RPS.
                """)
            st.markdown("</div>", unsafe_allow_html=True)

    def box_conclusion(kind:str):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Kesimpulan")
        if kind=="CNN":
            st.markdown("""
Arsitektur CNN sederhana dengan tiga blok konvolusi + **Dropout 0.5** dan **callback** memberikan akurasi tinggi (**≈94%**) dan generalisasi yang baik untuk 3 kelas RPS. Pipeline ini ringan untuk inferensi dan cocok sebagai **pengklasifikasi akhir** pada sistem sederhana atau setelah pemotongan ROI dari detektor.
            """)
        else:
            st.markdown("""
**YOLOv8n** terbukti **akurat dan efisien** pada dataset RPS. Kombinasi **FPN/PAN** dan **head anchor-free** menghasilkan metrik presisi sangat tinggi dengan latensi rendah, sehingga **layak untuk produksi** pada aplikasi **deteksi real-time** (kamera/stream) maupun batch processing.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    # RENDER per-box (layout: 1 box → 2 box sejajar → 1 box)
    box_dataset(model_choice)
    box_arch_eval(model_choice)
    box_conclusion(model_choice)

    st.markdown("</div>", unsafe_allow_html=True)
