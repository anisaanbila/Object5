# app.py — Vision Dashboard (RPS) — Dark + Purple
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Vision Dashboard — RPS", page_icon="🧠", layout="wide")

# ---------- THEME & STYLE ----------
st.markdown("""
<style>
:root{ --bg:#0F0F14; --panel:#151520; --panel-2:#1B1B2A;
       --text:#E7E7F0; --muted:#9EA0B3; --accent:#7C3AED; --accent-2:#9F67FF; }
header[data-testid="stHeader"]{ display:none; }
.block-container{ padding-top:3.5rem!important; padding-bottom:2rem; max-width:1300px; }
html,body,[data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--text); }
a{ color:var(--accent-2)!important; } h1,h2,h3,h4{ color:var(--text); }
.card{ background:linear-gradient(180deg,var(--panel),var(--panel-2)); border:1px solid #23233a;
       border-radius:16px; padding:18px 20px; box-shadow:0 10px 24px rgba(0,0,0,.25); }
.card.compact{ padding:14px 16px; }
.card-title{ font-weight:700; font-size:1rem; margin-bottom:.4rem; color:#cfd1e6; }
.caption{ color:var(--muted); font-size:.86rem; }
.kpi{ display:flex; gap:.6rem; align-items:center; padding:.6rem .9rem; background:#10101a;
      border:1px solid #23233a; border-radius:12px; }
.kpi .big{ font-weight:800; font-size:1.1rem; }
.pill{ background:rgba(124,58,237,.14); border:1px solid rgba(124,58,237,.45);
      color:var(--accent-2); padding:.35rem .7rem; border-radius:999px; font-size:.82rem; font-weight:600; }
.action{ background:linear-gradient(90deg,var(--accent),var(--accent-2)); color:#fff; font-weight:700;
         padding:.55rem 1rem; border-radius:12px; display:inline-flex; gap:.5rem; text-decoration:none; }
.icon-row{ display:flex; gap:.6rem; justify-content:flex-end; }
.icon-bubble{ width:36px;height:36px;border-radius:50%; display:flex; align-items:center; justify-content:center;
              background:#1c1c2a;border:1px solid #2b2b44; }
hr{ border-color:#23233a; }
.stTabs [role="tablist"]{ gap:1rem; }
.stTabs [role="tab"]{ border-bottom:2px solid transparent; }
.stTabs [role="tab"][aria-selected="true"]{ border-bottom:2px solid var(--accent-2); }
[data-testid="stFileUploader"] section div{ color:#9EA0B3!important; }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODELS ----------
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Anisa Nabila_Laporan 4.pt")                 # YOLOv8 detector
    clf  = tf.keras.models.load_model("model/Anisa Nabila_Laporan 2.h5")  # CNN classifier
    return yolo, clf
yolo_model, classifier = load_models()

# ---------- HEADER ----------
c1, c2, c3 = st.columns([1.6,1,1])
with c1:
    st.markdown(
        "<div class='card'><div class='card-title'>Dashboard</div>"
        "<h1 style='margin:0 0 .3rem 0;'>Welcome Back, Anisa! 👋</h1>"
        "<div class='caption'>UI untuk <b>Rock–Paper–Scissors</b>: deteksi (YOLOv8) & klasifikasi (CNN).</div>"
        "</div>", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class='card compact'>
      <div class='card-title'>Model Status</div>
      <div class='kpi'>✅<span class='big'>Ready</span><span class='caption'>YOLO & Classifier</span></div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class='card compact'>
      <div class='icon-row'>
        <div class='icon-bubble'>🔎</div><div class='icon-bubble'>⚙️</div>
        <div class='icon-bubble'>📷</div><div class='icon-bubble'>💾</div>
      </div>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-top:.75rem;'>
        <span class='pill'>RPS Vision</span><a class='action' href='#' onclick='return false;'>＋ Create Session</a>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- TABS ----------
tab_det, tab_cls, tab_profile, tab_docs = st.tabs([
    "🔍 Deteksi Objek (YOLO)", "🧪 Klasifikasi Gambar", "👤 Profil Developer", "📘 Penjelasan Model"
])

def uploader_card(key_label:str, title="Unggah Gambar"):
    st.markdown(f"<div class='card'><div class='card-title'>{title}</div>", unsafe_allow_html=True)
    file = st.file_uploader(" ", type=["png","jpg","jpeg"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return file

# --- TAB: DETEKSI ---
with tab_det:
    left, right = st.columns([1.04,1])
    with left:
        f = uploader_card("up_yolo", "Unggah Gambar • 🔍 Deteksi")
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
            if boxes is not None and len(boxes) > 0:
                st.markdown("<hr>", unsafe_allow_html=True); st.markdown("**Ringkasan:**")
                for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    st.write(f"• {names[int(cls_id)]} — conf: {conf:.2f}")
            else:
                st.info("Tidak ada objek terdeteksi.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB: KLASIFIKASI ---
with tab_cls:
    left, right = st.columns([1.04,1])
    with left:
        g = uploader_card("up_cls", "Unggah Gambar • 🧪 Klasifikasi")
        if g:
            img2 = Image.open(g).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
        if not g:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            img_resized = img2.resize((224,224))
            arr = image.img_to_array(img_resized); arr = np.expand_dims(arr,0)/255.0
            with st.spinner("Mengklasifikasikan..."): pred = classifier.predict(arr)
            prob = float(np.max(pred)); idx = int(np.argmax(pred))
            labels = ["paper","rock","scissors"]  # sesuaikan dengan urutan output model kamu
            name = labels[idx] if idx < len(labels) else str(idx)
            st.markdown(f"**Label Prediksi:** `📌 {name}`")
            st.markdown(f"**Probabilitas:** `{prob:.4f}`")
            st.markdown("<div class='caption'>Catatan: sesuaikan daftar <code>labels</code> dengan kelas model Anda.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB: PROFIL DEVELOPER (berisi pertanyaan, bukan form) ---
with tab_profile:
    st.markdown("<div class='card'><div class='card-title'>👤 Profil Developer — Mohon Jawab di Chat</div>", unsafe_allow_html=True)
    st.markdown("""
- **Nama yang ditampilkan** (dan panggilan):
- **Peran/role utama** (ML Engineer/Data Scientist/dll):
- **Tagline singkat** (1–2 kalimat, gaya formal/santai):
- **Skill inti (5–8)**: framework/bahasa/tools:
- **Proyek unggulan (≤3)**: judul + 1 kalimat ringkas:
- **Kontak & tautan**: email, GitHub, LinkedIn/Portofolio:
- **Riwayat pendidikan** (opsional): format timeline (tahun–sekarang, program, institusi):
- **Preferensi ikon dan warna** (emoji/FontAwesome, aksen tetap ungu atau kombinasinya):
""")
    st.markdown("<div class='caption'>Balas daftar ini di chat, nanti aku inject ke kartu profil di tab ini. ✨</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB: PENJELASAN MODEL (ringkasan utama) ---
with tab_docs:
    st.markdown("<div class='card'><div class='card-title'>📘 Ringkasan Utama — Rock–Paper–Scissors</div>", unsafe_allow_html=True)
    st.markdown("""
**Dataset & Kelas**  
• Rock–Paper–Scissors (Dicoding). 3 kelas: **paper (712)**, **rock (726)**, **scissors (750)** → total **2.188** gambar.  
• Pra-proses: untuk YOLO **640×640 + labeling Roboflow + split 80/10/10**; untuk CNN **224×224 + normalisasi/augmentasi + split 70/20/10**.  
""")
    st.markdown("""
**Arsitektur**  
• **YOLOv8n (anchor-free)**: Backbone–Neck (FPN/PAN)–Head (Detect). Cepat & ringan.  
• **CNN klasifikasi**: 3×(Conv2D+MaxPool) → Flatten → Dense 128 + **Dropout 0.5** → Softmax (3 kelas) + **callbacks** (EarlyStopping, ModelCheckpoint).
""")
    st.markdown("""
**Performa Inti**  
• **YOLOv8n** (100 epoch): Precision ≈ **0.996**, Recall **1.000**, mAP@50 **0.995**, mAP@50–95 **0.925**; inferensi ~**17 ms/gambar** (pre+infer+post).  
• **CNN**: Akurasi validasi ≈ **94%**; precision/recall/F1 stabil di tiap kelas.
""")
    st.markdown("""
**Kesimpulan**  
Model deteksi dan klasifikasi untuk RPS sama-sama **akurat & efisien**. YOLOv8n cocok untuk realtime; CNN memberikan klasifikasi yang stabil pada 3 kelas.
""")
    st.markdown("<div class='caption'>Angka-angka di atas diringkas dari laporanmu (YOLOv8 & CNN).</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
