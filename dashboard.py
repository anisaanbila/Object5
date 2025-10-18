# app.py — RPS Vision Dashboard (Gradient • Poppins • Sidebar Sticky+Collapse)
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from collections import Counter
from urllib.parse import urlencode

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
  padding-top:.8rem!important;
  padding-bottom:2rem;
  max-width:1300px;
}
/* Kolom sejajar atas (untuk header) */
.st-emotion-cache-ocqkz7, .st-emotion-cache-1y4p8pa{
  align-items:flex-start !important;
}

/* Futuristic gradient + grid + starfield */
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

/* Architecture flow (timeline) */
.flow{ position:relative; padding-left:46px; }
.flow:before{ content:""; position:absolute; left:26px; top:6px; bottom:6px; width:4px; background:linear-gradient(#160078,#7226FF); border-radius:4px; }
.flow .node{ position:relative; margin:18px 0; padding-left:0; color:#fff; font-weight:700; font-size:1.05rem;}
.flow .node:before{ content:""; position:absolute; left:-36px; top:2px; width:22px; height:22px; border-radius:50%; border:3px solid rgba(255,255,255,.92); background:rgba(255,255,255,.12); box-shadow:0 0 8px rgba(255,255,255,.35); }

/* Big result title */
.big-result{ font-size:2.2rem; font-weight:800; letter-spacing:.3px; margin:.6rem 0 0 0; color:#fff; }

/* Header right image */
.header-rps-img{ width:100%; max-width:360px; height:auto;
  filter: drop-shadow(0 0 18px rgba(255,255,255,.28)) drop-shadow(0 0 6px rgba(255,255,255,.25)); }

/* Force select labels to white */
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
# ROUTER + SIDEBAR (sticky, collapsible, icon-only)
# =========================
_q = st.experimental_get_query_params()
PAGE = (_q.get("page", ["home"])[0]).lower()
SB_COLLAPSED = _q.get("sb", ["0"])[0] == "1"  # 1 collapsed, 0 expanded

def nav_url(page: str = None, toggle_collapse: bool = False):
    p = PAGE if page is None else page
    sb = (not SB_COLLAPSED) if toggle_collapse else SB_COLLAPSED
    return "?" + urlencode({"page": p, "sb": int(sb)})

SB_WIDTH = 72 if SB_COLLAPSED else 300

st.markdown(
    f"""
    <style>
      .block-container {{ margin-left: {SB_WIDTH}px !important; }}

      .app-sidebar {{
        position: fixed; z-index: 1000; inset: 0 auto 0 0; width: {SB_WIDTH}px;
        background: rgba(18,18,42,.75); backdrop-filter: blur(8px);
        border-right: 1px solid rgba(255,255,255,.07);
        box-shadow: 0 10px 30px rgba(0,0,0,.35);
        padding: 14px 12px; display:flex; flex-direction:column; gap:12px;
      }}
      .sb-top {{
        display:flex; align-items:center; justify-content:space-between;
        padding: 8px 10px; border-radius:12px;
        background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
        border: 1px solid rgba(255,255,255,.06);
      }}
      .sb-title {{
        color:#fff; font-weight:800; letter-spacing:.2px; font-size:1.05rem;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        {"display:none;" if SB_COLLAPSED else ""}
      }}
      .sb-toggle {{
        width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center;
        background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12);
        transition:.18s ease all; text-decoration:none;
      }}
      .sb-toggle:hover {{ transform:translateY(-1px); background: rgba(255,255,255,.12); }}
      .sb-list {{ display:flex; flex-direction:column; gap:8px; margin-top:6px; }}
      .sb-item {{
        display:flex; align-items:center; gap:12px;
        border-radius:14px; padding:10px 12px;
        color:#fff; text-decoration:none; transition:.18s ease all;
      }}
      .sb-item .lbl {{ {"display:none;" if SB_COLLAPSED else ""} font-weight:400; letter-spacing:.2px; }}
      .sb-item:hover {{ background: rgba(255,255,255,.08); transform: translateY(-1px); }}
      .sb-item.active {{
        background: linear-gradient(90deg, rgba(114,38,255,.38), rgba(1,0,48,.26));
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.06), 0 8px 22px rgba(0,0,0,.28);
        font-weight:700;
      }}
      .sb-icon {{
        width:22px; height:22px; flex:0 0 22px; display:inline-flex; align-items:center; justify-content:center;
        filter: drop-shadow(0 0 6px rgba(255,255,255,.22));
      }}
      .sb-item[title]:hover::after {{
        content: attr(title); position:absolute; left: calc(100% + 12px); top: 50%; transform: translateY(-50%);
        background: rgba(0,0,0,.75); color:#fff; font-size:.78rem; padding:6px 8px; border-radius:8px;
        white-space:nowrap; pointer-events:none;
        {"opacity:1;visibility:visible;" if SB_COLLAPSED else "opacity:0;visibility:hidden;"}
      }}
      .sb-footer {{ margin-top:auto; }}
    </style>

    <nav class="app-sidebar">

      <div class="sb-top">
        <div class="sb-title">RPS Dashboard</div>
        <a class="sb-toggle" href="{nav_url(toggle_collapse=True)}" title="Collapse / Expand">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            {"<polyline points='15 18 9 12 15 6'></polyline>" if not SB_COLLAPSED else "<polyline points='9 18 15 12 9 6'></polyline>"}
          </svg>
        </a>
      </div>

      <div class="sb-list">
        <a class="sb-item {'active' if PAGE=='home' else ''}" href="{nav_url('home')}" title="Home">
          <span class="sb-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 9l9-7 9 7"></path><path d="M9 22V12h6v10"></path>
            </svg>
          </span><span class="lbl">Home</span>
        </a>

        <a class="sb-item {'active' if PAGE=='deteksi' else ''}" href="{nav_url('deteksi')}" title="Deteksi Objek">
          <span class="sb-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="3"></circle><path d="M19 12h3"></path><path d="M2 12h3"></path>
              <path d="M12 2v3"></path><path d="M12 19v3"></path>
            </svg>
          </span><span class="lbl">Deteksi Objek</span>
        </a>

        <a class="sb-item {'active' if PAGE=='klasifikasi' else ''}" href="{nav_url('klasifikasi')}" title="Klasifikasi Gambar">
          <span class="sb-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
              <polyline points="2 17 12 22 22 17"></polyline>
              <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
          </span><span class="lbl">Klasifikasi Gambar</span>
        </a>

        <a class="sb-item {'active' if PAGE=='penjelasan' else ''}" href="{nav_url('penjelasan')}" title="Penjelasan Model">
          <span class="sb-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M2 4h7a4 4 0 0 1 4 4v14"></path>
              <path d="M22 4h-7a4 4 0 0 0-4 4v14"></path>
            </svg>
          </span><span class="lbl">Penjelasan Model</span>
        </a>
      </div>

      <div class="sb-footer">
        <a class="sb-item {'active' if PAGE=='profil' else ''}" href="{nav_url('profil')}" title="Profil Pengembang">
          <span class="sb-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
              <circle cx="12" cy="7" r="4"></circle>
            </svg>
          </span><span class="lbl">Profil Pengembang</span>
        </a>
      </div>

    </nav>
    """,
    unsafe_allow_html=True,
)

# =========================
# SHARED HELPERS
# =========================
ICON_PATH = "rps_outline.png"  # ikon header (PNG outline gabungan R/P/S)

def uploader_card(key_label:str, title="Unggah Gambar"):
    st.markdown(f"<div class='card'><div class='card-title' style='font-size:1.35rem'>{title}</div>", unsafe_allow_html=True)
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

# =========================
# PAGES
# =========================
def page_home():
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
        st.markdown(
            """
            <style>
              .header-rps-wrap{
                  display:flex; justify-content:center; align-items:flex-start;
                  margin-top:-48px;
              }
              @media (max-width: 1200px){ .header-rps-wrap{ margin-top:-28px; } }
              @media (max-width: 992px){ .header-rps-wrap{ margin-top:-12px; } }
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

def page_deteksi():
    left, right = st.columns([1.04,1])
    with left:
        f = uploader_card("up_yolo", "Unggah Gambar • Deteksi (RPS)")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='card'><div class='card-title' style='font-size:1.35rem'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            img = None

    with right:
        st.markdown("<div class='card'><div class='card-title' style='font-size:1.35rem'>Hasil Deteksi</div>", unsafe_allow_html=True)
        if img is None:
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

def page_klasifikasi():
    left, right = st.columns([1.04,1])
    with left:
        g = uploader_card("up_cls", "Unggah Gambar • Klasifikasi (RPS)")
        if g:
            img2 = Image.open(g).convert("RGB")
            st.markdown("<div class='card'><div class='card-title' style='font-size:1.35rem'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            img2 = None

    with right:
        st.markdown("<div class='card'><div class='card-title' style='font-size:1.35rem'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
        if img2 is None:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            arr = image.img_to_array(img2.resize((224,224))); arr = np.expand_dims(arr,0)/255.0
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

def page_penjelasan():
    model_choice = st.selectbox("Pilih model yang ingin dijelaskan", ["YOLOv8", "CNN"], index=0)

    # Dataset (dengan counter per kelas)
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
    icons = {
      "Rock": """<path d="M18,30 c-4,0 -8,-3 -8,-7 v-8 c0-6 16-6 16,2 v6 c0,4 -4,7 -8,7z" stroke="white" stroke-width="3" fill="none"/>""",
      "Paper": """<path d="M14,30 c-3,-10 2,-18 8,-18 5,0 6,5 6,10 v8" stroke="white" stroke-width="3" fill="none"/><path d="M10,26 c-2,-7 1,-12 6,-12" stroke="white" stroke-width="3" fill="none"/>""",
      "Scissors": """<path d="M10,12 l8,12 M22,12 l-6,10 M12,26 c4,4 10,4 12,0" stroke="white" stroke-width="3" fill="none"/>"""
    }
    for (k,v), col in zip(counts.items(), colc):
        col.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-top:10px;">
          <div class="icon-bubble">
            <svg viewBox="0 0 36 36">{icons[k]}</svg>
          </div>
          <div>
            <div style="font-weight:700;font-size:1.05rem">{k}</div>
            <div style="font-weight:800;font-size:1.6rem">{v:,}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Arsitektur + Evaluasi
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Arsitektur</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Evaluasi</div>", unsafe_allow_html=True)
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
            metric_bar("Latency (skala cepat)", 1-0.017)  # 17ms ~ cepat
            st.markdown("Akurat & cepat — layak untuk **real-time**.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Kesimpulan
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Kesimpulan</div>", unsafe_allow_html=True)
    if model_choice == "CNN":
        st.markdown("CNN ringkas (3 blok konvolusi + **Dropout 0.5**) dengan **EarlyStopping/Checkpoint** memberi akurasi ~**94%** pada **RPS**. Cocok untuk pengklasifikasi akhir.")
    else:
        st.markdown("**YOLOv8n** presisi tinggi (**mAP@50 ≈ 0.995**) dengan latensi ~**17 ms/gambar**. FPN/PAN + head anchor-free efektif untuk deteksi **RPS** real-time.")
    st.markdown("</div>", unsafe_allow_html=True)

def page_profil():
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
# ROUTE
# =========================
if PAGE == "home":
    page_home()
elif PAGE == "deteksi":
    page_deteksi()
elif PAGE == "klasifikasi":
    page_klasifikasi()
elif PAGE == "penjelasan":
    page_penjelasan()
elif PAGE == "profil":
    page_profil()
else:
    page_home()
