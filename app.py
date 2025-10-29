import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="UTS BigData - SufiaHumaira", layout="centered")
st.title("UTS: Klasifikasi & Deteksi Objek — SufiaHumaira")

mode = st.radio("Pilih mode:", ["Klasifikasi Gambar", "Deteksi Objek"])

uploaded_file = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])

# helper simple untuk menampilkan sample images
if st.checkbox("Tampilkan sample images dari repo"):
    sample_dir = "sample_images"
    if os.path.exists(sample_dir):
        pics = os.listdir(sample_dir)[:8]
        for p in pics:
            st.image(os.path.join(sample_dir, p), width=150)
    else:
        st.info("Folder sample_images tidak ditemukan di repo lokal.")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar diupload', use_column_width=True)

    if st.button("Prediksi"):
        st.info("Model sedang memproses...")
        img = image.resize((224, 224))
        arr = np.array(img) / 255.0
        inp = np.expand_dims(arr, axis=0)

        if mode == "Klasifikasi Gambar":
            # Jika ada models/model.h5 maka gunakan (keras); kalau tidak, pakai dummy output
            try:
                from tensorflow.keras.models import load_model
                model_path = os.path.join("models", "model.h5")
                if os.path.exists(model_path):
                    model = load_model(model_path)
                    pred = model.predict(inp)
                    label = np.argmax(pred, axis=1)[0]
                    st.success(f"Hasil Klasifikasi (label index): {label}")
                else:
                    # dummy
                    st.warning("Tidak menemukan models/model.h5 — menampilkan contoh output dummy")
                    st.success("Hasil Klasifikasi (dummy): Kelas_A")
            except Exception as e:
                st.error(f"Error saat memuat model Keras: {e}")

        elif mode == "Deteksi Objek":
            # Jika ada models/model.pt maka gunakan (pytorch); kalau tidak, pakai dummy box
            try:
                import torch
                model_path = os.path.join("models", "model.pt")
                if os.path.exists(model_path):
                    model = torch.load(model_path, map_location='cpu')
                    model.eval()
                    st.success("Model PyTorch ditemukan. (Implementasikan forward pass sesuai modelmu)")
                else:
                    st.warning("Tidak menemukan models/model.pt — menampilkan kotak deteksi dummy")
                    st.success("Deteksi Objek (dummy): 1 objek terdeteksi di koordinat (x1,y1,x2,y2)")
            except Exception as e:
                st.error(f"Error saat memuat model PyTorch: {e}")

st.markdown("---")
st.caption("Project: UTS_BigData_SufiaHumaira — upload folder 'models' dengan model .h5/.pt untuk hasil nyata")
