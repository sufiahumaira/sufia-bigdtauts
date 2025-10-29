UTS_BigData_SufiaHumaira

Isi repo:
- models/: berisi model.h5 dan model.pt (atau gunakan script create_dummy_models.py untuk membuat dummy)
- sample_images/: 10 gambar contoh (5 klasifikasi + 5 deteksi)
- app.py: Streamlit app (jalankan `streamlit run app.py`)
- requirements.txt: dependencies

Cara menjalankan lokal:
1. buat virtualenv: python -m venv venv
2. aktifkan: source venv/bin/activate (Linux/Mac) atau venv\Scripts\activate (Windows)
3. pip install -r requirements.txt
4. python create_dummy_models.py    # opsional: buat dummy model
5. streamlit run app.py

Deploy ke Streamlit Cloud:
1. push repo ke GitHub
2. buka https://share.streamlit.io -> pilih repo -> pilih file app.py -> deploy

Contact: Sufia Humaira
