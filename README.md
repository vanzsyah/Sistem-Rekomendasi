# 📝 Tentang Proyek Ini

Proyek ini merupakan bagian dari skripsi yang telah berhasil diterbitkan di jurnal nasional. Penelitian ini fokus pada penerapan teknik machine learning untuk sistem rekomendasi berbasis data teks dan numerik.
[![Demo](https://img.shields.io/badge/-DEMO-red?style=for-the-badge&logo=streamlit&logoColor=white)][(https://sistem-rekomendasi-vanzsyah.streamlit.app/)]



# 📄 Publikasi Jurnal:

    Judul: Sistem Rekomendasi Pembelian Smartphone berbasis Algoritma K-Means dan Singular Value Decomposition
    Jurnal: Jurnal Nasional Teknologi & Sistem Informasi (Teknosi)
    Link: https://teknosi.fti.unand.ac.id/index.php/teknosi/article/view/2650

## Struktur Proyek

```
/Sistem-Rekomendasi
│
├── /data
│   └── clean_data_manual.csv       # Dataset yang digunakan untuk pelatihan dan clustering
│
├── /src
│   └── kmeansvd.py                 # Script untuk clustering dan pelatihan model
│   └── model_data.pkl              # Model dan hasil clustering yang disimpan
│
├── /App
│   └── app.py                      # Aplikasi Streamlit untuk rekomendasi produk
│
└── environment.yml
└── LICENSE.TXT
└── packages.txt
└── README.md
```

## Cara Menjalankan Proyek

### 1. Clone Repositori

```bash
git clone https://github.com/vanzsyah/Sistem-Rekomendasi.git
cd Sistem-Rekomendasi
```

### 2. Install Dependensi

Install library yang diperlukan:

```bash
conda env create -f environment.yml
```

### 3. Siapkan Data

Pastikan file dataset `clean_data_manual.csv` ditempatkan di folder `/data`.

### 4. Latih Model

Jalankan script pelatihan untuk menghasilkan model clustering dan sistem rekomendasi:

```bash
python src/kmeansvd.py
```

### 5. Jalankan Aplikasi Streamlit

Mulai aplikasi Streamlit untuk rekomendasi produk:

```bash
streamlit run App/app.py
```

Buka link localhost yang disediakan di browser Anda untuk berinteraksi dengan aplikasi.

## Fitur

- Pembersihan dan Pra-pemrosesan Teks: Menggunakan contractions dan stopwords removal dari NLTK untuk memastikan kualitas data.
- Ekstraksi Fitur Teks: Menggunakan TF-IDF Vectorizer untuk mengubah data teks menjadi fitur numerik yang dapat digunakan dalam model.
- Clustering Data: Menggunakan algoritma K-Means untuk mengelompokkan data secara otomatis berdasarkan kemiripan.
- Sistem Rekomendasi: Mengimplementasikan Singular Value Decomposition (SVD) dari pustaka Surprise untuk memberikan rekomendasi yang akurat.
- Evaluasi Model: Menggunakan metrik seperti Davies-Bouldin Score dan Cross-Validation untuk mengevaluasi kinerja model.
- Penyimpanan Model: Model machine learning disimpan menggunakan Joblib untuk efisiensi dan kemudahan penggunaan ulang.
- Antarmuka Pengguna Web: Dibangun dengan Streamlit untuk memungkinkan interaksi yang mudah dan cepat dengan sistem rekomendasi.

## Dependensi

Lihat `environment.yml` untuk semua library yang dibutuhkan.


