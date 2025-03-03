# Analisis Bike Sharing 🚴‍♂️📊

Proyek ini bertujuan untuk menganalisis data penyewaan sepeda (Bike Sharing) menggunakan berbagai teknik analisis data. Pengembangan dilakukan menggunakan **Visual Studio Code** sebagai lingkungan pengembangan utama.

## 📌 Fitur Proyek
- **Pertanyaan Bisnis**: Mengenai daftar pertanyaan yang akan dijawab dalam analisis ini.
- **Data Wrangling**: Meliputi Gathering Data, Assesing Data, Cleaning Data.
- **Eksplorasi Data Analis**: Visualisasi pola penggunaan sepeda.
- **Analisis Lanjutan**: Melakukan Clustering Manual, Time Series, Correlation dan Uji ANOVA.
- **Kesimpulan**: Jawaban dari pertanyaan bisnis yang telah dilakukan.


## 🛠️ Teknologi yang Digunakan
- **Python** (Pandas, NumPy, Scipy, Matplotlib, Seaborn)
- **Google Colaboratory** (sebagai editor dalam membuat analisis awal)
- **Streamlit** (Framework yang digunakan)
- **Visual Studio Code** (untuk membangun dashboard dengan streamlit)

## 🚀 Cara Menjalankan Proyek
1. **Clone Repository**
   ```sh
   git clone https://github.com/Ginanti-Riski/streamlit_dataPenyewaanSepeda.git
   cd streamlit_dataPenyewaanSepeda
2. **Buat Virtual Environment**
    ```sh
   python -m venv env
3. **Jalankan Analisis**  

   a. **Aktifkan Virtual Environment & Install Dependencies**  
      - Untuk **MacOS/Linux**:
        ```sh
        source env/bin/activate
        ```
      - Untuk **Windows**:
        ```sh
        env\Scripts\activate
        ```
      - Install dependencies:
        ```sh
        pip install -r requirements.txt
        ```

   b. **Jalankan Streamlit Dashboard**  
      ```sh
      streamlit run Dashboard/dashboard.py
      ```

### 📊 Dataset
Dataset yang digunakan berasal dari Bike Sharing Dataset yang berisi informasi peminjaman sepeda berdasarkan faktor cuaca, musim, hari, dan jam.

### 📌 Catatan
Jika ada kendala atau bug, silakan laporkan melalui Issues pada repository ini.
