import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, shapiro, f_oneway


# Judul Halaman
st.title("Analisa Bike Sharing Dataset")
st.write("### Daftar Pertanyaan yang Akan Dianalisis:")
st.write("1. Bagaimana pengaruh faktor cuaca dan musim terhadap jumlah penyewaan sepeda harian?")
st.write("2. Apakah ada perbedaan pola penyewaan sepeda antara hari kerja dan akhir pekan/libur?")
st.markdown("---")

# Sidebar Menu
menu = st.sidebar.selectbox("Pilih Menu", ["Data Wrangling", "Analisis Statistik", "Kesimpulan"])

# State untuk menyimpan data
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.df_clean = None

if menu == "Data Wrangling":
    sub_menu = st.sidebar.radio("Pilih Tahap", ["Data Gathering", "Assessing Data", "Cleaning Data"])
    
    if sub_menu == "Data Gathering":
        st.subheader("Upload Dataset CSV")
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.write("### Data yang Diunggah:")
            st.dataframe(st.session_state.df.head())

    elif sub_menu == "Assessing Data":
        if st.session_state.df is not None:
            st.subheader("Pengecekan Kualitas Data")

            st.write("Jumlah Missing Values:")
            st.write(st.session_state.df.isnull().sum())

            st.write("Jumlah Data Duplikat:", st.session_state.df.duplicated().sum())

            st.write("Deskripsi Statistik Dataset:")
            st.write(st.session_state.df.describe())

            # Deteksi Outlier Menggunakan IQR
            st.subheader("Deteksi Outlier Menggunakan IQR")
            numeric_columns = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns
            binary_columns = [col for col in numeric_columns if st.session_state.df[col].nunique() == 2]
            continuous_columns = [col for col in numeric_columns if col not in binary_columns]

            outlier_counts = {}  # Dictionary untuk menyimpan jumlah outlier per variabel

            for col in continuous_columns:
                Q1 = st.session_state.df[col].quantile(0.25)
                Q3 = st.session_state.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Hitung jumlah outlier di kolom ini
                num_outliers = ((st.session_state.df[col] < lower_bound) | (st.session_state.df[col] > upper_bound)).sum()

                if num_outliers > 0:
                    outlier_counts[col] = num_outliers

            # Tampilkan jumlah variabel yang memiliki outlier
            if outlier_counts:
                st.write(f"Jumlah variabel yang memiliki outlier: **{len(outlier_counts)}** dari {len(continuous_columns)}")
                df_outlier_info = pd.DataFrame(outlier_counts.items(), columns=["Variabel", "Jumlah Outlier"])
                df_outlier_info.index += 1  # Menambahkan nomor urut
                st.dataframe(df_outlier_info)  # Menampilkan dalam format tabel interaktif
            else:
                st.success("Tidak ada outlier yang terdeteksi dalam dataset.")

            # Visualisasi Outlier
            st.subheader("Visualisasi Outlier")
            if continuous_columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(data=st.session_state.df[continuous_columns], ax=ax)
                plt.xticks(rotation=40)
                st.pyplot(fig)
            else:
                st.warning("Tidak ada variabel kontinu untuk divisualisasikan.")
        else:
            st.warning("Silakan unggah data terlebih dahulu di Data Gathering!")

    elif sub_menu == "Cleaning Data":
        if st.session_state.df is not None:
            st.subheader("Data Setelah Dibersihkan")

            # Konversi kolom tanggal ke format datetime (hanya tanggal, tanpa waktu)
            datetime_columns = ["dteday"]
            for column in datetime_columns:
                if column in st.session_state.df.columns:
                    st.session_state.df[column] = pd.to_datetime(st.session_state.df[column]).dt.normalize()

            df_cleaned_final = st.session_state.df.copy()

            # Menampilkan data tanpa waktu di Streamlit
            st.write(df_cleaned_final.style.format({"dteday": lambda x: x.strftime("%Y-%m-%d")}))

            # Deteksi kolom numerik dan biner
            numeric_columns = df_cleaned_final.select_dtypes(include=['int64', 'float64']).columns
            binary_columns = [col for col in numeric_columns if df_cleaned_final[col].nunique() == 2]
            continuous_columns = [col for col in numeric_columns if col not in binary_columns]

            # Fungsi untuk menghapus outlier menggunakan metode IQR
            def remove_outliers_iqr(df, column, multiplier=1.0):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

            # Hapus outlier hanya dari kolom non-biner
            for col in continuous_columns:
                df_cleaned_final = remove_outliers_iqr(df_cleaned_final, col)

            # Cek apakah data tidak kosong setelah pembersihan
            if not df_cleaned_final.empty:
                st.session_state.df_clean = df_cleaned_final
                
                st.subheader("Statistik Data Setelah Cleaning")
                if "dteday" in st.session_state.df_clean.columns:
                    st.session_state.df_clean["dteday"] = pd.to_datetime(st.session_state.df_clean["dteday"]).dt.date
                    st.session_state.df_clean["dteday"] = pd.to_datetime(st.session_state.df_clean["dteday"])
                
                numeric_columns = st.session_state.df_clean.select_dtypes(include=['int64', 'float64']).columns
                st.write(st.session_state.df_clean[numeric_columns].describe())

                st.subheader("Visualisasi Data Setelah Outlier Dihapus")
                if continuous_columns:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.boxplot(data=st.session_state.df_clean[continuous_columns], ax=ax)
                    plt.xticks(rotation=40)
                    st.pyplot(fig)
                else:
                    st.warning("Tidak ada variabel kontinu untuk divisualisasikan.")

                st.write(f"📌 **Jumlah data sebelum pembersihan:** `{len(st.session_state.df)}`")
                st.write(f"📌 **Jumlah data setelah pembersihan:** `{len(st.session_state.df_clean)}`")

                if len(st.session_state.df_clean) < len(st.session_state.df) * 0.1:
                    st.warning("⚠️ Data yang tersisa kurang dari 10% setelah pembersihan outlier. Pertimbangkan untuk menyesuaikan parameter IQR.")
            else:
                st.warning("❗ Data menjadi kosong setelah pembersihan outlier. Silakan ubah parameter IQR atau cek dataset.")
        else:
            st.warning("Silakan unggah data terlebih dahulu di Data Gathering!")

elif menu == "Analisis Statistik":
    sub_analysis = st.sidebar.radio("Pilih Analisis", ["Analisis Awal", "Analisis Clustering Manual", "Analisis Time Series", "Analisis Korelasi dan Uji ANOVA"])
    
    if st.session_state.df_clean is not None:
        df_cleaned_final = st.session_state.df_clean
        
        if sub_analysis == "Analisis Awal":
                
            season_stats = df_cleaned_final.groupby("season")["cnt"].describe()
            st.write("### Statistik Deskriptif Jumlah Penyewaan Berdasarkan Musim")
            st.write(season_stats)
            # Insight Musim:
            st.write("📌 **Insight Musim:**")
            st.write("""
            - **Musim Gugur (Fall) memiliki penyewaan tertinggi**, kemungkinan karena cuaca yang lebih nyaman untuk bersepeda.
            - **Musim Dingin (Winter) memiliki penyewaan terendah**, mungkin disebabkan oleh suhu dingin dan kondisi yang kurang mendukung.
            """)

            weather_stats = df_cleaned_final.groupby("weathersit")["cnt"].describe()
            st.write("### Statistik Deskriptif Jumlah Penyewaan Berdasarkan Cuaca")
            st.write(weather_stats)
            # Insight Cuaca:
            st.write("📌 **Insight Cuaca:**")
            st.write("""
            - **Penyewaan tertinggi terjadi saat cuaca cerah atau sedikit berawan (kategori 1).**
            - **Saat cuaca buruk (hujan deras atau salju, kategori 3), penyewaan turun drastis.**
            - Ini menunjukkan bahwa kondisi cuaca sangat berpengaruh terhadap keputusan orang untuk menyewa sepeda.
            """)

            correlation_matrix = df_cleaned_final[['season', 'weathersit', 'cnt']].corr()
            st.write("### Korelasi Faktor Cuaca & Musim dengan Penyewaan Sepeda")
        
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            # Insight Korelasi
            st.write("📌 **Insight Korelasi:**")
            st.write("""
            - **Cuaca memiliki korelasi negatif dengan penyewaan sepeda (-0.234)**, artinya semakin buruk cuaca, semakin sedikit sepeda yang disewa.
            - **Musim juga mempengaruhi penyewaan**, tetapi tidak sebesar pengaruh cuaca.
            """)

            # Visualisasi Rata-rata Penyewaan Sepeda pada Hari Kerja vs Akhir Pekan
            st.write("### Rata-rata Penyewaan Sepeda pada Hari Kerja vs Akhir Pekan")

            # Warna untuk visualisasi
            base_color = "#A6D785"  # Light green
            highlight_color = "#228B22"  # Dark green

            # Membuat plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=df_cleaned_final['workingday'], 
                y=df_cleaned_final['cnt'], 
                ci=None, 
                palette=[base_color, highlight_color]  # Warna yang ditentukan
            )

            # Menambahkan judul dan label
            ax.set_title("Perbandingan Penyewaan Sepeda: Hari Kerja vs Akhir Pekan", fontsize=12)
            ax.set_ylabel("Rata-rata Penyewaan Sepeda", fontsize=12)
            ax.set_xlabel("Jenis Hari", fontsize=12)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Akhir Pekan/Libur", "Hari Kerja"], fontsize=11)

            # Menambahkan anotasi pada batang
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.0f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

            # Menampilkan grafik terlebih dahulu
            st.pyplot(fig)

            # Menampilkan insight di bawah grafik
            st.write("📌 **Insight Hari Kerja vs Akhir Pekan:**")
            st.write("""
            - **Penyewaan lebih tinggi pada hari kerja dibandingkan akhir pekan.**
            - Ini menunjukkan bahwa sepeda lebih banyak digunakan sebagai alat transportasi sehari-hari, bukan hanya untuk rekreasi.
            """)

            # Uji Statistik (T-Test)
            workday_rentals = df_cleaned_final[df_cleaned_final["workingday"] == 1]["cnt"]
            weekend_rentals = df_cleaned_final[df_cleaned_final["workingday"] == 0]["cnt"]
            t_stat, p_value = ttest_ind(workday_rentals, weekend_rentals, equal_var=False)

            # Menampilkan hasil uji t-test
            st.write(f"📊 **Hasil Uji t-test:** t-statistic = {t_stat:.2f}, p-value = {p_value:.5f}")
            st.write("📌 P-value yang sangat kecil mengindikasikan bahwa perbedaan jumlah penyewaan antara hari kerja dan akhir pekan signifikan secara statistik, bukan terjadi secara kebetulan.")

            
            # Mapping angka ke nama musim
            season_labels = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}

            # Menghitung rata-rata jumlah penyewaan sepeda per musim
            season_means = df_cleaned_final.groupby('season')['cnt'].mean()

            # Mengonversi indeks menjadi label musim
            season_means.index = season_means.index.map(season_labels)

            # Menentukan musim dengan jumlah penyewaan tertinggi
            max_season = season_means.idxmax()

            # Warna dasar untuk semua batang
            base_color = "#A6D785"  # Hijau muda
            highlight_color = "#228B22"  # Hijau gelap untuk batang tertinggi

            # Membuat daftar warna untuk setiap batang
            colors = [highlight_color if season == max_season else base_color for season in season_means.index]

            # Plot visualisasi untuk musim
            st.write("### Rata-rata Penyewaan Sepeda Berdasarkan Musim")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=season_means.index, y=season_means.values, palette=colors, ci=None, ax=ax)

            # Menambahkan label kuantitas di setiap batang
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.0f}',  
                            (p.get_x() + p.get_width() / 2, p.get_height()), 
                            ha='center', va='bottom', fontsize=12, fontweight='bold')

            # Menyesuaikan tampilan
            ax.set_title("Rata-rata Penyewaan Sepeda Berdasarkan Musim", fontsize=14)
            ax.set_xlabel("Musim", fontsize=12)
            ax.set_ylabel("Rata-rata Jumlah Penyewaan Sepeda", fontsize=12)
            ax.set_ylim(0, season_means.max() * 1.1)  

            # Tampilkan plot di Streamlit
            st.pyplot(fig)
            # Menampilkan Insight
            st.write("### 📌 **Insight Penyewaan Sepeda Berdasarkan Musim**")
            st.markdown("""
            1️⃣ **Musim panas (Summer) memiliki jumlah penyewaan tertinggi**  
            🔹 Musim panas (season 3) memiliki rata-rata penyewaan sepeda tertinggi dibandingkan musim lainnya.  
            🔹 Hal ini mungkin disebabkan oleh cuaca yang lebih mendukung untuk bersepeda, seperti suhu yang nyaman dan kondisi jalan yang lebih baik.  

            2️⃣ **Musim dingin (Winter) memiliki jumlah penyewaan terendah**  
            🔹 Musim dingin (season 1) menunjukkan rata-rata penyewaan yang jauh lebih rendah dibandingkan musim lainnya.  
            🔹 Ini bisa disebabkan oleh kondisi cuaca yang lebih ekstrem, seperti suhu dingin, hujan, atau salju yang membuat orang enggan bersepeda.  

            3️⃣ **Musim semi (Spring) dan musim gugur (Fall) memiliki jumlah penyewaan yang hampir sama**  
            🔹 Musim semi (season 2) dan musim gugur (season 4) memiliki jumlah penyewaan yang relatif mirip.  
            🔹 Ini menunjukkan bahwa kedua musim ini menawarkan kondisi yang cukup nyaman bagi pengguna sepeda.  

            4️⃣ **Cuaca berpengaruh terhadap tren penggunaan sepeda**  
            🔹 Bisa disimpulkan bahwa semakin baik cuaca dan kondisi lingkungan, semakin tinggi minat masyarakat dalam menyewa sepeda.  

            ---

            ### 🎯 **Rekomendasi Berdasarkan Insight:**  
            ✅ **Promosi penyewaan sepeda lebih agresif di musim dingin**  
            📌 Operator penyewaan sepeda bisa menawarkan diskon atau promosi khusus di musim dingin untuk meningkatkan jumlah penyewaan.  

            ✅ **Persiapan lebih banyak sepeda di musim panas**  
            📌 Karena permintaan meningkat di musim panas, perusahaan bisa menyiapkan lebih banyak sepeda agar bisa memenuhi kebutuhan pelanggan.  

            ✅ **Analisis lebih lanjut tentang faktor lain**  
            📌 Perlu dianalisis apakah faktor lain seperti hari libur atau hari kerja juga berpengaruh terhadap jumlah penyewaan.  

            ---

            🚴‍♂️ **Kesimpulan:**  
            Musim berperan penting dalam tren penyewaan sepeda, dengan musim panas sebagai puncaknya dan musim dingin sebagai yang terendah.  

            """)


            # Visualisasi untuk kondisi cuaca
            st.write("### Rata-rata Penyewaan Sepeda Berdasarkan Kondisi Cuaca")

            weather_labels = {1: "Clear", 2: "Mist", 3: "Light Rain/Snow", 4: "Heavy Rain/Snow"}
            weather_means = df_cleaned_final.groupby('weathersit')['cnt'].mean()

            # Menentukan kondisi cuaca dengan penyewaan tertinggi
            max_weather = weather_means.idxmax()

            # Warna batang
            weather_colors = ["#228B22" if weather == max_weather else "#A6D785" for weather in weather_means.index]

            # Buat plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=[weather_labels[w] for w in weather_means.index], y=weather_means.values, palette=weather_colors, ci=None, ax=ax)

            # Tambahkan label kuantitas di setiap batang
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.0f}',  
                            (p.get_x() + p.get_width() / 2, p.get_height()), 
                            ha='center', va='bottom', fontsize=12, fontweight='bold')

            # Sesuaikan tampilan
            ax.set_title("Rata-rata Penyewaan Sepeda Berdasarkan Kondisi Cuaca", fontsize=14)
            ax.set_xlabel("Kondisi Cuaca", fontsize=12)
            ax.set_ylabel("Rata-rata Jumlah Penyewaan Sepeda", fontsize=12)
            ax.set_ylim(0, weather_means.max() * 1.1)  

            # Tampilkan plot di Streamlit
            st.pyplot(fig)
            # **Insight**
            st.write("### 📌 **Insight Penyewaan Sepeda Berdasarkan Cuaca**")
            st.markdown(f"""
            1️⃣ **Cuaca yang lebih cerah meningkatkan jumlah penyewaan sepeda** 🏖️  
            🔹 Pada kondisi cuaca **Clear/Few Clouds/Partly Cloudy**, rata-rata penyewaan sepeda adalah yang tertinggi (**{season_means[1]:.0f} penyewaan**).  
            🔹 Ini menunjukkan bahwa orang lebih suka menyewa sepeda saat cuaca cerah.  

            2️⃣ **Cuaca berkabut atau mendung sedikit mengurangi penyewaan** 🌫️  
            🔹 Pada kondisi **Mist/Cloudy**, rata-rata penyewaan turun menjadi **{season_means[2]:.0f} penyewaan**.  
            🔹 Meskipun lebih rendah dari kondisi cerah, jumlah penyewaan masih cukup tinggi, menunjukkan bahwa kabut atau mendung tidak terlalu berdampak besar pada keputusan penyewaan.  

            3️⃣ **Cuaca hujan atau salju drastis menurunkan penyewaan sepeda** ☔❄️  
            🔹 Pada kondisi **Light Rain/Snow**, rata-rata penyewaan turun drastis menjadi **{season_means[3]:.0f} penyewaan**.  
            🔹 Ini masuk akal karena hujan atau salju membuat kondisi jalan lebih berbahaya dan kurang nyaman untuk bersepeda.  

            ---

            ### 🎯 **Rekomendasi berdasarkan insight:**  
            ✅ **Menyesuaikan jumlah sepeda berdasarkan cuaca** ☀️🌧️  
            📌Saat cuaca cerah, pastikan jumlah sepeda yang tersedia cukup untuk memenuhi permintaan yang tinggi.  
            📌 Saat cuaca buruk (hujan/salju), operator bisa mengurangi jumlah sepeda yang disediakan atau menawarkan layanan promosi khusus untuk menarik pelanggan.  

            ✅ **Mempersiapkan layanan tambahan untuk kondisi cuaca buruk** ☂️  
            📌 Menyediakan perlengkapan tambahan seperti jas hujan atau payung bagi pengguna sepeda saat kondisi mendung/hujan ringan agar penyewaan tetap berjalan.  
            📌 Menawarkan harga diskon atau promo khusus pada hari-hari dengan cuaca buruk untuk meningkatkan jumlah penyewaan.  

            ✅ **Melakukan prediksi tren penyewaan berbasis cuaca** 📊  
            📌 Dengan menggunakan data cuaca sebelumnya, bisa dibuat model prediksi untuk memperkirakan jumlah penyewaan berdasarkan kondisi cuaca.  

            ---

            🚴‍♂️ **Kesimpulan:**  
            Cuaca berperan besar dalam jumlah penyewaan sepeda, di mana kondisi cerah mendorong lebih banyak penyewaan, sedangkan hujan atau salju secara signifikan menurunkannya.
            """)
          
        elif sub_analysis == "Analisis Clustering Manual":
            st.subheader("📊 **Analisis Segmentasi Data dengan Clustering**")
            st.markdown("### 🔍 **Pembagian Kategori Penyewaan Sepeda**")

            # Hitung kuartil
            q1 = df_cleaned_final["cnt"].quantile(0.25)
            q3 = df_cleaned_final["cnt"].quantile(0.75)

            # Buat kategori rental berdasarkan kuartil
            def categorize_rental(count):
                if count < q1:
                    return "Low Rental"
                elif count > q3:
                    return "High Rental"
                else:
                    return "Medium Rental"

            df_cleaned_final["rental_category"] = df_cleaned_final["cnt"].apply(categorize_rental)

            # Pastikan urutan kategori benar
            category_order = ["Low Rental", "Medium Rental", "High Rental"]
            df_cleaned_final["rental_category"] = pd.Categorical(df_cleaned_final["rental_category"], categories=category_order, ordered=True)

            # Hitung jumlah masing-masing kategori
            category_counts = df_cleaned_final["rental_category"].value_counts().reindex(category_order)

            # Warna dasar dan highlight
            base_color = "#A6D785"  # Light green
            highlight_color = "#228B22"  # Dark green

            # Tentukan warna untuk batang tertinggi
            colors = [highlight_color if count == max(category_counts) else base_color for count in category_counts]

            # Visualisasi jumlah penyewaan berdasarkan kategori
            plt.figure(figsize=(8, 5))
            ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)

            # Tambahkan angka di atas setiap batang
            for p in ax.patches:
                ax.annotate(f"{int(p.get_height())}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom',
                            fontsize=12, fontweight='bold', color="black")

            # Atur tampilan
            plt.title("Distribusi Kategori Penyewaan Sepeda", fontsize=12)
            plt.xlabel("Kategori Rental", fontsize=12)
            plt.ylabel("Jumlah Hari", fontsize=12)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            # Hapus garis latar
            sns.despine()

            # Tampilkan plot di Streamlit
            st.pyplot(plt)

            # Insight
            st.markdown("""
            ## 🔎 **Insight dari Clustering Penyewaan Sepeda**  

            **1️⃣ "Medium Rental" Mendominasi** 📈  
            ✅ Sebagian besar hari dalam dataset berada dalam kategori **Medium Rental**, dengan **299 hari** di dalamnya.  
            ✅ Ini menunjukkan bahwa pola penyewaan sepeda cenderung **berada di tingkat menengah**, bukan ekstrem rendah atau tinggi.  

            **2️⃣ "Low" dan "High Rental" Memiliki Jumlah yang Sama** ⚖️  
            ✅ Kategori **Low Rental** dan **High Rental** masing-masing terjadi selama **150 hari**.  
            ✅ Artinya, jumlah hari dengan penyewaan yang sangat rendah **sama banyaknya** dengan jumlah hari dengan penyewaan tinggi.  

            **3️⃣ Distribusi yang Simetris** 📊  
            ✅ Penyebaran data menunjukkan bahwa jumlah penyewaan **berpusat di kategori Medium**, dengan jumlah hari di kategori Low dan High yang seimbang.  
            ✅ Hal ini bisa menunjukkan **tren musiman**, cuaca, atau faktor eksternal lain yang memengaruhi pola penyewaan sepeda.  

            **4️⃣ Potensi untuk Meningkatkan High Rental** 🚀  
            ✅ Karena jumlah hari dengan penyewaan tinggi **tidak mendominasi**, ada **peluang untuk meningkatkan jumlah hari** dalam kategori High Rental.  
            ✅ Beberapa strategi yang bisa diterapkan:  
            🔹 **Promosi atau diskon** di akhir pekan untuk menarik lebih banyak pelanggan.  
            🔹 **Event atau kampanye khusus** untuk mendorong penggunaan sepeda lebih sering.  
            🔹 **Penyediaan fasilitas tambahan** seperti layanan antar-jemput atau diskon bagi pelanggan tetap.  

            ---

            ## 📌 **Kesimpulan**  
            Data menunjukkan bahwa tren penyewaan sepeda **lebih sering berada di level menengah** dibandingkan ekstrem rendah atau tinggi.  
            Namun, ada **potensi besar untuk meningkatkan jumlah hari dengan penyewaan tinggi** melalui strategi bisnis yang tepat.  
            🚴💡 Dengan optimalisasi layanan dan promosi yang tepat, jumlah penyewaan dapat **didorong ke level yang lebih tinggi!**  
            """)


        elif sub_analysis == "Analisis Time Series":
            st.subheader("Tren Musiman Penyewaan Sepeda 🚴‍♂️📊")
            
            # Hitung rata-rata jumlah penyewaan per musim
            seasonal_trend = df_cleaned_final.groupby("season")["cnt"].mean()

            # Visualisasi tren musiman
            plt.figure(figsize=(8, 5))
            ax = sns.lineplot(
                x=seasonal_trend.index, 
                y=seasonal_trend.values, 
                marker="o", 
                color="#228B22",  # Dark Green
                linewidth=2.5
            )

            # Tambahkan angka di setiap titik
            for x, y in zip(seasonal_trend.index, seasonal_trend.values):
                ax.annotate(f"{int(y)}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=11, fontweight='bold', color="black")

            # Atur sumbu Y agar mulai dari 0
            plt.ylim(0, seasonal_trend.max() + 500)  # Tambahkan margin di atas

            # Atur label dan judul
            plt.title("Rata-rata Penyewaan Sepeda Berdasarkan Musim", fontsize=12)
            plt.ylabel("Rata-rata Penyewaan", fontsize=12)
            plt.xlabel("Musim", fontsize=12)
            plt.xticks(ticks=[1, 2, 3, 4], labels=["Winter", "Spring", "Summer", "Fall"], fontsize=11)
            plt.yticks(fontsize=11)

            # Hapus garis latar untuk tampilan lebih bersih
            sns.despine()

            # Tampilkan plot
            st.pyplot(plt)

            # Insight Analysis
            st.subheader("🔍 Insight: Tren Penyewaan Sepeda Berdasarkan Musim")
            
            st.markdown("""
            ### ❄️ Penyewaan Sepeda Terendah di Musim Dingin (Winter - **2647**)
            - Musim dingin menjadi periode dengan penyewaan sepeda paling sedikit.
            - Cuaca ekstrem seperti suhu rendah, hujan, atau salju mungkin menjadi penyebab utama rendahnya minat pengguna.
            - **Strategi:** Menawarkan diskon khusus atau fasilitas seperti pakaian hangat dan perlengkapan musim dingin untuk menarik penyewa.

            ### 🌸 Lonjakan Signifikan di Musim Semi (Spring - **4748**)
            - Saat cuaca mulai menghangat, penyewaan meningkat hampir **2x lipat** dibandingkan musim dingin.
            - Banyak orang kembali beraktivitas di luar ruangan, menjadikan sepeda pilihan transportasi yang lebih populer.
            - **Strategi:** Promosi keanggotaan atau paket langganan di awal musim semi bisa mendorong lebih banyak pelanggan.

            ### ☀️ Puncak Penyewaan di Musim Panas (Summer - **5490**)
            - Musim panas adalah periode **terbaik** untuk bisnis penyewaan sepeda.
            - Liburan musim panas, cuaca cerah, dan lebih banyak aktivitas luar ruangan berkontribusi terhadap lonjakan ini.
            - **Strategi:** Mengadakan event bersepeda, promo family pack, atau penyewaan dengan durasi lebih lama untuk menarik lebih banyak pelanggan.

            ### 🍂 Penurunan Bertahap di Musim Gugur (Fall - **4672**)
            - Penyewaan mulai menurun saat memasuki musim gugur, seiring cuaca yang mulai lebih dingin.
            - Banyak orang yang mulai mengurangi aktivitas luar ruangan menjelang musim dingin.
            - **Strategi:** Promo "Akhir Musim" atau penawaran diskon untuk langganan musim gugur bisa membantu mengurangi dampak penurunan ini.

            ---
            
            ### 📌 **Kesimpulan & Rekomendasi**
            🔹 **Cuaca sangat memengaruhi pola penyewaan sepeda** – memahami tren musiman bisa membantu strategi pemasaran yang lebih efektif.  
            🔹 **Fokus pada musim dingin** dengan insentif bagi penyewa agar minat tidak terlalu menurun drastis.  
            🔹 **Maksimalkan musim panas** dengan kampanye pemasaran dan program loyalitas.  
            🔹 **Persiapkan strategi transisi dari musim gugur ke musim dingin** agar tidak terjadi penurunan drastis dalam penyewaan.  

            🚀 **Dengan strategi yang tepat, tren musiman ini bisa dimanfaatkan untuk meningkatkan pendapatan dan memperluas jangkauan bisnis penyewaan sepeda!** 💡
            """)



        elif sub_analysis == "Analisis Korelasi dan Uji ANOVA":
            # Judul dan Header
            st.subheader("📊 Hubungan Antar Variabel & Uji ANOVA")

            # Hitung korelasi antara musim, cuaca, dan jumlah penyewaan
            correlation = df_cleaned_final[["season", "weathersit", "cnt"]].corr()

            # 📌 Heatmap Korelasi
            st.write("### 🔥 Heatmap Korelasi antara Musim, Cuaca, dan Penyewaan")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True, ax=ax)
            st.pyplot(fig)

            # 🔍 Insight Korelasi
            st.markdown(
                """
                **🔹 Insight:**
                - 📈 **Musim memiliki korelasi positif (0.43)** dengan penyewaan sepeda, artinya lebih banyak sepeda disewa saat musim gugur.
                - 🌧️ **Cuaca memiliki korelasi negatif (-0.23)**, menunjukkan bahwa semakin buruk cuaca, semakin sedikit sepeda yang disewa.
                - 🌤️ **Musim dan cuaca hampir tidak berkorelasi (0.018)**, artinya kondisi cuaca tidak selalu mengikuti pola musim.
                """
            )

            # 📌 Uji Normalitas Shapiro-Wilk
            st.write("### 🧪 Uji Normalitas Shapiro-Wilk")
            stat, p_shapiro = shapiro(df_cleaned_final["cnt"])
            st.write(f"📌 **p-value = {p_shapiro:.5f}**")

            if p_shapiro > 0.05:
                st.success("✅ Data terdistribusi normal. Lanjutkan dengan uji parametrik seperti ANOVA.")
            else:
                st.warning("⚠️ Data tidak terdistribusi normal. Pertimbangkan uji non-parametrik seperti Mann-Whitney.")

            # 📌 Uji ANOVA
            st.write("### 🏆 Uji ANOVA: Perbedaan Penyewaan Berdasarkan Musim")
            anova_result = f_oneway(
                df_cleaned_final[df_cleaned_final["season"] == 1]["cnt"],
                df_cleaned_final[df_cleaned_final["season"] == 2]["cnt"],
                df_cleaned_final[df_cleaned_final["season"] == 3]["cnt"],
                df_cleaned_final[df_cleaned_final["season"] == 4]["cnt"]
            )
            st.write(f"📌 **F-statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.5f}**")

            if anova_result.pvalue < 0.05:
                st.success("✅ Hasil ANOVA menunjukkan ada **perbedaan signifikan** dalam penyewaan berdasarkan musim.")
            else:
                st.warning("⚠️ Tidak ada perbedaan signifikan dalam penyewaan berdasarkan musim.")

            # 📌 Visualisasi Rata-rata Penyewaan Berdasarkan Cuaca
            st.write("### Rata-rata Penyewaan Berdasarkan Kategori Cuaca")

            # Warna dasar dan highlight
            base_color = "#A6D785"  # Light green
            highlight_color = "#228B22"  # Dark green

            # Membuat plot
            fig, ax = plt.subplots(figsize=(8, 5))
            weather_avg_rentals = df_cleaned_final.groupby("weathersit")["cnt"].mean()

            # Plot dengan warna kustom
            bars = sns.barplot(
                x=weather_avg_rentals.index, 
                y=weather_avg_rentals.values, 
                ax=ax, 
                palette=[base_color if i != weather_avg_rentals.idxmax() else highlight_color for i in weather_avg_rentals.index]
            )

            # Menampilkan angka di atas batang
            for bar in bars.patches:
                ax.annotate(
                    f'{bar.get_height():.0f}', 
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black'
                )

            # Menambahkan judul dan label
            ax.set_title("Rata-rata Penyewaan Berdasarkan Kategori Cuaca", fontsize=12)
            ax.set_xlabel("Kategori Cuaca", fontsize=12)
            ax.set_ylabel("Rata-rata Penyewaan", fontsize=12)
            ax.set_xticklabels(["Cerah", "Mendung", "Hujan/Salju"], fontsize=11)

            # Menampilkan plot di Streamlit
            st.pyplot(fig)


            st.markdown(
                """
                **🔹 Insight:**
                - 🌞 **Cuaca Cerah (Kategori 1)** memiliki penyewaan tertinggi (**4.876** sepeda/hari).
                - ☁️ **Cuaca Mendung (Kategori 2)** menurunkan penyewaan menjadi **4.035** sepeda/hari.
                - ⛈️ **Cuaca Buruk (Kategori 3)** sangat mengurangi penyewaan (**1.803** sepeda/hari).
                - **Strategi Bisnis**: 🚲 **Promosi diskon atau layanan tambahan** saat cuaca buruk dapat membantu meningkatkan penyewaan.
                """
            )

            # 📌 Visualisasi Rata-rata Penyewaan Berdasarkan Musim
            st.write("### Rata-rata Penyewaan Berdasarkan Kategori Musim")

            # Warna dasar dan highlight
            base_color = "#A6D785"  # Light green
            highlight_color = "#228B22"  # Dark green

            # Membuat plot
            fig, ax = plt.subplots(figsize=(8, 5))
            weather_avg_rentals = df_cleaned_final.groupby("season")["cnt"].mean()

            # Plot dengan warna kustom
            bars = sns.barplot(
                x=weather_avg_rentals.index, 
                y=weather_avg_rentals.values, 
                ax=ax, 
                palette=[base_color if i != weather_avg_rentals.idxmax() else highlight_color for i in weather_avg_rentals.index]
            )

            # Menampilkan angka di atas batang
            for bar in bars.patches:
                ax.annotate(
                    f'{bar.get_height():.0f}', 
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black'
                )

            # Menambahkan judul dan label
            ax.set_title("Rata-rata Penyewaan Berdasarkan Kategori Musim", fontsize=12)
            ax.set_xlabel("Kategori Musim", fontsize=12)
            ax.set_ylabel("Rata-rata Penyewaan", fontsize=12)
            ax.set_xticklabels(["Spring", "Summer", "Fall","Winter"], fontsize=11)

            # Menampilkan plot di Streamlit
            st.pyplot(fig)         
     
            st.markdown(
                """
                **🔹 Insight:**
                - 🍁 **Musim Gugur (Fall) memiliki penyewaan tertinggi** (**5.644** sepeda/hari).
                - 🌱 **Musim Semi (Spring) memiliki penyewaan terendah** (**2.604** sepeda/hari).
                - 📊 **Polanya: Spring → Summer → Fall (puncak) → Winter**.
                - **Strategi Bisnis**:
                - 🚴 **Tambahkan sepeda lebih banyak saat musim gugur** karena permintaan tinggi.
                - 🎯 **Gunakan promo & event saat musim semi** untuk meningkatkan penyewaan.
                """
            )

            # 📌 Kesimpulan dan Rekomendasi
            st.write("### 🎯 Kesimpulan & Rekomendasi")
            st.markdown(
                """
                ✅ **Kesimpulan:**
                - 📆 **Musim gugur adalah waktu terbaik** untuk bisnis rental sepeda.
                - 🌧️ **Cuaca buruk sangat memengaruhi penyewaan** sepeda.
                - 📊 Uji ANOVA menunjukkan **perbedaan signifikan** dalam jumlah penyewaan berdasarkan musim.

                🎯 **Rekomendasi:**
                - 🚴 **Sediakan lebih banyak sepeda di musim gugur** untuk memenuhi permintaan.
                - 💰 **Buat promo khusus saat musim semi & cuaca buruk** untuk meningkatkan penyewaan.
                - 🛠️ **Pertimbangkan sepeda tahan cuaca** untuk meningkatkan jumlah penyewaan sepanjang tahun.
                """
            )

        else:
            st.warning("Silakan lakukan pembersihan data terlebih dahulu!")

elif menu == "Kesimpulan":
    st.subheader("📌 Kesimpulan")
    
    # Menambahkan garis pemisah dekoratif
    st.markdown("---")

    # Kesimpulan Pertanyaan 1
    st.markdown("### 💡 Kesimpulan Pertanyaan 1")
    st.info(
        "Cuaca dan musim berpengaruh signifikan terhadap jumlah penyewaan sepeda. "
        "Pengguna cenderung lebih banyak menyewa sepeda pada musim gugur & musim panas. "
        "Cuaca buruk mengurangi jumlah penyewaan secara signifikan."
    )

    # Kesimpulan Pertanyaan 2
    st.markdown("### 💡 Kesimpulan Pertanyaan 2")
    st.success(
        "Penyewaan sepeda lebih tinggi pada hari kerja, kemungkinan besar karena penggunaan "
        "untuk transportasi kerja atau sekolah. Pada akhir pekan, jumlah penyewaan berkurang, "
        "kemungkinan karena orang lebih sedikit bepergian atau lebih memilih kendaraan lain untuk rekreasi."
    )

    # Menambahkan garis pemisah di akhir
    st.markdown("---")

   
