import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Title
st.title("Case Study Machine Learning GDGoC UNSRI 2024/2025")

# Introduction
st.markdown("""
### Nama    : Elisa Dwi Nanda  
### Jurusan : Sistem Komputer  
### Angkatan: 2023  
""")

# Dataset Description
st.header("Dataset yang Digunakan")
st.markdown("""
Dataset yang digunakan dalam case study ini adalah **'Data Science Salaries 2024'**, yang dapat diakses melalui tautan berikut: [Dataset Data Science Salaries 2024-Kaggle](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)
""")

st.subheader("Deskripsi Dataset")
st.markdown("""
Dataset ini berisi informasi tentang gaji para profesional di bidang data science untuk tahun 2024. Data mencakup berbagai atribut penting seperti:

- **work_year**: Tahun kerja.
- **experience_level**: Tingkat pengalaman (Entry-Level, Mid-Level, Senior, Executive).
- **employment_type**: Jenis pekerjaan (Full-Time, Part-Time, Contract, Freelance).
- **job_title**: Pekerjaan.
- **salary**: Gaji dalam mata uang lokal.
- **salary_currency**: Mata uang gaji.
- **salary_in_usd**: Gaji dalam USD.
- **employee_residence**: Negara tempat tinggal karyawan.
- **remote_ratio**: Persentase kerja jarak jauh (0, 50, atau 100).
- **company_location**: Lokasi perusahaan.
- **company_size**: Ukuran perusahaan berdasarkan jumlah karyawan (S: Small, M: Medium, L: Large).
""")

# Load Dataset
st.header("1. Dataset Information")

st.subheader("Memuat Dataset")
dataset_path = 'C:\\Users\\elisa\\OneDrive\\Documents\\Study Case GDGoc\\DataScience_salaries_2024.csv'
df = pd.read_csv(dataset_path)
st.write("**Informasi Dataset:**")
st.write(df.info())

st.write("**5 Baris Awal Dataset:**")
st.write(df.head())

st.markdown("""
### **Pada informasi dasar dataset ini, didapatkan sebagai berikut:**
1. Dataset memiliki atribut sebanyak **11 kolom**, dan entri sebanyak **14.838 baris data**.
2. Tidak ada nilai kosong pada dataset.
3. Atribut dataset terdiri dari **4 numerikal**, dan **7 kategorikal**.
""")

# Data Wrangling
st.header("2. Data Wrangling")
st.subheader("Mengecek Nilai Kosong dan Duplikat")

missing_values = df.isnull().sum()
st.write("Jumlah nilai kosong per kolom:", missing_values)

duplicate_rows = df.duplicated().sum()
st.write(f"Jumlah baris duplikat: {duplicate_rows}")

# Remove duplicates (if necessary)
df = df.drop_duplicates()
st.markdown("""
Terlihat memang tidak ada nilai kosong pada Dataset, sehingga tidak diperlukan penanganan lebih lanjut.
Namun, dataset memiliki **5711 duplikat**, yang wajar mengingat beberapa atribut seperti pekerjaan atau lokasi mungkin memiliki pola yang serupa.
""")

# Data Exploration
st.header("3. Data Exploration")
st.subheader("Deskripsi Statistik Dataset")

st.write(df.describe())
st.markdown("""
### Kesimpulan Deskriptif:
1. Rata-rata tahun kerja adalah **2023**, yang artinya sebagian besar data berasal dari tahun tersebut.
2. Rata-rata **salary_in_usd** adalah **149.874 USD**.
3. Kuartil pertama menunjukkan 25% dari individu mendapatkan gaji di bawah **102.100 USD**, sedangkan kuartil ketiga menunjukkan 25% mendapatkan gaji di atas **185.900 USD**, menandakan kesenjangan gaji.
4. Mayoritas pekerja bekerja secara **remote 100%**.
""")

# Visualizing Numerical Columns
st.header("4. Visualizing Numerical Columns")

# Salary Distribution
fig, ax = plt.subplots()
sns.histplot(df['salary_in_usd'], kde=True, color='blue', ax=ax)
ax.set_title('Distribution of Salary in USD')
ax.set_xlabel('Salary in USD')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Remote Ratio Distribution
fig, ax = plt.subplots()
sns.histplot(df['remote_ratio'], kde=True, color='green', ax=ax)
ax.set_title('Distribution of Remote Ratio')
ax.set_xlabel('Remote Ratio')
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.markdown("""
### **Insight dari Distribusi Data:**
#### **Distribusi Salary in USD:**
- Distribusi yang **skewed ke kanan**, menunjukkan sebagian besar data terkonsentrasi di gaji yang lebih rendah.
- Puncak distribusi berada pada angka sekitar **150.000 USD**.

#### **Distribusi Remote Ratio:**
- Distribusi juga **skewed ke kanan**, menunjukkan mayoritas pekerjaan dilakukan **on-site** (remote ratio 0%).
""")

# Correlation Heatmap
st.header("5. Correlation Heatmap")

# Calculate correlation
numeric_data = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix")
st.pyplot(fig)

st.markdown("""
Pada korelasi heatmap ini didapatkan bahwa **salary_in_usd** dan **remote_ratio** tidak memiliki korelasi yang signifikan.
""")

# Countplot for Categorical Variables
st.header("6. Visualisasi Variabel Kategorikal")

# Experience Level Count
fig, ax = plt.subplots()
sns.countplot(data=df, x='experience_level', palette='Set2', ax=ax)
ax.set_title('Experience Level Distribution')
st.pyplot(fig)

# Company Size Count
fig, ax = plt.subplots()
sns.countplot(data=df, x='company_size', palette='Set2', ax=ax)
ax.set_title('Company Size Distribution')
st.pyplot(fig)

# Employment Type Count
fig, ax = plt.subplots()
sns.countplot(data=df, x='employment_type', palette='Set2', ax=ax)
ax.set_title('Employment Type Distribution')
st.pyplot(fig)

st.markdown("""
### Insight:
1. Sebagian besar individu memiliki pengalaman di tingkat **Senior (SE)**.
2. Mayoritas berasal dari perusahaan **Medium (M)**.
3. Hampir semua individu bekerja **Full-Time (FT)**.
""")

# Boxplot: Salary vs Experience Level
st.header("7. Boxplot: Salary vs Experience Level")

fig, ax = plt.subplots()
sns.boxplot(data=df, x='experience_level', y='salary_in_usd', palette="Set3", ax=ax)
ax.set_title('Salary Distribution by Experience Level')
ax.set_xlabel("Experience Level")
ax.set_ylabel("Salary in USD")
st.pyplot(fig)

st.markdown("""
### Insight Boxplot:
1. **Semakin tinggi tingkat pengalaman, maka gajinya semakin tinggi.**
2. **Variasi gaji yang tinggi di setiap tingkat pengalaman.**
3. **Outlier pada boxplot** menunjukkan individu dengan gaji lebih besar dari tingkat pengalamannya.
""")

# Top Locations by Average Salary
st.header("8. Top Locations by Average Salary")

# Top 10 company locations by salary
top_locations = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
top_locations.plot(kind='bar', color='teal', ax=ax)
ax.set_title("Top 10 Company Locations by Average Salary")
ax.set_xlabel("Company Location")
ax.set_ylabel("Average Salary in USD")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

st.markdown("""
### Insight:
1. **Qatar (QA)** memiliki rata-rata gaji tertinggi, yaitu sekitar **300.000 USD**.
2. **Israel (IL)** dan **Puerto Rico (PR)** menyusul di bawahnya.
3. Lokasi memengaruhi gaji secara signifikan.
""")
