# %% [markdown]
# # **Case Study Machine Learning GDGoC UNSRI 2024/2025**

# %% [markdown]
# Nama    : Elisa Dwi Nanda<br>
# Jurusan : Sistem Komputer<br>
# Angkatan: 2023<br>

# %% [markdown]
# # **Dataset yang Digunakan**
# Dataset yang digunakan dalam case study ini adalah **'Data Science Salaries 2024'**, yang dapat diakses melalui tautan berikut: [Dataset Data Science Salaries 2024-Kaggle
# ](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)

# %% [markdown]
# ## **Deskripsi Dataset**
# Dataset ini berisi informasi tentang gaji para profesional di bidang data science untuk tahun 2024. Data mencakup berbagai atribut penting seperti:
# 
# - work_year: Tahun kerja.
# - experience_level: Tingkat pengalaman (Entry-Level, Mid-Level, Senior, Executive).
# - employment_type: Jenis pekerjaan (Full-Time, Part-Time, Contract, Freelance).
# - job_title: Pekerjaan.
# - salary: Gaji dalam mata uang lokal.
# - salary_currency: Mata uang gaji.
# - salary_in_usd: Gaji dalam USD.
# - employee_residence: Negara tempat tinggal karyawan.
# - remote_ratio: Persentase kerja jarak jauh (0, 50, atau 100).
# - company_location: Lokasi perusahaan.
# - company_size: Ukuran perusahaan berdasarkan jumlah karyawan (S: Small, M: Medium, L: Large).

# %% [markdown]
# ## Import Library yang Dibutuhkan

# %%
import pandas as pd
import numpy as np
import streamlit as st

# %% [markdown]
# ### **Dataset Information, 5 Baris Awal Dataset**

# %%
# Load Dataset
dataset_path = 'C:\\Users\\elisa\\OneDrive\\Documents\\Study Case GDGoc\\DataScience_salaries_2024.csv'
df = pd.read_csv(dataset_path)

# Display basic information
print("Dataset Info:")
df.info()
print("\nFirst 5 Rows:")
df.head()

# %% [markdown]
# ### **Pada informasi dasar dataset ini, didapatkan sebagai berikut:**
# 1. Dataset memiliki atribut sebanyak 11 kolom, dan entri sebanyak 14838 baris data.
# 2. Dari informasi dasar tersebut juga dapat diambil kesimpuan bahwa tidak ada nilai yang kosong dari dataset.
# 3. Atribut dataset terdiri dari 4 numerikal, dan 7 kategorikal

# %% [markdown]
# # **1. Data Wrangling**

# %%
# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Check for duplicates
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

# Remove duplicates (if necessary)
df = df.drop_duplicates()

# %% [markdown]
# ### **Mengecek Nilai Kosong**
# Terlihat memang benar tidak ada nilai kosong pada Dataset, sehingga tidak diperlukan penanganan lebih lanjut.

# %% [markdown]
# ### **Mengecek Nilai Duplikat**
# Jumlah duplikat: 5711<br>
# 
# Terlihat pada data set ini, memiliki duplikat yang sangat banyak. Namun hal itu wajar mengingat Dataset ini memiliki object dan numerikal yang memiliki persentase yang konsisten, seperti jenis perkerjaan, rasio jarak, dan pekerjaan yang sama, namun berbeda individu.

# %% [markdown]
# # 2. Data Exploration

# %%
# Basic description of the dataset
print("\nDataset description:")
df.describe()

# %% [markdown]
# Dari deskripsi data berikut, dapat disimpulkan beberapa hal:
# 
# 1. Rata-rata paa tahun kerja 2023..., yang artinnya sebagian besar tahun kerja berasal dari tahun 2023.
# 2. Rata-Rata Salary in usd adalah 149.874,
# 3. Kuartil pertama (25%) menunjukkan bahwa 25% dari individu mendapatkan gaji di bawah 102.100, sedangkan kuartil ketiga (75%) menunjukkan bahwa 25% dari individu mendapatkan gaji di atas 185.900. Menjelaskan jika ada kesenjangan yang signifikan pada gaji.
# 4. Pada kuartil 3 (75%) juga pada bagian remote, 100%, menunjukkan bahwa 25% individu yang gajinnya diatas 185.900 bekerja secara remote.

# %% [markdown]
# # 3. Visualizing Numerical Columns

# %%
# Salary distribution
plt.figure(figsize=(10,6))
sns.histplot(df['salary_in_usd'], kde=True, color='blue')
plt.title('Distribution of Salary in USD')
plt.xlabel('Salary in USD')
plt.ylabel('Frequency')
plt.show()

# Remote ratio distribution
plt.figure(figsize=(10,6))
sns.histplot(df['remote_ratio'], kde=True, color='green')
plt.title('Distribution of Remote Ratio')
plt.xlabel('Remote Ratio')
plt.ylabel('Frequency')
plt.show()

# %%
# Select only numeric columns
numeric_data = df.select_dtypes(include=[np.number])

# Calculate correlation
correlation_matrix = numeric_data.corr()

# %% [markdown]
# ### **Insight dari Distribusi Data**
# #### **Distribusi Salary in USD**
# - Distribusi yang **skewed ke kanan**, atau **skewed positive**, dapat terlihat ekor yang mengarah ke arah kanan. Hal ini berarti sebagian besar data terkonsentrasi di sebelah kiri.
# - Puncak distribusi berada pada angka sekitar **150.000 USD**, yang artinya banyak data yang berkumpul di sekitar kisaran gaji tersebut.
# 
# #### **Distribusi Remote Ratio**
# - Distribusi juga **skewed ke kanan**, yang artinya sebagian besar data terkonsentrasi di sebelah kiri.
# - Puncak distribusi berada di nilai **0**, yang artinya mayoritas pekerja melakukan pekerjaannya secara **on site** atau **non-remote.**

# %% [markdown]
# # **4. Correlation Heatmap**

# %%
# Visualize correlation matrix
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

correlation_matrix = df[['salary_in_usd', 'remote_ratio']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# Pada korelasi heatmap ini didapatkan bahwa salary in usd dan remote ratio tidak memiliki korelasi atau hubungan yang kuat.

# %%
# 5. Countplot for Categorical Variables

# Experience Level Count
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='experience_level', hue='experience_level', palette='Set2', legend=False)
plt.title('Experience Level Distribution')
plt.show()

# Company Size Count
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='company_size', hue='company_size', palette='Set2', legend=False)
plt.title('Company Size Distribution')
plt.show()

# Job Title Count
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='employment_type', hue='employment_type', palette='Set2', legend=False)
plt.title('Employment Type Distribution')
plt.show()

# %% [markdown]
# ### **Countplot Pengalaman**
# 1. Sebagian besar individu memiliki pengalaman di tingkat SE (Senior), diikuti oleh **MI (Mid)**, **EN (Entry)**, dan **EX (Expert).**
# ### **Countplot Tipe Perusahaan**
# 2. Sebagian besar individu berasal dari perusahaan **M (Medium)**, diikuti oleh perusahaan **L (Large)**, sedangkan perusahaan **S (Small)** memiliki jumlah individu yang jauh lebih sedikit. Hal ini menunjukkan bahwa sebagian besar individu bekerja di perusahaan dengan skala sedang (medium).
# ### **Countplot Tipe Pekerjaan**
# 3. Mayoritas atau hampir semua individu bekerja Fulltime (FT).

# %%
# 6. Boxplot: Salary vs Experience Level
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='experience_level', y='salary_in_usd', hue='experience_level', palette="Set3", dodge=False)
plt.title('Salary Distribution by Experience Level')
plt.xlabel("Experience Level")
plt.ylabel("Salary in USD")
plt.show()

# %% [markdown]
# ### Berdasarkan Boxplot tersebut, didapatkan sebagai berikut:
# 1. **Semakin tinggi tingkat pengalaman, maka gajinya semakin tinggi.**
# Hal ini ditunjukkan oleh **median gaji yang lebih tinggi** dibandingkan tingkat pengalaman lainnya.
# 
# 2. **Variasi gaji yang tinggi di setiap tingkat pengalaman.**
# Dapat dilihat dari rentang antar kuartil (**IQR**) yang cukup lebar, menunjukkan adanya penyebaran gaji yang signifikan.
# 
# 3. **Outlier pada boxplot.**
# Menunjukkan adanya individu yang memiliki gaji lebih besar dari tingkat pengalamannya.
# Hal ini dapat disebabkan oleh faktor-faktor seperti kemampuan khusus atau kebijakan perusahaan tertentu.

# %%
# Visualize the top locations with highest average salary
# Top 10 company locations by salary
top_locations = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).head(10)

# Bar plot for top locations
plt.figure(figsize=(10, 6))
top_locations.plot(kind='bar', color='teal')
plt.title("Top 10 Company Locations by Average Salary")
plt.xlabel("Company Location")
plt.ylabel("Average Salary in USD")
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ### **Berdasarkan visualisasi tersebut dapat diambil insight sebagai berikut:**
# 1. **Qatar (QA)** memiliki rata-rata gaji tertinggi, yaitu sekitar **300.000 USD.**
# 
# 2. **Israel (IL)** dan **Puerto Rico (PR)** memiliki rata-rata gaji yang lebih rendah daripada **Qatar (QA)**, tetapi masih lebih tinggi daripada **Amerika Serikat (US)** dan **Selandia Baru (NZ).**
# 
# 3. **Amerika Serikat (US)** dan **Selandia Baru (NZ)** memiliki rata-rata gaji yang lebih rendah dibandingkan dengan **Qatar (QA)**, **Israel (IL)**, dan **Puerto Rico (PR).**
# 
# ### **Insight**
# 1. Hampir semua individu adalah pekerja Full Time (FT).
# 2. Pengalaman memiliki pengaruh terhadap gaji, semakin tinggi pengalaman, semakin tinggi pula gaji. Namun ada beberapa faktor lain yang dapat mempengaruhi seperti lokasi, serta perusahaan.
# 3. Kebanyakan individu memiliki pengalaman sebagai Senior.
# 4. Rata-rata gaji adalah 149.874 USD.
# 5. Mayoritas pekerja bekerja secara on-site dengan rasio remote 0% yang menunjukkan bahwa pekerjaan ini lebih cenderung dilakukan di kantor.
# 6. Lokasi dengan gaji rata-rata tertinggi adalah Qatar, diikuti oleh Israel dan Puerto Rico.


