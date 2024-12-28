# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
# # 1. Data Wrangling

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
# # 2. Data Exploration

# %%
# Basic description of the dataset
print("\nDataset description:")
df.describe()

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
# # 4. Correlation Heatmap

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

# %%
# 6. Boxplot: Salary vs Experience Level
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='experience_level', y='salary_in_usd', hue='experience_level', palette="Set3", dodge=False)
plt.title('Salary Distribution by Experience Level')
plt.xlabel("Experience Level")
plt.ylabel("Salary in USD")
plt.show()

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


