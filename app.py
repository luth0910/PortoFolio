import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats.mstats import winsorize

def detect_and_winsorize(column):
    if pd.api.types.is_numeric_dtype(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = column[(column < lower_bound) | (column > upper_bound)]
        st.write(f'Outlier yang terdeteksi pada {column.name}: {outliers.count()} data')

        winsorized_column = winsorize(column, limits=[0.01, 0.01])
        return winsorized_column
    else:
        st.write(f'{column.name} bukan kolom numerik, tidak dilakukan winsorizing.')
        return column

df = pd.read_csv("World Happiness Report 2024.csv")

df2 = df.dropna().reset_index(drop=True)

st.title("Analisis World Happiness Report 2024 Menggunakan Linear Regresion")

st.subheader("Tampilan Data Awal")
st.write(df2.head(10))

st.write(f"Jumlah Baris: {df2.shape[0]}")
st.write(f"Jumlah Kolom: {df2.shape[1]}")

st.sidebar.title("Pengaturan Analisis")

numeric_columns = df2.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns = [col for col in numeric_columns if col != 'Life Ladder' and col != 'year']

pilihan_kolom = st.sidebar.selectbox(
    'Pilih kolom untuk membandingkan dengan Life Ladder:',
    numeric_columns
)

df2[pilihan_kolom] = detect_and_winsorize(df2[pilihan_kolom])

st.subheader("Distribusi Life Ladder")
fig, ax = plt.subplots()
sns.histplot(df2['Life Ladder'], kde=True, ax=ax)
ax.set_title('Distribusi Life Ladder')
ax.set_xlabel('Life Ladder')
ax.set_ylabel('Frekuensi')
st.pyplot(fig)

st.subheader(f'Distribusi Kolom: {pilihan_kolom}')
fig, ax = plt.subplots()
sns.boxplot(x=df2[pilihan_kolom], ax=ax)
ax.set_title(f'Distribusi {pilihan_kolom}')
ax.set_xlabel(pilihan_kolom)
st.pyplot(fig)

st.subheader(f'Hubungan antara {pilihan_kolom} dan Life Ladder')
fig, ax = plt.subplots()
sns.scatterplot(x=df2[pilihan_kolom], y=df2['Life Ladder'], ax=ax, hue=df2['Life Ladder'], palette='viridis')
ax.set_title(f'{pilihan_kolom} vs Life Ladder')
ax.set_xlabel(pilihan_kolom)
ax.set_ylabel('Life Ladder')
ax.legend(title='Life Ladder')
st.pyplot(fig)

X = df2[[pilihan_kolom]].values.reshape(-1, 1)
y = df2['Life Ladder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

y_pred_linear = model_linear.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

st.subheader("Hasil Evaluasi Model")
st.write(f'Model Linear Regression: MSE = {mse_linear}, R-squared = {r2_linear}')

st.subheader(f'Hasil Prediksi: {pilihan_kolom} vs Life Ladder')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test, y_test, color='blue', label='Nilai Sebenarnya')
ax.plot(X_test, y_pred_linear, color='red', label='Prediksi Linear', linewidth=2)
ax.set_title(f'{pilihan_kolom} vs Life Ladder (Test Set) - Linear Regression')
ax.set_xlabel(pilihan_kolom)
ax.set_ylabel('Life Ladder')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("Tabel Hasil Prediksi vs Nilai Sebenarnya (Linear Regression)")
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_linear})
st.write(results_df.head())

st.subheader("Deskripsi Statistik Hasil Prediksi")
st.write(results_df.describe())

df2 = df2.dropna(subset=['Life Ladder'])
aggregated_data = df2.groupby('Country name', as_index=False)['Life Ladder'].mean()
aggregated_data.reset_index(drop=True, inplace=True)

top_happiness_countries = aggregated_data.sort_values(by='Life Ladder', ascending=False).head(10)

st.subheader("10 Negara dengan Nilai Kebahagiaan Tertinggi")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_happiness_countries, x='Life Ladder', y='Country name', palette='viridis', ax=ax)
ax.set_title('10 Negara Teratas dengan Nilai Kebahagiaan Tertinggi', fontsize=16)
ax.set_xlabel('Life Ladder', fontsize=12)
ax.set_ylabel('Negara', fontsize=12)
ax.grid(axis='x')
st.pyplot(fig)

bottom_happiness_countries = aggregated_data.sort_values(by='Life Ladder', ascending=True).head(10)

st.subheader("10 Negara dengan Nilai Kebahagiaan Terendah")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=bottom_happiness_countries, x='Life Ladder', y='Country name', palette='coolwarm', ax=ax)
ax.set_title('10 Negara dengan Nilai Kebahagiaan Terendah Berdasarkan Life Ladder', fontsize=16)
ax.set_xlabel('Life Ladder', fontsize=12)
ax.set_ylabel('Negara', fontsize=12)
ax.grid(axis='x')
st.pyplot(fig)
