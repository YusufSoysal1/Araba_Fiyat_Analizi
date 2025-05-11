import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataframe = pd.read_excel("C:/Users/yusuf/Downloads/merc.xlsx")
print(dataframe.describe())
print(dataframe.isnull().sum())
sbn.histplot(dataframe["price"], bins=50, kde=True)
plt.title("Orijinal Fiyat Dağılımı")
plt.show()
#en yüksek fiyatlı arabaları attım
newdf = dataframe.sort_values("price", ascending=False).iloc[130:]
sbn.histplot(newdf["price"], bins=50, kde=True)
plt.title("130 En Yüksek Fiyatlı Araç Çıkarıldıktan Sonra")
plt.show()

numeric_cols = newdf.select_dtypes(include=np.number)

grouped_df = numeric_cols.groupby(newdf["year"]).mean()
print(grouped_df["price"])

features = ["year", "mileage", "engineSize"]
target = "price"

newdf = newdf[features + [target]].dropna()

X = newdf[features]
y = newdf[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(12, input_dim=X_train_scaled.shape[1], activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), batch_size=250, epochs=300)

plt.plot(history.history['loss'], label='Eğitim Kaybı (Training Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Validation Loss)')
plt.title('Model Kayıpları')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"📊 MAE: {mae}")
print(f"📊 MSE: {mse}")
print(f"📊 RMSE: {rmse}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color="red")
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.title("Gerçek vs. Tahmin Edilen Fiyatlar")
plt.show()

yil = int(input("Arabanın model yılı: "))
km = int(input("Kaç km'de?: "))
motor = float(input("Motor hacmi nedir?: "))
yeni_veri = pd.DataFrame({
    "year": [yil],
    "mileage": [km],
    "engineSize": [motor]
})
yeni_veri_scaled = scaler.transform(yeni_veri)
tahmin = model.predict(yeni_veri_scaled)
print(f"🚘 Tahmin Edilen Fiyat: {tahmin[0][0]:,.2f} $")