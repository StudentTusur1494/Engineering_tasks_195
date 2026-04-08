import pandas as pd
import numpy as np

# === ПРИМЕР 1: Очистка типов данных и удаление NaN ===

# Создаем пример "грязного" датафрейма
df_example = pd.DataFrame({
    'ID_измерения': [1, 2, 3, 4, 5],
    'Проводимость': [50.5, 'ошибка', 42.1, np.nan, 'N/A']
})

print("=== ДО ОЧИСТКИ ===")
print(df_example)
print("\nТипы данных:")
print(df_example.dtypes)

# 1. Принудительно переводим столбец в числа.
# Параметр errors='coerce' превратит весь нечитаемый текст в NaN (Not a Number)
df_example['Проводимость'] = pd.to_numeric(df_example['Проводимость'], errors='coerce')

# 2. Удаляем все строки, в которых есть NaN
df_clean_example = df_example.dropna().reset_index(drop=True)

print("\n=== ПОСЛЕ ОЧИСТКИ ===")
print(df_clean_example)
print("\nТипы данных:")
print(df_clean_example.dtypes)




# === ПРИМЕР 2: Булевы маски и фильтрация ===

df_mask_example = pd.DataFrame({
    'Материал': ['Медь', 'Алюминий', 'Сталь', 'Никель', 'Свинец'],
    'SE_дБ': [65, 55, 120, 80, 15] # 120 и 15 - явные аномалии для нашего примера
})

# Задача: оставить только те экраны, где SE находится в пределах от 40 до 100 дБ

# 1. Создаем маску (условие)
# ВАЖНО: Каждое условие берется в скобки (...), между ними ставится амперсанд & (логическое И)
mask = (df_mask_example['SE_дБ'] >= 40) & (df_mask_example['SE_дБ'] <= 100)

print("=== Созданная булева маска ===")
print(mask)

# 2. Применяем маску к датафрейму
df_filtered_example = df_mask_example[mask].reset_index(drop=True)

print("\n=== Отфильтрованный датафрейм ===")
print(df_filtered_example)




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# === ПРИМЕР 3: Пайплайн машинного обучения ===

# Генерируем случайные тренировочные данные: X (признаки) и y (целевая переменная)
np.random.seed(42)
X_dummy = pd.DataFrame({
    'Признак_1': np.random.uniform(1, 10, 100),
    'Признак_2': np.random.uniform(20, 50, 100)
})
# Допустим, y зависит от признаков по формуле: y = 2*X1 + 0.5*X2 + шум
y_dummy = 2 * X_dummy['Признак_1'] + 0.5 * X_dummy['Признак_2'] + np.random.normal(0, 1, 100)

# 1. Разделение выборки (20% на тест, 80% на обучение)
X_train_ex, X_test_ex, y_train_ex, y_test_ex = train_test_split(
    X_dummy, y_dummy, test_size=0.2, random_state=42
)

print(f"Всего данных: {len(X_dummy)} строк")
print(f"Обучающая выборка (Train): {len(X_train_ex)} строк")
print(f"Тестовая выборка (Test): {len(X_test_ex)} строк\n")

# 2. Создание и обучение модели
model_ex = LinearRegression()
model_ex.fit(X_train_ex, y_train_ex) # Учится ТОЛЬКО на X_train и y_train!

# 3. Предсказание на новых данных
y_pred_ex = model_ex.predict(X_test_ex)

print("Первые 5 предсказаний модели:")
print(np.round(y_pred_ex[:5], 2))
print("Первые 5 истинных значений:")
print(np.round(y_test_ex.values[:5], 2))




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === ПРИМЕР 4: Вычисление метрик ===

# Сравниваем истинные значения (y_test) и предсказанные (y_pred)
mae_ex = mean_absolute_error(y_test_ex, y_pred_ex)
mse_ex = mean_squared_error(y_test_ex, y_pred_ex)
r2_ex = r2_score(y_test_ex, y_pred_ex)

print("=== Оценка качества модели ===")
print(f"MAE (Средняя абсолютная ошибка): {mae_ex:.2f}")
print(f"MSE (Среднеквадратичная ошибка): {mse_ex:.2f}")
print(f"R² (Коэффициент детерминации)  : {r2_ex:.4f} (идеал = 1.0)")

# Вывод весов (влияние каждого признака)
print("\n=== Веса Линейной регрессии ===")
for col, coef in zip(X_dummy.columns, model_ex.coef_):
    print(f"{col}: {coef:.2f}")




import matplotlib.pyplot as plt

# === ПРИМЕР 5: График "Истина vs Предсказание" ===

plt.figure(figsize=(8, 6))

# Строим точки предсказаний (Истина по X, Предсказание по Y)
plt.scatter(y_test_ex, y_pred_ex, color='blue', alpha=0.6, label='Предсказания модели')

# Строим идеальную линию y = x (красный пунктир)
# Если модель идеальна, все синие точки лягут ровно на эту красную линию
min_val = y_test_ex.min()
max_val = y_test_ex.max()
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Идеальное совпадение')

# Оформление графика
plt.xlabel('Истинные значения (y_test)')
plt.ylabel('Предсказанные значения (y_pred)')
plt.title(f'Качество предсказаний (R² = {r2_ex:.2f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()





# Физические константы
mu0 = 4.0 * np.pi * 1e-7
eps0 = 8.854187817e-12
Z0 = 376.73


def calc_shield_se(freq, table):
    """Метод матриц передачи."""
    if table.ndim == 1:
        table = table[np.newaxis, :]
    n_layers = table.shape[0]
    SE = np.zeros(len(freq))
    for i in range(len(freq)):
        w = 2.0 * np.pi * freq[i]
        A_total = np.eye(2, dtype=complex)
        for v in range(n_layers):
            mu_r, eps_r, sigma_v, t_v, mat_type = table[v]
            Ma = mu_r * mu0
            Ea = eps_r * eps0
            if int(mat_type) == 1:
                z = np.sqrt((1j * w * Ma) / (sigma_v + 1j * w * Ea))
                g = np.sqrt((1j * w * Ma) * (sigma_v + 1j * w * Ea))
            else:
                sig_comp = w * eps0 * np.imag(eps_r)
                z = (1 + 1j) * np.sqrt(w * Ma / (sig_comp + 1e-30))
                g = 1j * np.sqrt(w * Ma * (sig_comp + 1j * w * eps0 * np.real(eps_r)))
            A_layer = np.array([
                [np.cosh(g * t_v), z * np.sinh(g * t_v)],
                [np.sinh(g * t_v) / z, np.cosh(g * t_v)]
            ], dtype=complex)
            A_total = A_total @ A_layer
        T = 2 * Z0 / (A_total[1, 0] * Z0 ** 2 + A_total[1, 1] * Z0 + A_total[0, 0] * Z0 + A_total[0, 1])
        SE[i] = 20.0 * np.log10(np.abs(1.0 / T))
    return SE


# === ГЕНЕРАЦИЯ ДАННЫХ ===
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
N = 3000

# Варьируем 3 параметра: Частота, mu_r, Проводимость
freq_hz = 10 ** np.random.uniform(6, 9, N)  # 1 МГц - 1 ГГц
freq_ghz = freq_hz / 1e9
mu_r = np.random.uniform(1, 100, N)
sigma_msm = np.random.uniform(1, 100, N)

# Константы (металл, 1 мкм)
t = np.full(N, 1e-6)
eps_r = np.ones(N)

print("Рассчитываем аналитическую модель (SE)...")
se_values = np.zeros(N)
for idx in range(N):
    # Подаем СИ (См/м) в функцию
    layer = np.array([mu_r[idx], eps_r[idx], sigma_msm[idx] * 1e6, t[idx], 1])
    se_values[idx] = calc_shield_se(np.array([freq_hz[idx]]), layer)[0]

df_ideal = pd.DataFrame({
    'freq_GHz': freq_ghz,
    'mu_r': mu_r,
    'sigma_MSm': sigma_msm,
    'SE': se_values
})

# === ИСКУССТВЕННОЕ ЗАГРЯЗНЕНИЕ ===
df_dirty = df_ideal.copy()
df_dirty['sigma_MSm'] = df_dirty['sigma_MSm'].astype(object)
rng = np.random.default_rng(42)

# 1. Пропуски (NaN) в частоте
df_dirty.loc[rng.choice(N, 60, replace=False), 'freq_GHz'] = np.nan
# 2. Текстовый мусор в проводимости
df_dirty.loc[rng.choice(N, 50, replace=False), 'sigma_MSm'] = rng.choice(['ошибка_чтения', 'N/A'])
# 3. Выбросы (Z-score) в mu_r: невозможные значения для наших данных
outlier_mu = rng.choice(N, 40, replace=False)
df_dirty.loc[outlier_mu, 'mu_r'] = rng.uniform(8000, 15000, 40)
# 4. Выбросы (IQR) в SE: сбой итогового расчёта
outlier_se = rng.choice(N, 30, replace=False)
df_dirty.loc[outlier_se, 'SE'] = df_dirty.loc[outlier_se, 'SE'] + 80

df = df_dirty.sample(frac=1, random_state=42).reset_index(drop=True)
print("Сырой датасет успешно сгенерирован в переменную `df`!")




# [Выведите количество пропусков (NaN) в каждом столбце df]
nulls_before = df.isnull().sum()
print("Пропуски до очистки:\n", nulls_before)

# [Преобразуйте столбец 'sigma_MSm' в числовой формат (float). Весь нечитаемый текст должен стать NaN]
df['sigma_MSm'] = pd.to_numeric(df['sigma_MSm'], errors='coerce')

# [Удалите все строки с пропусками (NaN) из датафрейма df и сохраните в df_dropped]
df_dropped = df.dropna().reset_index(drop=True)

# [Выведите размер датафрейма после удаления пустых строк]
print(f"Размер после удаления NaN: {df_dropped.shape}")




# --- 1. Очистка 'SE' методом IQR ---

# [Рассчитайте первый (Q1, 0.25) и третий (Q3, 0.75) квартили для столбца 'SE']
q1 = df_dropped['SE'].quantile(0.25)
q3 = df_dropped['SE'].quantile(0.75)

# [Рассчитайте межквартильный размах IQR]
iqr = q3 - q1

# [Отфильтруйте df_dropped: оставьте строки, где 'SE' находится строго в границах [Q1 - 1.5*IQR; Q3 + 1.5*IQR]. Сохраните в df_iqr]
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df_iqr = df_dropped[(df_dropped['SE'] >= lower_bound) & (df_dropped['SE'] <= upper_bound)].reset_index(drop=True)


# --- 2. Очистка 'mu_r' методом Z-score ---

# [Рассчитайте модуль Z-оценки для столбца 'mu_r' в датафрейме df_iqr]
z_scores_mu = np.abs((df_iqr['mu_r'] - df_iqr['mu_r'].mean()) / df_iqr['mu_r'].std())

# [Отфильтруйте df_iqr: оставьте только те строки, где Z-оценка < 3. Сохраните результат в итоговый датасет df_clean]
df_clean = df_iqr[z_scores_mu < 3].reset_index(drop=True)

print(f"Размер идеально чистого датасета: {df_clean.shape}")




import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# [Выделите матрицу признаков X ('freq_GHz', 'mu_r', 'sigma_MSm') и целевую переменную y ('SE') из df_clean]
X = df_clean[['freq_GHz', 'mu_r', 'sigma_MSm', 'SE']]
y = df_clean['SE']

# [Разделите выборку на обучающую (Train) и тестовую (Test). Тест = 20%, random_state=42]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === МОДЕЛЬ 1: ЛИНЕЙНАЯ РЕГРЕССИЯ ===
# [Создайте и обучите LinearRegression на обучающих данных]
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# [Сделайте предсказание y_pred_lr на тестовых данных]
y_pred_lr = lr_model.predict(X_test)

# [Рассчитайте MAE, MSE и R2 для Линейной регрессии]
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)


# === МОДЕЛЬ 2: СЛУЧАЙНЫЙ ЛЕС (Random Forest) ===
# [Создайте и обучите RandomForestRegressor (укажите random_state=42) на обучающих данных]
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# [Сделайте предсказание y_pred_rf на тестовых данных]
y_pred_rf = rf_model.predict(X_test)

# [Рассчитайте MAE, MSE и R2 для Случайного леса]
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# --- Вывод результатов ---
print("=== СРАВНЕНИЕ КАЧЕСТВА МОДЕЛЕЙ ===")
print(f"Линейная Регрессия: MAE = {mae_lr:.2f} дБ, MSE = {mse_lr:.2f}, R² = {r2_lr:.4f}")
print(f"Случайный Лес     : MAE = {mae_rf:.2f} дБ, MSE = {mse_rf:.2f}, R² = {r2_rf:.4f}")




# [Создайте фигуру с двумя графиками (subplots 1x2) размером 14x6]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- График 1: Линейная регрессия ---
# [Постройте scatter plot истинных (y_test) vs предсказанных (y_pred_lr) значений. Цвет: синий, alpha=0.5]
axes[0].scatter(y_test, y_pred_lr, color='blue', alpha=0.5)

# Добавляем идеальную линию y=x
min_val = min(y_test.min(), y_pred_lr.min())
max_val = max(y_test.max(), y_pred_lr.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное совпадение')

# Оформление первого графика
axes[0].set_xlabel('Истинные значения SE (дБ)')
axes[0].set_ylabel('Предсказанные значения SE (дБ)')
axes[0].set_title(f'Линейная регрессия (R² = {r2_lr:.3f})')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].axis('equal')  # Делаем оси одинакового масштаба

# --- График 2: Случайный лес ---
# [Постройте scatter plot истинных (y_test) vs предсказанных (y_pred_rf) значений. Цвет: зелёный, alpha=0.5]
axes[1].scatter(y_test, y_pred_rf, color='green', alpha=0.5)

# Добавляем идеальную линию y=x
min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное совпадение')

# Оформление второго графика
axes[1].set_xlabel('Истинные значения SE (дБ)')
axes[1].set_ylabel('Предсказанные значения SE (дБ)')
axes[1].set_title(f'Случайный лес (R² = {r2_rf:.3f})')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].axis('equal')  # Делаем оси одинакового масштаба

plt.tight_layout()
plt.show()




# --- Генерация сложных 4D данных (Блок преподавателя) ---
np.random.seed(99)
N_complex = 2000
freq_comp = 10 ** np.random.uniform(6, 9, N_complex)
mu_comp = np.random.uniform(1, 100, N_complex)
sigma_comp = np.random.uniform(1, 100, N_complex)
thick_comp = np.random.uniform(1e-6, 50e-6, N_complex) # от 1 мкм до 50 мкм
se_comp = np.zeros(N_complex)

print("Генерация сложных данных с толщиной. Подождите...")
for i in range(N_complex):
    layer = np.array([mu_comp[i], 1.0, sigma_comp[i] * 1e6, thick_comp[i], 1])
    se_comp[i] = calc_shield_se(np.array([freq_comp[i]]), layer)[0]

df_complex = pd.DataFrame({
    'freq_GHz': freq_comp / 1e9,
    'mu_r': mu_comp,
    'sigma_MSm': sigma_comp,
    'thickness_um': thick_comp * 1e6,
    'SE': se_comp
})
print("Готово!")

# === ВАШ КОД НИЖЕ ===

# [Выделите X (4 признака) и y (SE) из df_complex]
X_comp = df_complex[['freq_GHz', 'mu_r', 'sigma_MSm', 'thickness_um']]
y_comp = df_complex['SE']

# [Разделите данные на Train и Test (test_size=0.2)]
X_tr, X_te, y_tr, y_te = train_test_split(X_comp, y_comp, test_size=0.2, random_state=42)

# [Обучите RandomForestRegressor]
rf_complex = RandomForestRegressor(random_state=42, n_estimators=100)
rf_complex.fit(X_tr, y_tr)

# [Вычислите и выведите R2 на тестовой выборке]
pred_comp = rf_complex.predict(X_te)
r2_comp = r2_score(y_te, pred_comp)
print(f"R² сложной модели (4 признака): {r2_comp:.4f}")

# [Извлеките важность признаков с помощью rf_complex.feature_importances_]
importances = rf_complex.feature_importances_
feature_names = ['freq_GHz', 'mu_r', 'sigma_MSm', 'thickness_um']

# [Выведите важность каждого признака в процентах]
print("\n=== ВАЖНОСТЬ ПРИЗНАКОВ (Влияние на SE) ===")
for name, imp in zip(feature_names, importances):
    print(f"{name:15s}: {imp*100:.1f}%")

# Дополнительная визуализация для наглядности
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances * 100, color=['blue', 'red', 'green', 'orange'])
plt.xlabel('Физические параметры')
plt.ylabel('Важность (%)')
plt.title('Анализ важности признаков для эффективности экранирования (SE)')
plt.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for i, v in enumerate(importances * 100):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()