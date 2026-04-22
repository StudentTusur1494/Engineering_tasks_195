import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === ПРИМЕР 1: Базовые графики ===
x_ex = np.linspace(0, 10, 20)
y_ex = 2 * x_ex + 5 + np.random.normal(0, 2, 20)

plt.figure(figsize=(8, 4))
# Линия (plot)
plt.plot(x_ex, 2 * x_ex + 5, color='red', linestyle='--', label='Идеальная линия')
# Точки (scatter)
plt.scatter(x_ex, y_ex, color='blue', marker='o', label='Реальные данные')

plt.title('ПРИМЕР базового графика')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.legend()
plt.grid(True)
plt.show()




# === ПРИМЕР 2: Составные графики (Subplots) ===
errors_ex = np.random.normal(0, 1.5, 1000) # Имитация ошибок модели

# Создаем фигуру с 1 строкой и 2 колонками. axes - массив осей
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Левый график (axes[0])
axes[0].scatter(x_ex, y_ex, color='green')
axes[0].set_title('ПРИМЕР: График рассеяния (Scatter)')
axes[0].grid(True)

# Правый график (axes[1]) - Гистограмма
axes[1].hist(errors_ex, bins=20, color='orange', edgecolor='black')
axes[1].set_title('ПРИМЕР: Гистограмма ошибок (Histogram)')
axes[1].set_xlabel('Величина ошибки')

plt.tight_layout() # Чтобы графики не наезжали друг на друга
plt.show()




mu0 = 4.0 * np.pi * 1e-7
eps0 = 8.854187817e-12
Z0 = 376.73

def calc_shield_se(freq, table):
    """
    Расчёт эффективности экранирования (SE) методом матриц передачи.
    """
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

        T = 2 * Z0 / (A_total[1, 0] * Z0**2 + A_total[1, 1] * Z0 +
                       A_total[0, 0] * Z0 + A_total[0, 1])
        SE[i] = 20.0 * np.log10(np.abs(1.0 / T))

    return SE

# Фиксированные параметры по условию задачи
SIGMA = 5.8e7
EPS_R = 1.0
T_FIXED = 1e-4

# --- ГЕНЕРАЦИЯ ДЛЯ ГРАФИКОВ (Спектры) ---
freq_mid = np.linspace(1e+3, 1e+7, 1001) # 1 кГц - 10 МГц
mu_for_plots = [10, 30, 50, 70, 100]
se_spectra = {}
for m in mu_for_plots:
    layer = np.array([m, EPS_R, SIGMA, T_FIXED, 1])
    se_spectra[m] = calc_shield_se(freq_mid, layer)

# --- ГЕНЕРАЦИЯ ДЛЯ ML (Датасет на фиксированной частоте из среднего диапазона) ---
np.random.seed(42)
target_freq_mid = 100e6 # 100 МГц
mu_dataset = np.random.uniform(10, 100, 200)
se_dataset = []
for m in mu_dataset:
    layer = np.array([m, EPS_R, SIGMA, T_FIXED, 1])
    se_dataset.append(calc_shield_se(np.array([target_freq_mid]), layer)[0])

df_mid = pd.DataFrame({'mu_r': mu_dataset, 'SE': se_dataset})
print("Данные успешно сгенерированы!")
print(f"Доступны: словарь 'se_spectra' (для графиков) и датафрейм 'df_mid' (для ML).")




plt.figure(figsize=(10, 6))

# В цикле проходимся по ключам (mu) и значениям (массивам SE) из словаря
for mu_val, se_array in se_spectra.items():
    # [Постройте график (plt.plot). Ось X: freq_mid / 1e6 (перевод в МГц), Ось Y: se_array]
    # [Добавьте параметр label=f'$\mu_r$ = {mu_val}' и толщину линии linewidth=2]
    plt.plot(freq_mid/1e6, se_array, label=f'$\mu_r$ = {mu_val}', linewidth=2)
    # pass удалите эту строку после написания кода


# [Добавьте сетку]
plt.grid(True)

# [Добавьте заголовок графика с размером шрифта fontsize=14]
plt.title("ПРАКТИКА 1", fontsize=14)

# [Подпишите ось X как "Частота, МГц", а ось Y как "SE, дБ"]
plt.xlabel("Частота, МГц")
plt.ylabel("SE, дБ")

# [Отобразите легенду, указав title='Магн. проницаемость']
plt.legend(title="Магн. проницаемость")

plt.show()




# [1. Выделите признаки: X -'mu_r', y - столбцом 'SE']
X = df_mid[['mu_r']]
y = df_mid['SE']

# [2. Разделите на Train (80%) и Test (20%) с помощью train_test_split, random_state=42]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [3. Создайте модель LinearRegression() и обучите её на тренировочных данных (.fit)]
model_mid = LinearRegression()
model_mid.fit(X_train, y_train)

# [4. Сделайте предсказание на тестовой выборке X_test (.predict)]
y_pred = model_mid.predict(X_test)

# [5. Вычислите R2 score, сравнив y_test и y_pred]
r2_mid = r2_score(y_test, y_pred) # замените 0.0 на вашу функцию

print("╔════════════════════════════════════════╗")
print(f"║ Качество модели (Средние частоты)      ║")
print(f"║ R² = {r2_mid:.4f}                      ║")
print("╚════════════════════════════════════════╝")




# [Создайте subplots: 1 строка, 2 колонки. Размер figsize=(14, 5)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- ЛЕВЫЙ ГРАФИК (axes[0]): Реальные данные и Линия регрессии ---
# [Постройте scatter plot для X_test и y_test (цвет синий, alpha=0.6, label='Истинные значения')]
axes[0].scatter(X_test, y_test, color='blue', alpha=0.6, label="Истинные значения")

# Генерируем точки для отрисовки идеальной прямой линии модели
x_line = np.linspace(10, 100, 100).reshape(-1, 1)
y_line = model_mid.predict(x_line)

# [Отрисуйте линию регрессии (plot) по точкам x_line и y_line (цвет красный, linewidth=2, label='Прямая регрессии')]
axes[0].plot(x_line, y_line, color='red', linewidth=2, label="Прямая регрессия")

axes[0].set_title("ПРАКТИКА 3: Предсказание SE от $\mu_r$", fontsize=12)
axes[0].set_xlabel("Магнитная проницаемость, $\mu_r$")
axes[0].set_ylabel("Эффективность SE, дБ")
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

# --- ПРАВЫЙ ГРАФИК (axes[1]): Гистограмма ошибок ---
# [Рассчитайте массив ошибок (разница между y_test и y_pred)]
errors = y_test - y_pred

# [Постройте гистограмму (hist) для массива errors. Параметры: bins=15, color='purple', edgecolor='black', alpha=0.7]
axes[1].hist(errors, bins=15, color='purple', edgecolor='black', alpha=0.7)

axes[1].set_title("ПРАКТИКА 3: Гистограмма ошибок модели", fontsize=12)
axes[1].set_xlabel("Ошибка (Истина - Предсказание), дБ")
axes[1].set_ylabel("Количество совпадений")
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()




# --- ГЕНЕРАЦИЯ ВАРИАНТА ПРЕПОДАВАТЕЛЕМ ---
student_id = int(input("Введите ваш номер в списке группы (число): "))

np.random.seed(student_id)
is_low = np.random.choice([True, False])
variant_freq = 1e6 if is_low else 5e9 # 1 МГц (Низкий) или 5 ГГц (Высокий)
variant_name = "Низкий (1 МГц)" if is_low else "Высокий (5 ГГц)"

print(f"\n⚡ ВАШ ВАРИАНТ: {variant_name} диапазон")

mu_var = np.random.uniform(10, 100, 200)
se_var = []
for m in mu_var:
    layer = np.array([m, EPS_R, SIGMA, T_FIXED, 1])
    se_var.append(calc_shield_se(np.array([variant_freq]), layer)[0])

df_variant = pd.DataFrame({'mu_r': mu_var, 'SE': se_var})
print("Датасет 'df_variant' готов к работе!")




# === ВАШ КОД ДЛЯ ПРАКТИКИ 4 НИЖЕ (Мой вариант - Высокий 5 ГГц) диапазон ===

# 1. Выделите признаки X и y из df_variant, разделите на Train/Test (20% на тест)
X = df_variant[['mu_r']]
y = df_variant['SE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Обучите новую модель Линейной регрессии
model_var = LinearRegression()
model_var.fit(X_train, y_train)

# 3. Посчитайте R2 score и выведите его на экран (через print)
y_pred_var = model_var.predict(X_test)
r2_var = r2_score(y_test, y_pred_var)

print("╔════════════════════════════════════════╗")
print(f"║ Качество модели ({variant_name} диапазон)      ║")
print(f"║ R² = {r2_var:.4f}                      ║")
print("╚════════════════════════════════════════╝")

# 4. Постройте график: scatter (тестовые точки X_test, y_test) + plot (красная линия предсказаний модели)
# Не забудьте добавить легенду, подписи осей и заголовок!
plt.figure(figsize=(8, 6))

plt.scatter(X_test, y_test, color='blue', alpha=0.6, label="Предсказания модели")

x_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line = model_var.predict(x_line)

plt.plot(x_line, y_line, color='red', linewidth=2, label='Прямая регрессии')

plt.title(f'Линейная регрессия SE от μr ({variant_name} диапазон)', fontsize=14)
plt.xlabel('Магнитная проницаемость, μr')
plt.ylabel('Эффективность SE, дБ')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()