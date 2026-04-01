import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt

# Для воспроизводимости случайных чисел (бонусное задание)
import random

# Настройка графиков для красивого отображения
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
mu0 = 4.0 * np.pi * 1e-7     # магнитная постоянная, Гн/м
eps0 = 8.854187817e-12        # электрическая постоянная, Ф/м
Z0 = 376.73                   # импеданс свободного пространства, Ом

print("Библиотеки импортированы успешно!")
print(f"μ₀ = {mu0:.4e} Гн/м")
print(f"ε₀ = {eps0:.4e} Ф/м")
print(f"Z₀ = {Z0:.2f} Ом")




def calc_shield_se(freq, table):
    """
    freq  — 1D массив частот, Гц
    table — 2D массив (N_слоёв, 5): [mu_r, eps_r, sigma, t, type]
            type: 1 — металл, 2 — композит
    Возвращает: SE — 1D массив, дБ
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




# === ЗАДАНИЕ 1: Формирование массивов экранов ===

# --- Шаг 1: Массив частот ---
# создайте массив частот от 10 кГц (1e4) до 1 ГГц (1e9), 100 точек
freq = np.linspace(1e4, 1e9, 100)

print(f"Массив частот: {len(freq)} точек")
print(f"  от {freq[0]:.0f} Гц до {freq[-1]:.2e} Гц")

# --- Шаг 2: Параметры экранов ---
# Толщина для всех — 1 мм = 0.001 м
t = 0.001  # м

# создайте список названий экранов
names = ["Медь", "Алюминий", "Сталь", "Никель", "Олово", "Латунь"]

# создайте 6 массивов table для каждого металла
# Формат: np.array([mu_r, eps_r, sigma, толщина, тип])
table_cu = np.array([1.0, 1.0, 5.8e7, 1e-3, 1])
table_al = np.array([1.0, 1.0, 3.5e7, 1e-3, 1])
table_steel = np.array([100, 1.0, 1.0e7, 1e-3, 1])
table_ni = np.array([100, 1.0, 1.45e7, 1e-3, 1])
table_sn = np.array([1.0, 1.0, 8.7e7, 1e-3, 1])
table_brass = np.array([1.0, 1.0, 1.5e7, 1e-3, 1])

# ✏️  соберите все таблицы в список
tables = [table_cu, table_al, table_steel, table_ni, table_sn, table_brass]

# --- Шаг 3: Расчёт SE для каждого экрана ---
# создайте пустой 2D массив se_all размером (6, 100) и заполните его в цикле
se_all = np.zeros((6, 100))

for ind, item in enumerate(tables):
    se_all[ind, :] = calc_shield_se(freq, item)


print(f"\nРазмер массива se_all: {se_all.shape}")

# --- Шаг 4: Булевы маски ---
gost_level_1 = 50  # дБ

print(f"\n=== Анализ: SE ≥ {gost_level_1} дБ ===")

for idx, name in enumerate(names):
    # создайте булеву маску для проверки SE ≥ gost_level_1
    mask = se_all[idx, :] >= gost_level_1
    # подсчитайте количество True в маске
    n_pass = np.sum(mask)
    print(f"  {name:12s}: {n_pass:3d} из {len(freq)} частот прошли порог")

# --- Шаг 5: Лучший экран по среднему SE ---
# вычислите средний SE для каждого экрана (по оси 1) и найдите индекс максимума
se_mean = np.mean(se_all, axis=1)
best_idx = np.argmax(se_mean)

print(f"\n=== Средний SE по всем частотам ===")
for i, name in enumerate(names):
    print(f"  {name:12s}: {se_mean[i]:.2f} дБ")




# === ЗАДАНИЕ 2: Варьирование толщины ===

# создайте массив толщин в мм
thicknesses_mm = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
# переведите в метры
thicknesses_m = thicknesses_mm * 1e-3

# Параметры меди (без толщины и типа — добавим в цикле)
sigma_cu = 5.8e7
mu_r_cu = 1.0
eps_r_cu = 1.0

# Массив для результатов
se_thickness = np.zeros((len(thicknesses_m), len(freq)))

# ✏️ заполните se_thickness — для каждой толщины создайте table и вызовите calc_shield_se
for i, t in enumerate(thicknesses_m):
    table_cu = np.array([mu_r_cu, eps_r_cu, sigma_cu, t, 1])
    se_thickness[i, :] = calc_shield_se(freq, table_cu)

# --- График ---
plt.figure(figsize=(12, 7))

# постройте кривые SE(f) для каждой толщины с подписями в легенде
for i, t_mm in enumerate(thicknesses_mm):
    plt.loglog(freq, se_thickness[i, :], linewidth=2,
               label=f'Медь {t_mm} мм')

# добавьте горизонтальную линию ГОСТ = 60 дБ
gost_level_2 = 60
plt.axhline(y=gost_level_2, color='r', linestyle='--', linewidth=2,
            label=f'Требование: {gost_level_2} дБ')

plt.xlabel('Частота, Гц')
plt.ylabel('Эффективность экранирования SE, дБ')
plt.title('Влияние толщины медного экрана на эффективность экранирования')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# --- Анализ с помощью векторизации (БЕЗ циклов по элементам!) ---
print(f"\n=== Анализ по толщинам (ГОСТ = {gost_level_2} дБ) ===")

# ✏️ найдите минимальное SE для каждой толщины по оси 1
se_min_per_thickness = np.min(se_thickness, axis=1)

# посчитайте процент частот, где SE ≥ ГОСТ для каждой толщины
# Подсказка: создайте булеву маску для всей матрицы, затем np.sum по оси 1, делите на len(freq)*100
mask_gost = se_thickness >= gost_level_2
pct_pass = np.sum(mask_gost, axis=1) / len(freq) * 100

# === ЗАДАНИЕ 3: Сводная таблица ===

# Параметры
metals_info = {
    'Медь': {'mu_r': 1.0, 'eps_r': 1.0, 'sigma': 5.8e7},
    'Алюминий': {'mu_r': 1.0, 'eps_r': 1.0, 'sigma': 3.5e7},
    'Сталь': {'mu_r': 100.0, 'eps_r': 1.0, 'sigma': 1.0e7},
    'Никель': {'mu_r': 100.0, 'eps_r': 1.0, 'sigma': 1.45e7},
    'Олово': {'mu_r': 1.0, 'eps_r': 1.0, 'sigma': 8.7e6},
    'Латунь': {'mu_r': 1.0, 'eps_r': 1.0, 'sigma': 1.5e7},
}

# задайте массив толщин (мм) и порог ГОСТ
thicknesses_task3 = [0.3, 0.5, 1.0, 2.0]
gost_level_3 = 50  # дБ

# ✏️ Создайте пустые списки и заполните их в двойном цикле
# Для каждого металла и каждой толщины: рассчитайте SE, извлеките min/max/mean
results = {
    'Металл': [],
    'Толщина_мм': [],
    'SE_min_дБ': [],
    'SE_max_дБ': [],
    'SE_mean_дБ': [],
    'ГОСТ': []
}

for metal_name, metal_props in metals_info.items():
    for thickness_mm in thicknesses_task3:
        thickness_m = thickness_mm / 1000

        # Создаем таблицу слоя
        table = np.array([metal_props['mu_r'], metal_props['eps_r'],
                          metal_props['sigma'], thickness_m, 1])

        # Рассчитываем SE
        se_values = calc_shield_se(freq, table)

        # Вычисляем статистики
        se_min = np.min(se_values)
        se_max = np.max(se_values)
        se_mean = np.mean(se_values)
        gost_pass = 'Да' if se_min >= gost_level_3 else 'Нет'

        # Заполняем результаты
        results['Металл'].append(metal_name)
        results['Толщина_мм'].append(thickness_mm)
        results['SE_min_дБ'].append(round(se_min, 2))
        results['SE_max_дБ'].append(round(se_max, 2))
        results['SE_mean_дБ'].append(round(se_mean, 2))
        results['ГОСТ'].append(gost_pass)

# Создаем DataFrame с псевдонимом pnd
df = pnd.DataFrame(results)

# --- Вывод полной таблицы ---
print("=== Полная таблица результатов ===")
print(df.to_string(index=False))
print(f"\nПроверка колонок: {df.columns.tolist()}")




# === ЗАДАНИЕ 3.2: Фильтрация и сортировка DataFrame ===

# ТОП-5 лучших экранов по SE_min
# отсортируйте df по столбцу 'SE_min_дБ' по убыванию и возьмите первые 5 строк
top5 = df.nlargest(5, 'SE_min_дБ')

print("\n=== ТОП-5 лучших экранов (по минимальному SE) ===")
print(top5[['Металл', 'Толщина_мм', 'SE_min_дБ', 'SE_mean_дБ', 'ГОСТ']].to_string(index=False))

# Экраны, НЕ прошедшие ГОСТ
df_fail = df[df['ГОСТ'] == 'Нет']
print("\n=== Экраны, НЕ прошедшие ГОСТ (SE_min < 50 дБ) ===")
print(df_fail[['Металл', 'Толщина_мм', 'SE_min_дБ']].to_string(index=False))

# Экраны, прошедшие ГОСТ
# отфильтруйте строки, где столбец 'ГОСТ' равен 'Да'
df_pass = df[df['ГОСТ'] == 'Да']
print("\n=== Экраны, прошедшие ГОСТ (SE_min >= 50 дБ) ===")
print(df_pass[['Металл', 'Толщина_мм', 'SE_min_дБ']].to_string(index=False))




# === ЗАДАНИЕ 3.3: Столбчатая диаграмма прошедших/не прошедших ГОСТ ===

# подсчитайте количество 'Да' и 'Нет' в столбце 'ГОСТ'
n_pass_gost = len(df[df['ГОСТ'] == 'Да'])
n_fail_gost = len(df[df['ГОСТ'] == 'Нет'])

# постройте столбчатую диаграмму с подписями значений
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Левый график: общий результат ГОСТ ---
categories = ['Прошли ГОСТ', 'Не прошли ГОСТ']
values = [n_pass_gost, n_fail_gost]
colors = ['#2ecc71', '#e74c3c']

bars1 = axes[0].bar(categories, values, color=colors, edgecolor='black', linewidth=1)
axes[0].set_title('Общее количество экранов по результатам ГОСТ', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Количество конфигураций', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Добавляем подписи значений
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)

# --- Правый график: по металлам (сколько толщин прошло ГОСТ для каждого металла) ---
# для каждого металла подсчитайте количество прошедших ГОСТ конфигураций
gost_by_metal = df[df['ГОСТ'] == 'Да'].groupby('Металл').size()
all_metals = df['Металл'].unique()
gost_counts = [gost_by_metal.get(metal, 0) for metal in all_metals]

bars2 = axes[1].bar(all_metals, gost_counts, color='#3498db', edgecolor='black', linewidth=1)
axes[1].set_title('Количество толщин, прошедших ГОСТ (по металлам)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Количество прошедших конфигураций', fontsize=12)
axes[1].set_xlabel('Металл', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

# Добавляем подписи значений
for bar in bars2:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('task3_gost_analysis.png', dpi=150)
plt.show()

print(f"\nСтатистика ГОСТ:")
print(f"  Прошли ГОСТ: {n_pass_gost} конфигураций")
print(f"  Не прошли ГОСТ: {n_fail_gost} конфигураций")




# === ЗАДАНИЕ 4: Сравнительные графики (Matplotlib) ===

# Толщина 1 мм для всех металлов
thickness_mm_4 = 1.0
thickness_m_4 = thickness_mm_4 / 1000

# Словарь с параметрами металлов
metals_for_plot = {
    'Медь': {'mu_r': 1.0, 'sigma': 5.8e7},
    'Алюминий': {'mu_r': 1.0, 'sigma': 3.5e7},
    'Сталь': {'mu_r': 100.0, 'sigma': 1.0e7},
    'Никель': {'mu_r': 100.0, 'sigma': 1.45e7},
    'Олово': {'mu_r': 1.0, 'sigma': 8.7e6},
    'Латунь': {'mu_r': 1.0, 'sigma': 1.5e7}
}

# Рассчитываем SE для каждого металла при толщине 1 мм
se_1mm = {}
se_mean_1mm = {}

for metal_name, props in metals_for_plot.items():
    table = np.array([props['mu_r'], 1.0, props['sigma'], thickness_m_4, 1])
    se_values = calc_shield_se(freq, table)
    se_1mm[metal_name] = se_values
    se_mean_1mm[metal_name] = np.mean(se_values)

# --- График 1: Кривые SE(f) для всех металлов ---
plt.figure(figsize=(12, 8))

colors_plot = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
for i, (metal_name, se_values) in enumerate(se_1mm.items()):
    plt.semilogx(freq, se_values, color=colors_plot[i], linewidth=2, label=metal_name)

# Линия ГОСТ
gost_level_4 = 50
plt.axhline(y=gost_level_4, color='black', linestyle='--', linewidth=2,
            label=f'ГОСТ {gost_level_4} дБ')

plt.xlabel('Частота, Гц', fontsize=12)
plt.ylabel('Эффективность экранирования SE, дБ', fontsize=12)
plt.title(f'Сравнение эффективности экранирования металлов (толщина {thickness_mm_4} мм)',
          fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('task4_comparison_lines.png', dpi=150)
plt.show()

# --- График 2: Горизонтальная столбчатая диаграмма среднего SE ---
plt.figure(figsize=(10, 6))

metals_list = list(se_mean_1mm.keys())
mean_values = list(se_mean_1mm.values())

# Определяем цвета в зависимости от прохождения ГОСТ
bar_colors = ['green' if val >= gost_level_4 else 'red' for val in mean_values]

# Горизонтальная столбчатая диаграмма
bars = plt.barh(metals_list, mean_values, color=bar_colors, edgecolor='black', linewidth=1)

# Вертикальная линия ГОСТ
plt.axvline(x=gost_level_4, color='blue', linestyle='--', linewidth=2,
            label=f'ГОСТ {gost_level_4} дБ')

plt.xlabel('Средняя эффективность экранирования SE, дБ', fontsize=12)
plt.ylabel('Металл', fontsize=12)
plt.title(f'Средний SE для различных металлов (толщина {thickness_mm_4} мм)',
          fontsize=14, fontweight='bold')

# Добавляем подписи значений
for i, (bar, val) in enumerate(zip(bars, mean_values)):
    plt.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f} дБ', va='center', fontsize=10)

plt.legend(loc='lower right')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('task4_comparison_barh.png', dpi=150)
plt.show()

# Вывод результатов
print(f"\n=== Сравнение металлов при толщине {thickness_mm_4} мм ===")
print(f"Порог ГОСТ: {gost_level_4} дБ")
print("-" * 50)
for metal, mean_val in se_mean_1mm.items():
    status = "ПРОШЁЛ" if mean_val >= gost_level_4 else "НЕ ПРОШЁЛ"
    print(f"{metal:12s}: {mean_val:6.2f} дБ {status}")
