# =============================================================================
# ГЕНЕРАЦИЯ ДАТАСЕТА ДЛЯ ПРАКТИЧЕСКОЙ РАБОТЫ
# =============================================================================

import numpy as np
import pandas as pd

# Физические константы
mu0 = 4.0 * np.pi * 1e-7
eps0 = 8.854187817e-12
Z0 = 376.73

def calc_shield_se(freq, table):
    """
    Расчёт эффективности экранирования (SE) методом матриц передачи.

    Параметры:
    ----------
    freq : array
        Массив частот (Гц)
    table : array
        Массив параметров слоёв [mu_r, eps_r, sigma, t, mat_type]

    Возвращает:
    -----------
    SE : array
        Эффективность экранирования в дБ
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

# Фиксированные параметры материала
SIGMA = 5.8e7      # Проводимость (См/м) — медь
EPS_R = 1.0        # Относительная диэлектрическая проницаемость
T_FIXED = 1e-4     # Толщина экрана (м) — 0.1 мм

print("=" * 70)
print("ГЕНЕРАЦИЯ ДАТАСЕТА: ЭФФЕКТИВНОСТЬ ЭЛЕКТРОМАГНИТНОГО ЭКРАНИРОВАНИЯ")
print("=" * 70)

# --- ГЕНЕРАЦИЯ ДАННЫХ ---
np.random.seed(42)
target_freq = 1e+6  # 1 МГц (фиксированная частота)
n_samples = 300     # Количество измерений

# Генерируем случайные значения магнитной проницаемости μᵣ
mu_dataset = np.random.uniform(10, 100, n_samples)

# Вычисляем идеальные (аналитические) значения SE
se_ideal = []
for m in mu_dataset:
    layer = np.array([m, EPS_R, SIGMA, T_FIXED, 1])
    se_ideal.append(calc_shield_se(np.array([target_freq]), layer)[0])

se_ideal = np.array(se_ideal)

# ДОБАВЛЯЕМ ШУМ (имитация реальных измерений)
# Шум — нормальное распределение с std = 3% от среднего значения SE
noise_level = 0.03 * np.mean(se_ideal)
noise = np.random.normal(0, noise_level, n_samples)
se_measured = se_ideal + noise

# Создаём DataFrame
df = pd.DataFrame({
    'mu_r': mu_dataset,
    'SE_dB': se_measured
})

print(f"\n✅ Данные успешно сгенерированы!")
print(f"📊 Размер датасета: {df.shape[0]} образцов")
print(f"🎯 Частота измерений: {target_freq/1e6:.1f} МГц")
print(f"📈 Диапазон μᵣ: {mu_dataset.min():.1f} – {mu_dataset.max():.1f}")
print(f"🔊 Уровень шума: {noise_level:.2f} дБ (σ)")

print("\n💡 Датасет готов для EDA!")
print("   Переменная 'df' содержит таблицу с данными")




# =============================================================================
# ПРАКТИЧЕСКОЕ ЗАДАНИЕ 1: ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ
# =============================================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 1: ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ")
print("=" * 70)

import matplotlib.pyplot as plt
import seaborn as sns

# 1.1 Гистограммы для обоих признаков
fig, axes = plt.subplots(1, 2, figsize=(14, 5))


# Гистограмма для μᵣ
axes[0].hist(df['mu_r'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Магнитная проницаемость μᵣ', fontsize=12)
axes[0].set_ylabel('Частота', fontsize=12)
axes[0].set_title('Распределение μᵣ', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)


# Гистограмма для SE
axes[1].hist(df['SE_dB'], bins=20, edgecolor='black', alpha=0.7, color='darkorange')
axes[1].set_xlabel('Эффективность экранирования SE, дБ', fontsize=12)
axes[1].set_ylabel('Частота', fontsize=12)
axes[1].set_title('Распределение SE', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 1.2 Boxplot для обоих признаков
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot для μᵣ
df.boxplot(column='mu_r', ax=axes[0])
axes[0].set_title('Boxplot: μᵣ', fontsize=14, fontweight='bold')
axes[0].set_ylabel('μᵣ', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Boxplot для SE
df.boxplot(column='SE_dB', ax=axes[1])
axes[1].set_title('Boxplot: SE (дБ)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('SE, дБ', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 Вопросы для анализа:")
print("   1. Равномерно ли распределены значения μᵣ?")
print("   2. Какое распределение имеет SE (нормальное/скошенное)?")
print("   3. Видны ли выбросы на boxplot?")




# =============================================================================
# ПРАКТИЧЕСКОЕ ЗАДАНИЕ 2: АНАЛИЗ ВЗАИМОСВЯЗЕЙ
# =============================================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 2: АНАЛИЗ КОРРЕЛЯЦИЙ И ЗАВИСИМОСТЕЙ")
print("=" * 70)

# 2.1 Матрица корреляций (вычисляем для двух столбцов)
correlation_matrix = df[['mu_r', 'SE_dB']].corr()
print("\n📊 Матрица корреляций:")
print(correlation_matrix)

# 2.2 Тепловая карта корреляций
plt.figure(figsize=(6, 5))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.3f',
    cmap='coolwarm',
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)
plt.title('Корреляция между μᵣ и SE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 2.3 Scatter plot (диаграмма рассеяния)
plt.figure(figsize=(8, 6))
plt.scatter(df['mu_r'], df['SE_dB'], alpha=0.6, color='steelblue', edgecolor='k')
plt.xlabel('Магнитная проницаемость μᵣ', fontsize=12)
plt.ylabel('Эффективность экранирования SE, дБ', fontsize=12)
plt.title('Зависимость SE от μᵣ (с шумом)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2.4 Scatter plot с цветовой кодировкой (по значениям SE)
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['mu_r'], df['SE_dB'], c=df['SE_dB'], cmap='viridis', alpha=0.7, edgecolor='k')
plt.colorbar(sc, label='SE, дБ')
plt.xlabel('Магнитная проницаемость μᵣ', fontsize=12)
plt.ylabel('Эффективность экранирования SE, дБ', fontsize=12)
plt.title('Scatter plot с цветовой кодировкой по SE', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n💡 Вопросы для анализа:")
print("   1. Какова корреляция между μᵣ и SE?")
print("   2. Линейная ли зависимость или нелинейная?")
print("   3. Какой характер зависимости (прямая/обратная)?")




# =============================================================================
# ПРАКТИЧЕСКОЕ ЗАДАНИЕ 3: ВЫЯВЛЕНИЕ ВЫБРОСОВ
# =============================================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 3: ПОИСК ВЫБРОСОВ (OUTLIERS)")
print("=" * 70)

def find_outliers_iqr(data, column):
    """
    Находит выбросы методом IQR.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    return outliers, lower_bound, upper_bound

# 3.1 Поиск выбросов для μᵣ
outliers_mu, lower_mu, upper_mu = find_outliers_iqr(df, 'mu_r')
print(f"\n🔍 μᵣ: нормальный диапазон [{lower_mu:.2f}, {upper_mu:.2f}]")
print(f"   Найдено выбросов: {len(outliers_mu)}")
if len(outliers_mu) > 0:
    print(f"   Индексы выбросов: {outliers_mu.index.tolist()}")

# 3.2 Поиск выбросов для SE
outliers_se, lower_se, upper_se = find_outliers_iqr(df, 'SE_dB')
print(f"\n🔍 SE: нормальный диапазон [{lower_se:.2f}, {upper_se:.2f}]")
print(f"   Найдено выбросов: {len(outliers_se)}")
if len(outliers_se) > 0:
    print(f"   Индексы выбросов: {outliers_se.index.tolist()}")

# 3.3 Визуализация выбросов на scatter plot
plt.figure(figsize=(10, 6))

# Все точки
plt.scatter(df['mu_r'], df['SE_dB'], alpha=0.5, color='lightgray', label='Нормальные точки')

# Подсветка выбросов SE (если есть)
if len(outliers_se) > 0:
    plt.scatter(outliers_se['mu_r'], outliers_se['SE_dB'],
                color='red', s=80, edgecolor='k', label='Выбросы (SE)')
# Подсветка выбросов μᵣ (если есть)
if len(outliers_mu) > 0:
    plt.scatter(outliers_mu['mu_r'], outliers_mu['SE_dB'],
                color='orange', s=80, marker='s', edgecolor='k', label='Выбросы (μᵣ)')

plt.xlabel('Магнитная проницаемость μᵣ', fontsize=12)
plt.ylabel('SE, дБ', fontsize=12)
plt.title('Визуализация выбросов', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3.4 Статистика до и после удаления выбросов
print("\n📊 Статистика:")
print(f"\nДо удаления выбросов:")
print(f"   Количество образцов: {len(df)}")
print(f"   Среднее SE: {df['SE_dB'].mean():.2f} дБ")
print(f"   Std SE: {df['SE_dB'].std():.2f} дБ")

if len(outliers_se) > 0:
    df_clean = df.drop(outliers_se.index)
    print(f"\nПосле удаления выбросов (только по SE):")
    print(f"   Количество образцов: {len(df_clean)}")
    print(f"   Среднее SE: {df_clean['SE_dB'].mean():.2f} дБ")
    print(f"   Std SE: {df_clean['SE_dB'].std():.2f} дБ")
    print(f"\n   Удалено образцов: {len(outliers_se)}")

print("\n💡 Вопрос: нужно ли удалять выбросы в данной задаче?")
print("   Подсказка: это физическая модель с добавленным шумом, а не реальные измерения")