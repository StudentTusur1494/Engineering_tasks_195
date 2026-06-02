# =============================================================================
# ПРИМЕР: EDA НА КЛАССИЧЕСКОМ ДАТАСЕТЕ IRIS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Устанавливаем стиль графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("ПРИМЕР EDA: ДАТАСЕТ IRIS (Ирисы Фишера)")
print("=" * 70)

# --- ЗАГРУЗКА ДАННЫХ ---
iris = load_iris()
df_iris = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
df_iris['species'] = iris.target
df_iris['species_name'] = df_iris['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

print("\n📌 Датасет Iris содержит измерения 150 цветков ириса трёх видов")
print("   Признаки: длина и ширина чашелистика и лепестка")



# =============================================================================
# ШАГ 1: ПЕРВИЧНЫЙ ОСМОТР ДАННЫХ
# =============================================================================

print("\n" + "=" * 70)
print("ШАГ 1: ПЕРВИЧНЫЙ ОСМОТР")
print("=" * 70)

# 1.1 Размер датасета
print(f"\n📊 Размер датасета: {df_iris.shape[0]} строк, {df_iris.shape[1]} столбцов")

# 1.2 Первые строки
print("\n📋 Первые 5 строк:")
print(df_iris.head())

# 1.3 Информация о типах данных
print("\n📋 Информация о столбцах:")
print(df_iris.info())

# 1.4 Базовая статистика
print("\n📊 Статистические характеристики:")
print(df_iris.describe())

# 1.5 Проверка на пропуски
print("\n🔍 Проверка на пропущенные значения:")
missing = df_iris.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✅ Пропусков нет!")




# =============================================================================
# ШАГ 2: ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ
# =============================================================================

print("\n" + "=" * 70)
print("ШАГ 2: АНАЛИЗ РАСПРЕДЕЛЕНИЙ ПРИЗНАКОВ")
print("=" * 70)

# 2.1 Гистограммы для всех числовых признаков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features = iris.feature_names

for i, ax in enumerate(axes.flat):
    ax.hist(df_iris[features[i]], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel(features[i], fontsize=11)
    ax.set_ylabel('Частота', fontsize=11)
    ax.set_title(f'Распределение: {features[i]}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 2.2 Boxplot для выявления выбросов
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, ax in enumerate(axes.flat):
    df_iris.boxplot(column=features[i], by='species_name', ax=ax)
    ax.set_xlabel('Вид', fontsize=11)
    ax.set_ylabel(features[i], fontsize=11)
    ax.set_title(f'Boxplot: {features[i]}', fontsize=12, fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n💡 Boxplot показывает медиану, quartiles и выбросы")
print("   Выбросы — это точки за пределами 'усов'")




# =============================================================================
# ШАГ 3: АНАЛИЗ ВЗАИМОСВЯЗЕЙ МЕЖДУ ПРИЗНАКАМИ
# =============================================================================

print("\n" + "=" * 70)
print("ШАГ 3: АНАЛИЗ КОРРЕЛЯЦИЙ И ЗАВИСИМОСТЕЙ")
print("=" * 70)

# 3.1 Матрица корреляций
correlation_matrix = df_iris[features].corr()
print("\n📊 Матрица корреляций:")
print(correlation_matrix)

# 3.2 Тепловая карта корреляций
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,          # Показывать числа
    fmt='.2f',           # Формат: 2 знака после запятой
    cmap='coolwarm',     # Цветовая схема
    square=True,         # Квадратные ячейки
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)
plt.title('Матрица корреляций признаков Iris', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Красные цвета — сильная положительная корреляция")
print("   Синие цвета — слабая или отрицательная корреляция")




# =============================================================================
# ШАГ 4: PAIRPLOT - ПОПАРНЫЕ ЗАВИСИМОСТИ
# =============================================================================

print("\n" + "=" * 70)
print("ШАГ 4: ПОПАРНЫЕ ГРАФИКИ (PAIRPLOT)")
print("=" * 70)

# Pairplot показывает все возможные комбинации признаков
sns.pairplot(
    df_iris,
    hue='species_name',     # Раскрасить по видам
    diag_kind='hist',        # На диагонали — гистограммы
    plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
    height=2.5
)
plt.suptitle('Попарные зависимости признаков Iris', y=1.02, fontsize=14, fontweight='bold')
plt.show()

print("\n💡 Pairplot позволяет быстро увидеть:")
print("   - Как распределены признаки (диагональ)")
print("   - Как признаки связаны друг с другом")
print("   - Как разделяются классы в пространстве признаков")




# =============================================================================
# ШАГ 5: ВЫЯВЛЕНИЕ ВЫБРОСОВ
# =============================================================================

print("\n" + "=" * 70)
print("ШАГ 5: ПОИСК ВЫБРОСОВ (OUTLIERS)")
print("=" * 70)

# Метод: значения за пределами Q1 - 1.5*IQR и Q3 + 1.5*IQR считаются выбросами

def find_outliers(data, column):
    """Находит выбросы методом IQR"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\n🔍 Поиск выбросов в каждом признаке:")
for feature in features:
    outliers, lower, upper = find_outliers(df_iris, feature)
    print(f"\n   {feature}:")
    print(f"   Диапазон нормы: [{lower:.2f}, {upper:.2f}]")
    print(f"   Найдено выбросов: {len(outliers)}")
    if len(outliers) > 0:
        print(f"   Индексы: {outliers.index.tolist()}")




# =============================================================================
# ШАГ 6: ПОДГОТОВКА ДАННЫХ ДЛЯ МАШИННОГО ОБУЧЕНИЯ
# =============================================================================

print("\n" + "=" * 70)
print("ШАГ 6: ПОДГОТОВКА ДАННЫХ")
print("=" * 70)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 6.1 Разделение на признаки и целевую переменную
X = df_iris[features].values
y = df_iris['species'].values

print(f"\n📊 Матрица признаков X: {X.shape}")
print(f"📊 Вектор целевых значений y: {y.shape}")

# 6.2 Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% на тест
    random_state=42,
    stratify=y          # Сохранить пропорции классов
)

print(f"\n✅ Обучающая выборка: {X_train.shape[0]} образцов")
print(f"✅ Тестовая выборка: {X_test.shape[0]} образцов")

# 6.3 Стандартизация
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Данные нормализованы")
print(f"   Среднее до: {X_train[:, 0].mean():.2f}, после: {X_train_scaled[:, 0].mean():.2e}")
print(f"   Std до: {X_train[:, 0].std():.2f}, после: {X_train_scaled[:, 0].std():.2f}")

# 6.4 Проверка распределения классов
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n📊 Распределение классов в обучающей выборке:")
for class_id, count in zip(unique, counts):
    class_name = ['setosa', 'versicolor', 'virginica'][class_id]
    print(f"   {class_name}: {count} ({count/len(y_train)*100:.1f}%)")