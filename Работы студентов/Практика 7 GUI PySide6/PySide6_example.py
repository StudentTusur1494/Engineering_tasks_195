"""
=======================================================================
 📚 КАК РАБОТАТЬ С GUI (PySide6) И СТРОКАМ
=======================================================================

В оконных приложениях программа не идет сверху вниз, как в обычном скрипте.
Она "ждет", пока пользователь что-то сделает (нажмет кнопку, введет текст).
Этот механизм называется "Сигналы и Слоты" (Signals & Slots).


-----------------------------------------------------------------------
 0. ЧТО ТАКОЕ LAYOUT (КОМПОНОВКА) И ЗАЧЕМ ОН НУЖЕН?
-----------------------------------------------------------------------
В PySide6 мы НЕ задаем координаты кнопок в пикселях (x=100, y=50).
Если окно растянут, всё сломается. Мы используем "умные коробки" — Layout'ы.

    • QHBoxLayout (Horizontal) — выстраивает виджеты в линию слева направо [ A | B | C ].
    • QVBoxLayout (Vertical)   — выстраивает виджеты в колонку сверху вниз.

Как это работает (принцип матрешки):
1. Создаем Layout:         my_box = QVBoxLayout()
2. Кладем в него виджет:   my_box.addWidget(my_button)
3. Кладем Layout в окно:   window.setLayout(my_box)

Важно:
- Чтобы положить виджет в Layout, пишем:  layout.addWidget(widget)
- Чтобы положить ОДИН LAYOUT ВНУТРЬ ДРУГОГО: layout.addLayout(other_layout)

-----------------------------------------------------------------------
 1. КАК ЗАСТАВИТЬ КНОПКУ РАБОТАТЬ (СИГНАЛЫ И СЛОТЫ)
-----------------------------------------------------------------------
Чтобы при нажатии на кнопку выполнялся ваш код, нужно связать её "сигнал"
(нажатие) с вашим "слотом" (методом/функцией).

    self.my_button.clicked.connect(self.calculate_data)

    ВНИМАНИЕ! ГЛАВНАЯ ОШИБКА НОВИЧКОВ:
    Не ставьте скобки () после названия метода внутри connect!

-----------------------------------------------------------------------
 2. ЧТЕНИЕ ДАННЫХ ИЗ ИНТЕРФЕЙСА (ЧТО ВВЕЛ ПОЛЬЗОВАТЕЛЬ?)
-----------------------------------------------------------------------
Виджеты хранят данные в разных форматах. Чтобы произвести расчет,
эти данные нужно "достать" правильным методом:

    • QLineEdit (Поле ввода)
      text = self.my_input.text()          -> Возвращает СТРОКУ ("123").
      num = float(self.my_input.text())    -> Если нужна математика, обязательно оборачиваем в float() или int()!

    • QComboBox (Выпадающий список)
      choice = self.my_combo.currentText() -> Возвращает СТРОКУ (название выбранного пункта).

    • QSpinBox (Счетчик с числами)
      value = self.my_spin.value()         -> Возвращает ЧИСЛО (int). Переводить не нужно.

    • QCheckBox / QRadioButton (Галочки и Точки)
      is_on = self.my_check.isChecked()    -> Возвращает логическое значение (True или False).

-----------------------------------------------------------------------
 3. ЗАПИСЬ ДАННЫХ В ИНТЕРФЕЙС (ВЫВОД РЕЗУЛЬТАТА)
-----------------------------------------------------------------------
Когда расчет окончен, результат нужно показать на экране:

    • QLabel, QLineEdit:   self.my_label.setText(f"Ответ: {result}")
    • QSpinBox:            self.my_spin.setValue(10)
    • QCheckBox:           self.my_check.setChecked(True)

-----------------------------------------------------------------------
 4. БАЗОВЫЙ ПАРСИНГ (КАК ПОНЯТЬ ВВЕДЕННУЮ ФОРМУЛУ)
-----------------------------------------------------------------------
Пользователь может ввести "SIN(x)", "  sin(x)  " или "Sin".
Программа должна понимать это одинаково. Поэтому строку нужно чистить:

    raw_text = self.equation_input.text()  # Допустим, ввели "   Cos(x)  "

    clean_text = raw_text.strip().lower()
    # .strip() удаляет лишние пробелы в начале и в конце
    # .lower() переводит все буквы в маленькие
    # Итог: "cos(x)"

    # Как проверить, какую функцию просит пользователь?
    if "sin" in clean_text:
        # Считаем синус
    elif "cos" in clean_text:
        # Считаем косинус
=======================================================================
"""

import sys
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox, QRadioButton,
    QPushButton, QTableWidget, QTableWidgetItem, QGroupBox, QHeaderView
)

# Импорты для Matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ============================================================
# БЛОК 1: Полный класс для вставки графика Matplotlib в GUI
# ============================================================
# (Студентам не нужно писать это самим, просто используем готовый холст)

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


# ============================================================
# БЛОК 2: Пример создания GUI (График, Таблица и Настройки)
# ============================================================

class PlotSettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Теория: Основные виджеты PySide6")
        self.setGeometry(100, 100, 900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- ЛЕВАЯ ЧАСТЬ: График и Таблица ---
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=2)

        self.canvas = MplCanvas()
        left_layout.addWidget(self.canvas, stretch=2)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(2)
        self.data_table.setHorizontalHeaderLabels(["Координата X", "Координата Y"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left_layout.addWidget(self.data_table, stretch=1)

        # --- ПРАВАЯ ЧАСТЬ: Панель настроек ---
        settings_layout = QVBoxLayout()
        main_layout.addLayout(settings_layout, stretch=1)

        # QLineEdit
        settings_layout.addWidget(QLabel("Название графика:"))
        self.title_input = QLineEdit("Парабола")
        settings_layout.addWidget(self.title_input)

        # QComboBox
        settings_layout.addWidget(QLabel("Тип линии:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["-", "--", "-.", ":"])
        settings_layout.addWidget(self.style_combo)

        # QSpinBox
        settings_layout.addWidget(QLabel("Толщина линии:"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 5)
        self.width_spinbox.setValue(2)
        settings_layout.addWidget(self.width_spinbox)

        # QRadioButton
        color_group = QGroupBox("Цвет графика")
        color_layout = QVBoxLayout()
        self.radio_blue = QRadioButton("Синий")
        self.radio_blue.setChecked(True)
        self.radio_red = QRadioButton("Красный")
        self.radio_green = QRadioButton("Зеленый")

        color_layout.addWidget(self.radio_blue)
        color_layout.addWidget(self.radio_red)
        color_layout.addWidget(self.radio_green)
        color_group.setLayout(color_layout)
        settings_layout.addWidget(color_group)

        # QCheckBox
        self.grid_checkbox = QCheckBox("Показывать сетку")
        self.grid_checkbox.setChecked(True)
        settings_layout.addWidget(self.grid_checkbox)

        self.legend_checkbox = QCheckBox("Показывать легенду")
        self.legend_checkbox.setChecked(True)
        settings_layout.addWidget(self.legend_checkbox)

        settings_layout.addStretch()

        # QPushButton
        self.apply_btn = QPushButton("Построить и рассчитать")
        self.apply_btn.clicked.connect(self.update_plot_and_table)
        settings_layout.addWidget(self.apply_btn)

        # Отрисовка графика сразу при запуске программы
        self.update_plot_and_table()

    def update_plot_and_table(self):
        """Метод считывает данные из интерфейса и обновляет холст/таблицу"""

        title = self.title_input.text()
        line_style = self.style_combo.currentText()
        line_width = self.width_spinbox.value()

        if self.radio_red.isChecked():
            color = "red"
        elif self.radio_green.isChecked():
            color = "green"
        else:
            color = "blue"

        show_grid = self.grid_checkbox.isChecked()
        show_legend = self.legend_checkbox.isChecked()

        # Генерируем точки (x от -5.0 до 5.0 с шагом 0.5)
        x_data = [x * 0.5 for x in range(-10, 11)]
        y_data = [x ** 2 for x in x_data]

        # Заполняем таблицу
        self.data_table.setRowCount(len(x_data))
        for row in range(len(x_data)):
            self.data_table.setItem(row, 0, QTableWidgetItem(f"{x_data[row]:.1f}"))
            self.data_table.setItem(row, 1, QTableWidgetItem(f"{y_data[row]:.2f}"))

        # Перерисовываем график
        self.canvas.axes.clear()
        self.canvas.axes.plot(
            x_data, y_data, linestyle=line_style, linewidth=line_width,
            label="y = x^2", color=color
        )
        self.canvas.axes.set_title(title)
        if show_grid: self.canvas.axes.grid(True)
        if show_legend: self.canvas.axes.legend()
        self.canvas.draw()


# ============================================================
# БЛОК 3: Тривиальный пример анализа текста (базовый парсинг)
# ============================================================

def simple_text_parsing_example():
    """
    Пример того, как программа может понять текст пользователя
    с помощью базовых строковых методов и условий if/elif.
    """
    print("\n--- Пример 3: Тривиальный анализ текста ---")

    user_input = "   Cos(x)  "
    print(f"Оригинальный ввод пользователя: '{user_input}'")

    # Очищаем строку: удаляем пробелы по краям и переводим в нижний регистр
    clean_input = user_input.strip().lower()
    print(f"Очищенный ввод для программы: '{clean_input}'\n")

    x_values = [0.0, 1.0, 2.0, 3.0]
    print("Начинаем расчет по точкам:")

    for x in x_values:
        # Проверяем, есть ли подстрока во введенном тексте
        if "sin" in clean_input:
            y = math.sin(x)
            func_name = "sin"
        elif "cos" in clean_input:
            y = math.cos(x)
            func_name = "cos"
        elif "tan" in clean_input:
            y = math.tan(x)
            func_name = "tan"
        else:
            y = x
            func_name = "x"

        print(f"При x = {x}, функция {func_name}(x) = {y:.4f}")


# ============================================================
# ЗАПУСК ПРИМЕРОВ
# ============================================================
if __name__ == "__main__":
    # 1. Показываем работу с текстом в консоли
    simple_text_parsing_example()

    # 2. Запускаем оконное приложение
    app = QApplication(sys.argv)
    window = PlotSettingsWindow()
    window.show()
    sys.exit(app.exec())