# ============================================================
#  ФАЙЛ 2: ПРАКТИЧЕСКОЕ ЗАДАНИЕ
#  Создание интерактивного инженерного калькулятора
# ============================================================

"""
=======================================================================
 📝 ОПИСАНИЕ ПРАКТИЧЕСКОЙ РАБОТЫ
=======================================================================

ЧАСТЬ А: Базовый интерактивный калькулятор
------------------------------------------
Ваша задача — создать оконное приложение, в котором график перестраивается
АВТОМАТИЧЕСКИ (без кнопки "Построить") при любом изменении настроек.

Интерфейс должен состоять из:
1. Левой части: Холст для графика (готовый класс MplCanvas).
2. Правой части (панель управления):
   - Два текстовых поля (QLineEdit) для задания диапазона X (от и до).
   - Четыре радиокнопки (QRadioButton) для выбора функции:
     sin(x), cos(x), x^2, sqrt(x).
   - Чекбоксы для включения сетки и легенды.

Ключевой момент: все виджеты должны быть подключены напрямую к методу
отрисовки графика через сигналы .textChanged.connect() или .toggled.connect().

ЧАСТЬ Б: Самостоятельная работа (Усложнение)
------------------------------------------
Переделайте логику калькулятора:
1. Удалите группу радиокнопок.
2. Добавьте ОДНО текстовое поле (QLineEdit), куда пользователь будет
   вписывать уравнение математической функции (например: "sin(x) + 2*x").
3. График должен перестраиваться при вводе текста.
4. Для расчета значений Y используйте функцию eval() в связке со словарем
(см. подсказку ниже).

💡 Краткая подсказка по eval() для самостоятельной работы:
eval() выполняет строку как код Питона. Чтобы программа понимала математику:

   import math

   # Словарь функций, которые мы разрешаем использовать
   safe_math = {"sin": math.sin, "cos": math.cos, "sqrt": math.sqrt}

   # Считаем значение Y для конкретного X
   safe_math["x"] = 5.0 # Добавляем текущий X в словарь
   y = eval("sin(x) * 2", {"__builtins__": None}, safe_math)

=======================================================================
"""

import sys
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QCheckBox, QRadioButton, QGroupBox
)

# Импорты для Matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ============================================================
# ГОТОВЫЙ ИНСТРУМЕНТ: Класс для холста Matplotlib
# ============================================================

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


# ============================================================
# ЧАСТЬ А: БАЗОВОЕ ЗАДАНИЕ (Заполните пропуски)
# ============================================================

class InteractiveCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Практика: Интерактивный калькулятор")
        self.setGeometry(100, 100, 800, 500)

        # Базовая настройка главного окна (оставляем готовой)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая часть: Холст графика (оставляем готовой)
        self.canvas = MplCanvas()
        main_layout.addWidget(self.canvas, stretch=2)

        # ==========================================================
        # ПАНЕЛЬ НАСТРОЕК (ПРАВАЯ ЧАСТЬ)
        # ==========================================================

        # ШАГ 1: Создайте вертикальный Layout (QVBoxLayout) и назовите его settings_layout.
        # Добавьте его внутрь главного main_layout (используйте addLayout, stretch=1).
        # >>> ВАШ КОД ЗДЕСЬ <<<
        settings_layout = QVBoxLayout()
        main_layout.addLayout(settings_layout, stretch=1)

        # ШАГ 2: Создайте текстовую метку (QLabel) с текстом "Диапазон X:"
        # и добавьте её в settings_layout.
        # >>> ВАШ КОД ЗДЕСЬ <<<
        x_range_label = QLabel("Диапозон X:")
        settings_layout.addWidget(x_range_label)

        # ШАГ 3: Создайте поле ввода (QLineEdit) со значением "-10".
        # Создайте метку "От:" и добавьте метку и поле ввода в settings_layout.
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.x_min_input = QLineEdit("-10")
        x_min_label = QLabel("От:")
        settings_layout.addWidget(x_min_label)
        settings_layout.addWidget(self.x_min_input)

        # ШАГ 4: Сделайте то же самое для максимального значения:
        # со значением "10", метку "До:" и добавьте их в layout.
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.x_max_input = QLineEdit("10")
        x_max_label = QLabel("До:")
        settings_layout.addWidget(x_max_label)
        settings_layout.addWidget(self.x_max_input)

        # --- Блок выбора функции ---
        func_group = QGroupBox("Выберите функцию:")
        func_layout = QVBoxLayout()  # Вспомогательный layout для рамки

        # ШАГ 5: Создайте 4 радиокнопки (QRadioButton) и сохраните их как атрибуты:
        # Дайте им соответствующие текстовые названия ("sin(x)", "cos(x)" и т.д.).
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.cos_radio = QRadioButton("sin(x)")
        self.sin_radio = QRadioButton("cos(x)")
        self.square_radio = QRadioButton("x^2")
        self.sqrt_radio = QRadioButton("sqrt(x)")


        # ШАГ 6: Сделайте кнопку sin(x) выбранной по умолчанию (метод setChecked).
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.sin_radio.setChecked(True)

        # ШАГ 7: Добавьте все 4 радиокнопки внутрь func_layout.
        # >>> ВАШ КОД ЗДЕСЬ <<<
        func_layout.addWidget(self.cos_radio)
        func_layout.addWidget(self.sin_radio)
        func_layout.addWidget(self.square_radio)
        func_layout.addWidget(self.sqrt_radio)

        func_group.setLayout(func_layout)
        settings_layout.addWidget(func_group)  # Добавляем рамку с кнопками на панель

        # ШАГ 8: Создайте чекбокс (QCheckBox) с текстом "Показывать сетку".
        # Сохраните его как self.grid_checkbox, включите по умолчанию и добавьте в settings_layout.
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.grid_checkbox = QCheckBox("Показывать сетку")
        self.grid_checkbox.setChecked(True)
        settings_layout.addWidget(self.grid_checkbox)

        settings_layout.addStretch()  # Пружина, чтобы прижать настройки наверх

        # ==========================================================
        # ПОДКЛЮЧЕНИЕ СИГНАЛОВ (АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ)
        # ==========================================================

        # ШАГ 9: Подключите сигналы к методу self.update_graph.
        # - Для полей ввода (x_min, x_max) используйте сигнал .textChanged
        # - Для всех радиокнопок и чекбокса используйте сигнал .toggled
        # Напоминание: метод в connect передается БЕЗ скобок!
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.x_min_input.textChanged.connect(self.update_graph)
        self.x_max_input.textChanged.connect(self.update_graph)
        self.sin_radio.toggled.connect(self.update_graph)
        self.cos_radio.toggled.connect(self.update_graph)
        self.square_radio.toggled.connect(self.update_graph)
        self.sqrt_radio.toggled.connect(self.update_graph)
        self.grid_checkbox.toggled.connect(self.update_graph)

        # Вызываем метод один раз при запуске, чтобы нарисовать стартовый график
        self.update_graph()

    def update_graph(self):
        """Метод считывает настройки, рассчитывает Y и рисует график"""

        # ШАГ 10: Считайте текст из self.x_min_input и self.x_max_input.
        # Переведите текст в дробное число (float) и сохраните в переменные x_min и x_max.
        # Обязательно оберните этот код в блок try...except ValueError: return
        # >>> ВАШ КОД ЗДЕСЬ <<<
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
        except ValueError:
            return

        # Защита от неправильного диапазона
        if 'x_min' in locals() and 'x_max' in locals():
            if x_min >= x_max:
                return
        else:
            return

        # Генерируем массив X (100 точек)
        step = (x_max - x_min) / 100
        x_data = [x_min + i * step for i in range(101)]
        y_data = []
        label_text = "Функция"

        # ШАГ 11: Напишите логику расчета Y (if / elif / else).
        # Проверяйте, какая радиокнопка нажата (метод isChecked()).
        # Расчет для y_data делайте через генератор списков (например: [math.sin(x) for x in x_data]).
        # Для sqrt(x) используйте math.sqrt(abs(x)), чтобы избежать ошибки корня из минуса.
        # Не забудьте менять label_text для каждого графика.
        # >>> ВАШ КОД ЗДЕСЬ <<<
        if self.sin_radio.isChecked():
            y_data = [math.sin(x) for x in x_data]
            label_text = "sin(x)"
        elif self.cos_radio.isChecked():
            y_data = [math.cos(x) for x in x_data]
            label_text = "cos(x)"
        elif self.square_radio.isChecked():
            y_data = [x ** 2 for x in x_data]
            label_text = "x^2"
        else:
            y_data = [math.sqrt(abs(x)) for x in x_data]  # abs для избежания ошибок
            label_text = "sqrt(x)"

        # ==========================================================
        # ОТРИСОВКА ГРАФИКА
        # ==========================================================

        # ШАГ 12: Очистите старый график (метод clear у self.canvas.axes).
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.canvas.axes.clear()
        # ШАГ 13: Постройте новый график (метод plot у self.canvas.axes).
        # Передайте x_data, y_data, label=label_text, color="blue", linewidth=2.
        # Установите заголовок и подписи осей X и Y.
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.canvas.axes.plot(x_data, y_data, label=label_text, color='blue', linewidth=2)
        self.canvas.axes.set_title('График функции')
        self.canvas.axes.set_xlabel('X')
        self.canvas.axes.set_ylabel('Y')


        # ШАГ 14: Проверьте состояние чекбокса сетки.
        # Если он включен, отобразите сетку (метод grid(True) у self.canvas.axes).
        # >>> ВАШ КОД ЗДЕСЬ <<<
        if self.grid_checkbox.isChecked():
            self.canvas.axes.grid(True)
        else:
            self.canvas.axes.grid(False)

        self.canvas.axes.legend()  # Оставляем легенду включенной всегда для простоты

        # ШАГ 15: Дайте команду холсту перерисоваться (метод draw у self.canvas).
        # >>> ВАШ КОД ЗДЕСЬ <<<
        self.canvas.draw()


# ============================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================
#if __name__ == "__main__":
    #app = QApplication(sys.argv)

    # Для выполнения ЧАСТИ А запускаем InteractiveCalculator
    #window = InteractiveCalculator()

    #window.show()
    #sys.exit(app.exec())

# ============================================================
# ЧАСТЬ Б: САМОСТОЯТЕЛЬНАЯ РАБОТА (Шаблон)
# ============================================================
# 1. Создайте класс AdvancedCalculator(QMainWindow) скопировав код из InteractiveCalculator.
# 2. Замените QGroupBox с радиокнопками на QLabel("Введите уравнение:")
#    и QLineEdit (например, self.equation_input).
# 3. Подключите self.equation_input.textChanged к методу update_graph.
# 4. В методе update_graph удалите блок if/elif с радиокнопками.
# 5. Вместо него напишите цикл, который проходит по массиву x_data и
#    высчитывает y_data с помощью функции eval() и вашего уравнения.
# 6. В блоке запуска (if __name__ == "__main__") замените вызов окна
#    на window = AdvancedCalculator().

class AdvancedCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Усложнённый калькулятор")
        self.setGeometry(100, 100, 800, 500)

        # Базовая настройка главного окна
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая часть: Холст графика
        self.canvas = MplCanvas()
        main_layout.addWidget(self.canvas, stretch=2)

        # Панель настроек (правая часть)
        settings_layout = QVBoxLayout()
        main_layout.addLayout(settings_layout, stretch=1)

        # Метка и поле для ввода уравнения
        equation_label = QLabel("Введите уравнение:")
        settings_layout.addWidget(equation_label)
        self.equation_input = QLineEdit("sin(x)")
        settings_layout.addWidget(self.equation_input)

        # Диапазон X
        x_range_label = QLabel("Диапазон X:")
        settings_layout.addWidget(x_range_label)

        self.x_min_input = QLineEdit("-10")
        x_min_label = QLabel("От:")
        settings_layout.addWidget(x_min_label)
        settings_layout.addWidget(self.x_min_input)

        self.x_max_input = QLineEdit("10")
        x_max_label = QLabel("До:")
        settings_layout.addWidget(x_max_label)
        settings_layout.addWidget(self.x_max_input)

        self.grid_checkbox = QCheckBox("Показывать сетку")
        self.grid_checkbox.setChecked(True)
        settings_layout.addWidget(self.grid_checkbox)

        settings_layout.addStretch()

        self.x_min_input.textChanged.connect(self.update_graph)
        self.x_max_input.textChanged.connect(self.update_graph)
        self.equation_input.textChanged.connect(self.update_graph)
        self.grid_checkbox.toggled.connect(self.update_graph)

        self.update_graph()

    def update_graph(self):
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
        except ValueError:
            return
        if x_min >= x_max:
            return
        step = (x_max - x_min) / 100
        x_data = [x_min + i * step for i in range(101)]
        y_data = []

        safe_math = {
            "sin": math.sin,
            "cos": math.cos,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e
        }
        equation = self.equation_input.text().strip()
        if not equation:
            equation = "0"
        for x in x_data:
            try:
                safe_math["x"] = x
                y = eval(equation, {"__builtins__": None}, safe_math)
                y_data.append(y)
            except Exception:
                y_data.append(float('nan'))

        label_text = equation

        # Отрисовка графика
        self.canvas.axes.clear()
        self.canvas.axes.plot(x_data, y_data, label=label_text, color='blue', linewidth=2)
        self.canvas.axes.set_title('График функции')
        self.canvas.axes.set_xlabel('X')
        self.canvas.axes.set_ylabel('Y')

        if self.grid_checkbox.isChecked():
            self.canvas.axes.grid(True)
        else:
            self.canvas.axes.grid(False)
        self.canvas.axes.legend()
        self.canvas.draw()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedCalculator()
    window.show()
    sys.exit(app.exec())