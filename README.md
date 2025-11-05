# Веб-приложение для классификации изображений

Это веб-приложение, построенное с использованием FastAPI и TensorFlow, для классификации изображений с использованием предобученной модели CIFAR-10. Приложение позволяет пользователям загружать изображение через веб-интерфейс, и оно предсказывает класс изображения из набора данных CIFAR-10.

## Возможности

- Загрузка изображения через веб-интерфейс
- Классификация изображения на основе нейросетевой модели (поддерживается обученная модель ResNet50 или простая CNN)
- Отображение предсказанного класса и уверенности
- Сохранение истории предсказаний в базе данных SQLite
- Просмотр истории предсказаний через веб-интерфейс

## Требования

- Python 3.12.0+
- TensorFlow (для машинного обучения)
- FastAPI (для веб-сервера)
- SQLAlchemy (для работы с базой данных)
- Pillow (для обработки изображений)
- Jinja2 (для шаблонизации HTML)

## Установка и запуск

### Шаг 1: Клонируйте репозиторий

```bash
git clone https://github.com/ваш-username/image-classifier-fastapi.git
cd image-classifier-fastapi
```
### Шаг 2: Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
```
### Шаг 3: Установите зависимости
```bash
pip install -r requirements.txt
```
### Шаг 4: Инициализируйте базу данных
```bash
python initdb.py
```
### Шаг 5: Подготовьте модель CIFAR-10
Если обученная модель cifar10_model.h5 отсутствует в директории models/, вы можете создать её заново. Это потребует времени и вычислительных ресурсов. Для этого используйте следующий скрипт:

```bash
import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Загрузка данных CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Предобработка данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Создание модели (простая CNN для быстрого обучения)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение модели
print("Начало обучения...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64)

# Сохранение модели
model.save('models/cifar10_model.h5')
print("✅ Модель сохранена как 'models/cifar10_model.h5'")
```

### Шаг 6: Поместите модель в папку models/:
- Убедитесь, что файл модели cifar10_model.h5 находится в директории models/.
- Если модели нет — обучите её с помощью скрипта:
```bash
python train_model.py
```
### Шаг 7: Запустите сервер
```bash
uvicorn app.main:app --reload
```
### Шаг 8: Откройте в браузере
http://127.0.0.1:8000

### Шаг 9: Тестирование
```bash
pytest tests/ -v
```

### Структура проекта
```bash
image-classifier-fastapi/
│
├── app/                         # Основной модуль приложения
│   ├── main.py                  # Точка входа FastAPI
│   ├── models.py                # Pydantic-схемы
│   ├── database.py              # Работа с SQLite (паттерн Repository)
│   ├── classifier.py            # Логика классификации
│   └── utils.py                 # Декораторы и утилиты
│
├── models/                      # Предобученная модель
│   └── cifar10_model.h5
│
├── static/                      # Статические файлы
│   └── uploads/                 # Загруженные изображения
│
├── templates/                   # HTML-шаблоны (Jinja2)
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── history.html
│
├── tests/                       # Тесты API
│   └── test_api.py
│
├── logs/                        # Логи приложения
│
├── requirements.txt             # Зависимости
├── README.md                    # Инструкция
├── initdb.py                    # Инициализация БД
└── train_model.py               # Обучение модели (опционально)
```
Используемые технологии
FastAPI: Для создания веб-приложения и API.
TensorFlow: Для машинного обучения и работы с моделью классификации изображений.
SQLite: Для хранения истории предсказаний пользователей.
HTML/CSS/Jinja2: Для фронтенда приложения (шаблонизация и отображение страниц).
Pydantic: Для валидации входящих и исходящих данных.
SQLAlchemy: Для работы с базой данных через ORM.
Pillow: Для обработки и предобработки загружаемых изображений.
