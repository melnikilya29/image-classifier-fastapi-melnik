import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Загрузка данных
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Подготовка данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Создание модели
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

# Обучение
print("Начало обучения (это может занять время)...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64)

# Сохранение
model.save('models/cifar10_model.h5')
print("✅ Модель сохранена как 'models/cifar10_model.h5'")