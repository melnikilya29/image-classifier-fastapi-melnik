import tensorflow as tf
from PIL import Image
import numpy as np
import os
from .utils import validate_confidence

def load_model():
    model_path = "models/cifar10_model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Модель не найдена. Поместите файл cifar10_model.h5 в папку models/")
    return tf.keras.models.load_model(model_path)

@validate_confidence
def predict_class(model, image_path):
    # Загрузка и предобработка
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Предсказание
    predictions = model.predict(img_batch)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    return classes[class_idx], confidence