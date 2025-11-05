import time
import os
from functools import wraps

def log_time(func):
    """Декоратор: логирует время выполнения функции"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️ {func.__name__} завершена за {time.time() - start:.2f} сек")
        return result
    return wrapper

def validate_confidence(func):
    """Декоратор: проверяет, что уверенность в [0, 1]"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        label, conf = func(*args, **kwargs)
        if not (0.0 <= conf <= 1.0):
            raise ValueError("Уверенность должна быть от 0 до 1")
        return label, conf
    return wrapper

def retry(max_attempts=3):
    """Декоратор: повторяет вызов при ошибке"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"🔁 Попытка {attempt + 1} не удалась. Повтор...")
            return None
        return wrapper
    return decorator

async def save_uploaded_file(file):
    """Сохраняет загруженный файл в static/uploads/"""
    os.makedirs("static/uploads", exist_ok=True)
    path = f"static/uploads/{file.filename}"
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)
    return path