from fastapi import FastAPI, Request, File, UploadFile, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import os

from . import database, classifier, utils
from .database import get_db

app = FastAPI(title="CIFAR-10 Image Classifier", description="FastAPI-версия классификатора изображений")

# Подключение статики и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Загрузка модели при старте
model = classifier.load_model()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Сохранение файла
        file_path = await utils.save_uploaded_file(file)
        
        # Классификация
        predicted_class, confidence = classifier.predict_class(model, file_path)
        
        # Сохранение в БД
        database.create_prediction(db, file.filename, predicted_class, confidence)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2%}",
            "image_url": f"/static/uploads/{file.filename}"
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/history/", response_class=HTMLResponse)
def history(request: Request, db: Session = Depends(get_db)):
    predictions = database.get_all_predictions(db)
    return templates.TemplateResponse("history.html", {
        "request": request,
        "predictions": predictions
    })