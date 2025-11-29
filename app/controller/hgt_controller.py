import os
from fastapi import APIRouter, HTTPException
from app.service.hgt_service import HGTService
from app.service.metrics_service import MetricsService
from app.dto.predict_dto import PredictResponseDTO
from app.dto.train_dto import TrainingRequestDTO, TrainingResponseDTO
from app.dto.apply_dto import ApplyRequestDTO

router = APIRouter(prefix='/api')
hgt_service = HGTService()
metrics_service = MetricsService()

@router.post("/predict", response_model=PredictResponseDTO)
async def predict_resources():
    """Прогнозирование оптимальных ресурсов для всех сервисов"""
    try:
        metrics = await metrics_service.collect_metrics()
        
        predictions = await hgt_service.predict_resources(metrics)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/metrics")
async def get_current_metrics():
    """Получение текущих метрик (для отладки)"""
    try:
        metrics = await metrics_service.collect_metrics()

 
        return metrics.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")
    

@router.post("/train", response_model=TrainingResponseDTO)
async def train_model(training_request: TrainingRequestDTO | None = None):
    """Обучение модели на предоставленных данных"""
    try:
        # Ищем последний сгенерированный CSV файл
        csv_files = [f for f in os.listdir(".") if f.startswith("synthetic_metrics_") and f.endswith(".csv")]
        
        if not csv_files:
            raise HTTPException(status_code=404, detail="No synthetic CSV files found. Generate one first.")
        
        # Берем самый свежий файл
        latest_csv = sorted(csv_files)[-1]
   
        training_request = TrainingRequestDTO(csv_file_path=latest_csv)
        response = await hgt_service.train_from_csv(training_request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/model/load")
async def load_model():
    """Загрузка последней обученной модели"""
    try:
        hgt_service.load_model()
        return {"message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
@router.post("/apply")
async def apply_scaling(
    request: ApplyRequestDTO,
):
    try:
        result = hgt_service.apply(request)
        return {"success": True, "status": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Apply failed: {str(e)}")