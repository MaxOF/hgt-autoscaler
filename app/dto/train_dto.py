from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime

class TrainingSampleDTO(BaseModel):
    metrics: Dict[str, Any]  # Входные метрики
    optimal_resources: Dict[str, Any]  # Оптимальные ресурсы (target)
    timestamp: datetime

class TrainingRequestDTO(BaseModel):
    csv_file_path: str  # Путь к CSV файлу
    epochs: Optional[int] = None
    validation_split: float = 0.2
    batch_size: int = 32

class TrainingResponseDTO(BaseModel):
    success: bool
    message: str
    loss_history: List[float] = []
    val_loss_history: List[float] = []
    model_path: str = None
    training_samples: int = 0
    validation_samples: int = 0