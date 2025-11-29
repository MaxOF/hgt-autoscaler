from pydantic import BaseModel
from typing import Dict
from .common_dto import ActionDTO

class PredictResponseDTO(BaseModel):
    actions: Dict[str, ActionDTO]
