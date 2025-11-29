from pydantic import BaseModel
from typing import Dict, List

class ActionDTO(BaseModel):
    replicas: int
    cpu_mcores: int
    mem_mib: int

class ServiceActionDTO(BaseModel):
    service: str
    action: ActionDTO

class ActionsResponseDTO(BaseModel):
    actions: Dict[str, ActionDTO]  # {service: {replicas,cpu_mcores,mem_mib}}
