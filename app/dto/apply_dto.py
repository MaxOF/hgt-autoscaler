from typing import Dict
from pydantic import BaseModel

class ApplyActionDTO(BaseModel):
    replicas: int
    cpu_mcores: int
    mem_mib: int

class ApplyRequestDTO(BaseModel):
    actions: Dict[str, ApplyActionDTO]
