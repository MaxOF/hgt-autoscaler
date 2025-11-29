from pydantic import BaseModel
from typing import Dict, List, Optional

class ServiceMetricsDTO(BaseModel):
    # Основные метрики нагрузки
    svc_total_rps: float  # Общий RPS по сервису
    svc_rps_topk_sum: float  # Сумма RPS топ-5 маршрутов
    svc_rps_long_tail: float  # RPS вне топ-5 ("хвост")
    svc_active_routes: float  # Количество активных маршрутов
    svc_http_p95_ms: float  # p95 латентности
    
    # Дополнительные сигналы
    dial_failed_rps: Optional[float] = 0.0  # Неуспешные коннекты
    
    # Текущие ресурсы (из Kubernetes)
    current_replicas: int
    current_cpu_mcores: int
    current_mem_mib: int

class SystemMetricsDTO(BaseModel):
    services: Dict[str, ServiceMetricsDTO]  # {service_name: metrics}
    timestamp: int