from dataclasses import dataclass, field
from typing import Dict, List
from types import SimpleNamespace

@dataclass
class PrometheusCfg:
    url: str = "http://localhost:9090"
    scrape_step: str = "15s"
    queries = {
        # === HTTP НАГРУЗКА (пер-сервис) ==========================================
        # Общий RPS по каждому сервису
        "svc_total_rps": 'sum by (instance) (rate(requests_total_by_service[1m]))',

        # Сумма RPS топ-5 самых "тяжёлых" маршрутов внутри каждого сервиса
        "svc_rps_topk_sum": 
            'sum by (instance) (topk(5, sum by (instance, route) (rate(requests_total_by_service[1m]))))',

        # "Длина хвоста": всё, что вне топ-5 (пер-сервис)
        "svc_rps_long_tail":
            'sum by (instance) (rate(requests_total_by_service[1m])) - sum by (instance) (topk(5, sum by (instance, route) (rate(requests_total_by_service[1m]))))',

        # Количество активных маршрутов (сколько route > 0 RPS) внутри сервиса
        "svc_active_routes":
            'sum by (instance) ((sum by (instance, route) (rate(requests_total_by_service[1m])) > bool 0))',

        # p95 латентности по каждому сервису (агрегация по всем маршрутам сервиса)
        "svc_http_p95_ms":
            'histogram_quantile(0.95, sum by (le, instance) (rate(requests_duration_in_seconds_by_service_bucket[2m]))) * 1000',

        # === ДОП. СИГНАЛЫ (оставляем, если нужны для стабильности решения) =======
        # Неуспешные исходящие коннекты (в целом по инстансу — пригодится как экзогенный шум сети)
        "dial_failed_rps": 'sum(rate(net_conntrack_dialer_conn_failed_total[1m]))',

        # Маска применения действий (если автоскейлер развёрнут как сервис)
        "ready": "prometheus_ready"
    }


@dataclass
class DependenciesCfg:
    services: List[Dict] = field(default_factory=lambda: [
        {"name": "orders-service", "calls": ["products-service","payments-service"], "queues_out":["orders-events"]},
        {"name": "products-service", "calls": []},
        {"name": "payments-service", "calls": []},
    ])
    queues: List[Dict] = field(default_factory=lambda: [
        {"name": "orders-events", "topics_out": ["billing-topic"]}
    ])
    topics: List[Dict] = field(default_factory=lambda: [
        {"name": "billing-topic"}
    ])

@dataclass
class BoundsCfg:
    replicas: Dict[str, int] = field(default_factory=lambda: {"min": 1, "max": 50})
    cpu_mcores: Dict[str, int] = field(default_factory=lambda: {"min": 50, "max": 4000})
    mem_mib: Dict[str, int] = field(default_factory=lambda: {"min": 64, "max": 8192})

@dataclass
class SafetyCfg:
    hysteresis_windows: int = 2
    max_replica_step: int = 2
    cooldown_steps: int = 1

@dataclass
class ModelCfg:
    type: str = "HGT"
    hidden_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100

@dataclass
class K8sCfg:
    namespace: str = "default"
    deployment_label_key: str = "app.kubernetes.io/name"

@dataclass
class Settings:
    decision_interval_seconds: int = 300
    prometheus: PrometheusCfg = field(default_factory=PrometheusCfg)
    dependencies = SimpleNamespace(
        services=[
            {"name": "orders-service", "calls": ["payments-service"], "queues_out": ["q-orders"]},
            {"name": "payments-service", "calls": [], "queues_out": ["q-payments"]},
            {"name": "products-service", "calls": ["orders-service"], "queues_out": []},
        ],
        queues=[{"name": "q-orders", "topics_out": ["t-orders"]},
                {"name": "q-payments", "topics_out": ["t-payments"]}],
        topics=[{"name": "t-orders"}, {"name": "t-payments"}],
    )
    nodes: List[Dict] = field(default_factory=lambda: [{"name":"node-a"},{"name":"node-b"}])
    bounds = SimpleNamespace(
        replicas={"min": 1, "max": 50},
        cpu_mcores={"min": 50, "max": 8000},
        mem_mib={"min": 64, "max": 32768},
    )
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    model = SimpleNamespace(
        lr=1e-3,
        weight_decay=1e-4,
        epochs=2,
        use_edge_weights=True,                
        hidden_channels=64,
        heads=2,
        dropout=0.1,
    )
    artifacts_dir: str = "./artifacts"
    kubernetes: K8sCfg = field(default_factory=K8sCfg)
    file_path: str | None = "./synthetic_metrics_20251110_103408.csv"
    edge_attr_dims = {
        ("service", "calls", "service"): 1,     # например, вес вызова
        ("service", "produces", "queue"): 1,
        ("queue", "publishes", "topic"): 1,
        ("topic", "subscribes", "service"): 1,
        ("queue", "consumes", "service"): 1,
    }

settings = Settings()
