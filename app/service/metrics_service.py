import logging
from typing import Dict, List
from datetime import datetime
from prometheus_api_client import PrometheusConnect
from app.dto.metrics_dto import SystemMetricsDTO, ServiceMetricsDTO
from config.settings import settings

logger = logging.getLogger(__name__)

class MetricsService:
    def __init__(self):
        self.prometheus = PrometheusConnect(
            url=settings.prometheus.url,
            disable_ssl=True
        )
        self.queries = settings.prometheus.queries
        self.dependencies = settings.dependencies
    
    async def collect_metrics(self) -> SystemMetricsDTO:
        """Сбор всех метрик для всех сервисов из конфига"""
        service_names = [service["name"] for service in self.dependencies.services]
        service_metrics = {}
        
        for service_name in service_names:
            try:
                metrics = await self._collect_service_metrics(service_name)
                service_metrics[service_name] = metrics
            except Exception as e:
                logger.error(f"Failed to collect metrics for {service_name}: {e}")
                # Заполняем метрики по умолчанию в случае ошибки
                service_metrics[service_name] = await self._get_default_metrics(service_name)
        
        return SystemMetricsDTO(
            services=service_metrics,
            timestamp=int(datetime.now().timestamp())
        )
    
    async def _collect_service_metrics(self, service_name: str) -> ServiceMetricsDTO:
        """Сбор метрик для конкретного сервиса"""
        # Получаем текущие ресурсы из Kubernetes
        current_resources = await self._get_current_resources(service_name)
        
        # Сбор метрик нагрузки из Prometheus
        metrics_data = {}
        for metric_name, query in self.queries.items():
            try:
                # Подставляем имя сервиса в запрос
                formatted_query = query.replace("instance", f'"{service_name}"')
                result = self.prometheus.custom_query(query=formatted_query)
                value = self._extract_metric_value(result, metric_name)
                metrics_data[metric_name] = value
                logger.debug(f"Metric {metric_name} for {service_name}: {value}")
            except Exception as e:
                logger.warning(f"Failed to get {metric_name} for {service_name}: {e}")
                metrics_data[metric_name] = 0.0
        
        return ServiceMetricsDTO(
            svc_total_rps=metrics_data.get("svc_total_rps", 0.0),
            svc_rps_topk_sum=metrics_data.get("svc_rps_topk_sum", 0.0),
            svc_rps_long_tail=metrics_data.get("svc_rps_long_tail", 0.0),
            svc_active_routes=metrics_data.get("svc_active_routes", 0.0),
            svc_http_p95_ms=metrics_data.get("svc_http_p95_ms", 0.0),
            dial_failed_rps=metrics_data.get("dial_failed_rps", 0.0),
            **current_resources
        )
    
    async def _get_current_resources(self, service_name: str) -> Dict:
        """Получение текущих ресурсов сервиса из Kubernetes"""
        # TODO: Реализовать получение из Kubernetes API
        # Временная заглушка - в реальности нужно получать из K8s API
        try:
            # Здесь будет запрос к Kubernetes API для получения текущих ресурсов
            # deployment = k8s_apps_v1.read_namespaced_deployment(name=service_name, namespace=settings.kubernetes.namespace)
            # current_replicas = deployment.spec.replicas
            # resources = deployment.spec.template.spec.containers[0].resources
            
            # Временные значения для демонстрации
            return {
                "current_replicas": 2,
                "current_cpu_mcores": 500,
                "current_mem_mib": 512
            }
        except Exception as e:
            logger.warning(f"Failed to get current resources for {service_name}: {e}")
            return {
                "current_replicas": 1,
                "current_cpu_mcores": 500,
                "current_mem_mib": 512
            }
    
    def _extract_metric_value(self, prometheus_result, metric_name: str) -> float:
        """Извлечение значения из ответа Prometheus"""
        if not prometheus_result:
            logger.debug(f"No data for metric {metric_name}")
            return 0.0
        
        try:
            # Prometheus возвращает список результатов, берем первый
            result = prometheus_result[0]
            value = float(result['value'][1])
            logger.debug(f"Extracted value for {metric_name}: {value}")
            return value
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to extract value for {metric_name}: {e}, result: {prometheus_result}")
            return 0.0
    
    async def _get_default_metrics(self, service_name: str) -> ServiceMetricsDTO:
        """Метрики по умолчанию при ошибке сбора"""
        current_resources = await self._get_current_resources(service_name)
        return ServiceMetricsDTO(
            svc_total_rps=0.0,
            svc_rps_topk_sum=0.0,
            svc_rps_long_tail=0.0,
            svc_active_routes=0.0,
            svc_http_p95_ms=0.0,
            dial_failed_rps=0.0,
            **current_resources
        )
    
    def get_service_dependencies(self, service_name: str) -> List[str]:
        """Получение зависимостей сервиса"""
        for service in self.dependencies.services:
            if service["name"] == service_name:
                return service.get("calls", [])
        return []
    
    def get_all_services(self) -> List[str]:
        """Получение списка всех сервисов"""
        return [service["name"] for service in self.dependencies.services]