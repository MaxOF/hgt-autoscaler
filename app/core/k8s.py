from typing import Dict
import logging

from kubernetes import client, config
from kubernetes.client import AppsV1Api, CoreV1Api, V1Deployment, V1Scale

logger = logging.getLogger(__name__)


class K8sActuator:
    """
    Применяет решения автоскейлера к Kubernetes:
    - горизонтальное масштабирование (replicas)
    - вертикальное масштабирование CPU/Memory для контейнера(ов)
    """

    def __init__(self, namespace: str, label_key: str):
        self.ns = namespace
        self.key = label_key

        # Инициализация Kubernetes клиента:
        # - внутри кластера: config.load_incluster_config()
        # - локально (dev): config.load_kube_config()
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except Exception:
            config.load_kube_config()
            logger.info("Loaded local kubeconfig")

        self.apps_api: AppsV1Api = client.AppsV1Api()
        self.core_api: CoreV1Api = client.CoreV1Api()

    def apply(self, actions: Dict[str, Dict[str, int]]):
        for svc, a in actions.items():
            try:
                replicas = int(a["replicas"])
                cpu_mcores = int(a["cpu_mcores"])
                mem_mib = int(a["mem_mib"])
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"[K8s] invalid action payload for {svc}: {a} ({e})")
                continue

            logger.info(
                f"[K8s] applying action for {svc}: "
                f"replicas={replicas}, cpu={cpu_mcores}m, mem={mem_mib}Mi"
            )

            try:
                self._apply_for_service(
                    service_name=svc,
                    replicas=replicas,
                    cpu_mcores=cpu_mcores,
                    mem_mib=mem_mib,
                )
            except Exception as e:
                logger.exception(f"[K8s] failed to apply scaling for {svc}: {e}")

    def _apply_for_service(
        self,
        service_name: str,
        replicas: int,
        cpu_mcores: int,
        mem_mib: int,
    ):
        # 1. Находим Deployment по лейблу, например: app=orders-service
        label_selector = f"{self.key}={service_name}"
        deployments = self.apps_api.list_namespaced_deployment(
            namespace=self.ns,
            label_selector=label_selector,
        )

        if not deployments.items:
            logger.warning(
                f"[K8s] no deployments found for {service_name} "
                f"(label_selector={label_selector})"
            )
            return

        # В большинстве случаев ожидаем один deployment на сервис
        deployment: V1Deployment = deployments.items[0]
        dep_name = deployment.metadata.name

        # 2. Горизонтальное масштабирование (replicas) через /scale
        self._scale_deployment(dep_name, replicas)

        # 3. Вертикальное масштабирование CPU/Memory через patch Deployment
        self._patch_deployment_resources(dep_name, cpu_mcores, mem_mib)

    def _scale_deployment(self, deployment_name: str, replicas: int):
        """
        Обновление spec.replicas через subresource /scale.
        """
        body = {"spec": {"replicas": replicas}}
        logger.debug(f"[K8s] scaling deployment {deployment_name} to {replicas} replicas")

        self.apps_api.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=self.ns,
            body=body,
        )

    def _patch_deployment_resources(
        self,
        deployment_name: str,
        cpu_mcores: int,
        mem_mib: int,
    ):
        """
        Патчим ресурсы для всех контейнеров в pod template Deployment'а.
        Для простоты — одинаковые лимиты/requests для всех контейнеров.
        """
        cpu_str = f"{cpu_mcores}m"
        mem_str = f"{mem_mib}Mi"

        # Патчим только поля ресурсов, остальное не трогаем
        patch_body = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": None,
                                "resources": {
                                    "requests": {
                                        "cpu": cpu_str,
                                        "memory": mem_str,
                                    },
                                    "limits": {
                                        "cpu": cpu_str,
                                        "memory": mem_str,
                                    },
                                },
                            }
                        ]
                    }
                }
            }
        }

        # Чтобы безопасно патчить по именам контейнеров, читаем текущий деплоймент:
        dep = self.apps_api.read_namespaced_deployment(
            name=deployment_name,
            namespace=self.ns,
        )

        containers = dep.spec.template.spec.containers
        containers_patch = []
        for c in containers:
            containers_patch.append(
                {
                    "name": c.name,
                    "resources": {
                        "requests": {"cpu": cpu_str, "memory": mem_str},
                        "limits": {"cpu": cpu_str, "memory": mem_str},
                    },
                }
            )

        patch_body["spec"]["template"]["spec"]["containers"] = containers_patch

        logger.debug(
            f"[K8s] patching resources for deployment {deployment_name}: "
            f"cpu={cpu_str}, mem={mem_str}"
        )

        self.apps_api.patch_namespaced_deployment(
            name=deployment_name,
            namespace=self.ns,
            body=patch_body,
        )
