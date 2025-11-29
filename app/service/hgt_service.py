# app/service/hgt_service.py
import logging
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from app.core.model import HGTModel, HGTModelConfig
from app.dto.metrics_dto import ServiceMetricsDTO, SystemMetricsDTO
from app.dto.predict_dto import PredictResponseDTO, ActionDTO
from app.dto.train_dto import TrainingRequestDTO, TrainingResponseDTO
from config.settings import settings
from app.core.k8s import K8sActuator
from app.dto.apply_dto import ApplyRequestDTO

logger = logging.getLogger(__name__)

MetaEdge = Tuple[str, str, str]

def _single_graph_collate(batch: List[Any]):
    """
    Коллатер, который возвращает единственный элемент батча без склейки.
    Используем для гетерографов, чтобы edge_index_dict/edge_attr_dict не превратились в списки.
    """
    # batch = [(x_dict, edge_index_dict, edge_attr_dict, targets)]
    return batch[0]

class HGTService:
    def __init__(self):
        self.config = HGTModelConfig(settings, settings.dependencies)
        self.model = HGTModel(self.config)
        self.bounds = settings.bounds
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=settings.model.lr,
            weight_decay=settings.model.weight_decay
        )
        self.criterion = torch.nn.MSELoss()
        self.artifacts_dir = settings.artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.actuator = K8sActuator(
            namespace=settings.kubernetes.namespace,
            label_key=settings.kubernetes.label_key,
        )

    # =======================
    # ======= TRAIN =========
    # =======================
    async def train_from_csv(self, training_request: TrainingRequestDTO) -> TrainingResponseDTO:
        try:
            csv_path = training_request.csv_file_path or settings.file_path
            if not os.path.exists(csv_path):
                return TrainingResponseDTO(success=False, message=f"CSV not found: {csv_path}")

            df = pd.read_csv(csv_path)
            if df.empty:
                return TrainingResponseDTO(success=False, message="CSV is empty")

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            timestamps = df['timestamp'].sort_values().unique()

            samples = []
            for ts in timestamps:
                ts_df = df[df['timestamp'] == ts]
                sys_metrics = self._create_system_metrics_from_df(ts_df, ts)
                optimal = self._create_optimal_resources_from_df(ts_df)
                samples.append({'metrics': sys_metrics, 'optimal_resources': optimal, 'timestamp': ts})

            if not samples:
                return TrainingResponseDTO(success=False, message="No samples produced from CSV")

            split = int(len(samples) * (1 - training_request.validation_split))
            train_samples = samples[:split]
            val_samples = samples[split:]

            # ВАЖНО: батчим по одному графу и не склеиваем
            train_loader = torch.utils.data.DataLoader(
                self._create_dataset(train_samples),
                batch_size=1,
                shuffle=True,
                collate_fn=_single_graph_collate
            )
            val_loader = torch.utils.data.DataLoader(
                self._create_dataset(val_samples),
                batch_size=1,
                shuffle=False,
                collate_fn=_single_graph_collate
            )

            loss_history, val_loss_history = [], []
            epochs = training_request.epochs or settings.model.epochs
     
            for epoch in range(epochs):
                tr_loss = self._train_epoch(train_loader)
                v_loss = self._validate_epoch(val_loader)
                loss_history.append(tr_loss)
                val_loss_history.append(v_loss)
                if epoch % 5 == 0 or epoch == epochs - 1:
                    logger.info(f"[HGT] Epoch {epoch}/{epochs}  train={tr_loss:.4f}  val={v_loss:.4f}")

            model_path = self._save_model()
            return TrainingResponseDTO(
                success=True,
                message="Training completed successfully",
                loss_history=loss_history,
                val_loss_history=val_loss_history,
                model_path=model_path,
                training_samples=len(train_samples),
                validation_samples=len(val_samples)
            )

        except Exception as e:
            logger.exception("Training from CSV failed")
            return TrainingResponseDTO(success=False, message=f"Training failed: {e}")

    def _create_system_metrics_from_df(self, df: pd.DataFrame, ts: datetime) -> SystemMetricsDTO:
        services_metrics = {}
        for _, row in df.iterrows():
            services_metrics[row['service']] = ServiceMetricsDTO(
                svc_total_rps=float(row['svc_total_rps']),
                svc_rps_topk_sum=float(row['svc_rps_topk_sum']),
                svc_rps_long_tail=float(row['svc_rps_long_tail']),
                svc_active_routes=float(row['svc_active_routes']),
                svc_http_p95_ms=float(row['svc_http_p95_ms']),
                dial_failed_rps=float(row['dial_failed_rps']),
                current_replicas=int(row['current_replicas']),
                current_cpu_mcores=int(row['current_cpu_mcores']),
                current_mem_mib=int(row['current_mem_mib'])
            )
        return SystemMetricsDTO(services=services_metrics, timestamp=int(ts.timestamp()))

    def _create_optimal_resources_from_df(self, df: pd.DataFrame) -> Dict[str, Dict]:
        optimal = {}
        for _, row in df.iterrows():
            optimal[row['service']] = {
                'replicas': int(row.get('optimal_replicas', row['current_replicas'])),
                'cpu_mcores': int(row.get('optimal_cpu_mcores', row['current_cpu_mcores'])),
                'mem_mib': int(row.get('optimal_mem_mib', row['current_mem_mib'])),
            }
        return optimal

    def _create_dataset(self, samples: List[Dict]):
        service = self
        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(samples)
            def __getitem__(self, idx):
                s = samples[idx]
                x_dict, edge_index_dict, edge_attr_dict = service.prepare_graph_data(s['metrics'])
                targets = service._prepare_targets(s['optimal_resources'])
                return x_dict, edge_index_dict, edge_attr_dict, targets
        return _DS()

    def _prepare_targets(self, optimal_resources: Dict) -> torch.Tensor:
        services = settings.dependencies.services
        tgt = []
        for svc in services:
            name = svc['name']
            res = optimal_resources.get(name, {})
            tgt.append([
                res.get('replicas', 1),
                res.get('cpu_mcores', 500),
                res.get('mem_mib', 512),
            ])
        return torch.tensor(tgt, dtype=torch.float32)

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        for x_dict, eidx, eattr, y in loader:
            self.optimizer.zero_grad()
            preds = self.model(x_dict, eidx, eattr if self.config.use_edge_weights else None)
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / max(1, len(loader))

    def _validate_epoch(self, loader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for x_dict, eidx, eattr, y in loader:
                preds = self.model(x_dict, eidx, eattr if self.config.use_edge_weights else None)
                loss = self.criterion(preds, y)
                total += loss.item()
        return total / max(1, len(loader))

    def _save_model(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = os.path.join(self.artifacts_dir, f"hgt_model_{ts}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'bounds': self.bounds
        }, p)
        latest = os.path.join(self.artifacts_dir, "hgt_model_latest.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'bounds': self.bounds
        }, latest)
        return p

    def load_model(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(self.artifacts_dir, "hgt_model_latest.pt")
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"No model found at {model_path}, using untrained model")

    # ==========================
    # ======= PREDICT ==========
    # ==========================
    async def predict_resources(self, metrics: SystemMetricsDTO) -> PredictResponseDTO:
        try:
            x_dict, edge_index_dict, edge_attr_dict = self.prepare_graph_data(metrics)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(
                    x_dict, edge_index_dict,
                    edge_attr_dict if self.config.use_edge_weights else None
                )
            actions = self._predictions_to_actions(preds, list(metrics.services.keys()))
            return PredictResponseDTO(actions=actions)
        except Exception as e:
            logger.exception("Prediction failed")
            return self._get_fallback_actions(metrics)

    def _predictions_to_actions(self, predictions: torch.Tensor, service_names: List[str]) -> Dict[str, ActionDTO]:
        actions = {}
        arr = predictions.detach().cpu().numpy()
        for i, name in enumerate(service_names):
            if i >= len(arr): break
            pred = arr[i]
            replicas = int(np.clip(round(float(pred[0])), self.bounds.replicas['min'], self.bounds.replicas['max']))
            cpu      = int(np.clip(round(float(pred[1])), self.bounds.cpu_mcores['min'], self.bounds.cpu_mcores['max']))
            mem      = int(np.clip(round(float(pred[2])), self.bounds.mem_mib['min'], self.bounds.mem_mib['max']))
            actions[name] = ActionDTO(replicas=replicas, cpu_mcores=cpu, mem_mib=mem)
        return actions

    def _get_fallback_actions(self, metrics: SystemMetricsDTO) -> PredictResponseDTO:
        actions = {}
        for name, m in metrics.services.items():
            actions[name] = ActionDTO(
                replicas=m.current_replicas,
                cpu_mcores=m.current_cpu_mcores,
                mem_mib=m.current_mem_mib
            )
        return PredictResponseDTO(actions=actions)

    # =========================================
    # === GRAPH CONSTRUCTION + EDGE ATTR ======
    # =========================================
    def prepare_graph_data(self, metrics: SystemMetricsDTO):
        deps = settings.dependencies
        services = deps.services
        queues = deps.queues
        topics = deps.topics

        # индексы узлов
        svc_idx = {s['name']: i for i, s in enumerate(services)}
        q_idx = {q['name']: i for i, q in enumerate(queues)}
        t_idx = {t['name']: i for i, t in enumerate(topics)}

        # узловые признаки
        service_feats = []
        for s in services:
            m = metrics.services.get(s['name'])
            if m:
                service_feats.append([
                    m.svc_total_rps,
                    m.svc_rps_topk_sum,
                    m.svc_rps_long_tail,
                    m.svc_active_routes,
                    m.svc_http_p95_ms,
                    m.dial_failed_rps,
                    m.current_replicas,
                ])
            else:
                service_feats.append([0.0]*7)

        x_dict = {
            'service': torch.tensor(service_feats, dtype=torch.float32),
            'queue': torch.tensor([[0.0, 1.0] for _ in queues], dtype=torch.float32),
            'topic': torch.tensor([[1.0] for _ in topics], dtype=torch.float32),
        }

        # рёбра
        edge_index_dict: Dict[MetaEdge, torch.Tensor] = {}

        # service -> service
        s2s_edges = []
        for s in services:
            src = svc_idx[s['name']]
            for callee in s.get('calls', []):
                if callee in svc_idx:
                    dst = svc_idx[callee]
                    s2s_edges.append([src, dst])
        if s2s_edges:
            edge_index_dict[('service', 'calls', 'service')] = torch.tensor(s2s_edges, dtype=torch.long).t().contiguous()

        # service -> queue
        s2q_edges = []
        for s in services:
            src = svc_idx[s['name']]
            for q in s.get('queues_out', []):
                if q in q_idx:
                    dst = q_idx[q]
                    s2q_edges.append([src, dst])
        if s2q_edges:
            edge_index_dict[('service', 'produces', 'queue')] = torch.tensor(s2q_edges, dtype=torch.long).t().contiguous()

        # queue -> topic
        q2t_edges = []
        for q in queues:
            src = q_idx[q['name']]
            for t in q.get('topics_out', []):
                if t in t_idx:
                    dst = t_idx[t]
                    q2t_edges.append([src, dst])
        if q2t_edges:
            edge_index_dict[('queue', 'publishes', 'topic')] = torch.tensor(q2t_edges, dtype=torch.long).t().contiguous()

        # topic -> service (условная подписка)
        t2s_edges = []
        for t in topics:
            dst = t_idx[t['name']]
            for q in queues:
                if t['name'] in q.get('topics_out', []):
                    for s in services:
                        if q['name'] in s.get('queues_out', []):
                            src = svc_idx[s['name']]
                            t2s_edges.append([dst, src])
        if t2s_edges:
            edge_index_dict[('topic', 'subscribes', 'service')] = torch.tensor(t2s_edges, dtype=torch.long).t().contiguous()

        # queue -> service (условное потребление)
        q2s_edges = []
        for q in queues:
            src = q_idx[q['name']]
            for s in services:
                if q['name'] in s.get('queues_out', []):
                    dst = svc_idx[s['name']]
                    q2s_edges.append([src, dst])
        if q2s_edges:
            edge_index_dict[('queue', 'consumes', 'service')] = torch.tensor(q2s_edges, dtype=torch.long).t().contiguous()

        # edge_attr_dict (опционально)
        edge_attr_dict: Optional[Dict[MetaEdge, torch.Tensor]] = None
        if self.config.use_edge_weights:
            edge_attr_dict = {}
            def ones_for(edge_tensor: torch.Tensor, edim: int) -> torch.Tensor:
                E = edge_tensor.size(1)
                return torch.ones((E, max(1, edim)), dtype=torch.float32)

            for et, eidx in edge_index_dict.items():
                edim = getattr(settings, "edge_attr_dims", {}).get(et, 1)
                edge_attr_dict[et] = ones_for(eidx, edim)

        return x_dict, edge_index_dict, edge_attr_dict
    
    def apply(self, request: ApplyRequestDTO) -> Dict[str, str]:

        self.actuator.apply(request.actions)

        return {svc: "applied" for svc in request.actions.keys()}