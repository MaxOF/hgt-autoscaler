# app/core/model.py
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
from typing import Dict, Tuple, Optional

MetaEdge = Tuple[str, str, str]

class HGTModelConfig:
    def __init__(self, settings, dependencies):
        self.hidden = settings.model.hidden_channels
        self.heads = settings.model.heads
        self.dropout = settings.model.dropout
        self.use_edge_weights = settings.model.use_edge_weights
        self.dependencies = dependencies
        # Метаданные гетерографа
        self.node_types = ['service', 'queue', 'topic']
        self.edge_types: Tuple[MetaEdge, ...] = tuple({
            ('service', 'calls', 'service'),
            ('service', 'produces', 'queue'),
            ('queue', 'publishes', 'topic'),
            ('topic', 'subscribes', 'service'),
            ('queue', 'consumes', 'service'),
        })
        # Размерности edge_attr по типам (если есть)
        self.edge_attr_dims: Dict[MetaEdge, int] = getattr(settings, "edge_attr_dims", {})

class HGTModel(nn.Module):
    def __init__(self, config: HGTModelConfig):
        super().__init__()
        self.cfg = config
        H = self.cfg.hidden

        # Входные размерности признаков узлов:
        # service: 7 признаков (как у тебя)
        # queue: 2 (заглушки)
        # topic: 1 (заглушки)
        self.in_dims = {'service': 7, 'queue': 2, 'topic': 1}

        # Линейные энкодеры узлов до общего hidden размера
        self.node_enc = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(self.in_dims[ntype], H),
                nn.ReLU(),
                nn.Dropout(self.cfg.dropout)
            )
            for ntype in self.cfg.node_types
        })

        # Если edge_attr будут, у каждого типа ребра свой encoder → H
        if self.cfg.use_edge_weights:
            self.edge_enc = nn.ModuleDict()
            for et in self.cfg.edge_types:
                edim = self.cfg.edge_attr_dims.get(et, 0)
                if edim and edim > 0:
                    self.edge_enc[str(et)] = nn.Linear(edim, H)
                else:
                    # Заглушка на случай отсутствия реальной размерности
                    self.edge_enc[str(et)] = nn.Linear(1, H)
        else:
            self.edge_enc = None

        # Один-два слоя HGTConv (можно нарастить)
        self.hgt1 = HGTConv(
            in_channels=H,
            out_channels=H,
            metadata=(self.cfg.node_types, self.cfg.edge_types),
            heads=self.cfg.heads
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.cfg.dropout)

        self.hgt2 = HGTConv(
            in_channels=H,
            out_channels=H,
            metadata=(self.cfg.node_types, self.cfg.edge_types),
            heads=self.cfg.heads
        )

        # Readout для сервисов → предсказываем [replicas, cpu_mcores, mem_mib]
        self.out_head = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, 3)
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[MetaEdge, torch.Tensor],
        edge_attr_dict: Optional[Dict[MetaEdge, torch.Tensor]] = None
    ) -> torch.Tensor:

        # 1) Encode node features
        h = {nt: self.node_enc[nt](x) for nt, x in x_dict.items()}

        # 2) (Опционально) Вплавляем рёберные признаки в узлы:
        #    Для каждого типа ребра агрегируем edge_attr на источнике и на приёмнике
        #    и добавляем relation-aware вклад в h[source] и/или h[target].
        if self.cfg.use_edge_weights and edge_attr_dict is not None and self.edge_enc is not None:
            for et, ei in edge_index_dict.items():
                key = str(et)
                if key not in self.edge_enc:
                    continue
                if et not in edge_attr_dict:
                    continue

                eattr = edge_attr_dict[et]  # shape: [E, edge_feat_dim]
                proj = self.edge_enc[key](eattr)  # [E, H]

                src_ntype, _, dst_ntype = et
                src, dst = ei  # [2, E]

                # Суммируем вклад инцидентных ребёр в эмбеддинги узлов (residual edge-aware bias)
                # На источнике:
                if src_ntype in h:
                    add_src = torch.zeros_like(h[src_ntype])
                    add_src.index_add_(0, src, proj)  # суммарный сигнал по исходящим
                    h[src_ntype] = h[src_ntype] + self.drop(add_src)

                # На приёмнике:
                if dst_ntype in h:
                    add_dst = torch.zeros_like(h[dst_ntype])
                    add_dst.index_add_(0, dst, proj)  # суммарный сигнал по входящим
                    h[dst_ntype] = h[dst_ntype] + self.drop(add_dst)

        # 3) HGT слой 1
        h = self.hgt1(h, edge_index_dict)
        h = {nt: self.drop(self.act(x)) for nt, x in h.items()}

        # 4) HGT слой 2
        h = self.hgt2(h, edge_index_dict)
        h = {nt: self.drop(self.act(x)) for nt, x in h.items()}

        # 5) Head по сервисам
        service_emb = h['service']                        # [num_services, H]
        out = self.out_head(service_emb)                  # [num_services, 3]
        return out
