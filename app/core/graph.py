from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
import torch

def build_hetero_graph(
    features: Dict[str, Dict[str, float]],
    deps: Dict,
    pods_by_service: Dict[str, int] | None = None,
    nodes: List[Dict] | None = None,
):
    data = HeteroData()
    services = sorted(features.keys())
    svc_index = {s:i for i,s in enumerate(services)}

    # service node features
    svc_feats = []
    print(services)
    for s in services:
        f = features[s]
        svc_feats.append([f.get("cpu_mcores",0.0), f.get("mem_mib",0.0),
                          f.get("rps_in",0.0), f.get("rps_out",0.0),
                          f.get("p95_ms",0.0), f.get("error_rate",0.0)])
    print(svc_feats)
    data["service"].x = torch.tensor(svc_feats, dtype=torch.float32)

    # queues / topics
    queues = [q["name"] for q in deps.get("queues",[])]
    topics = [t["name"] for t in deps.get("topics",[])]
    data["queue"].x = torch.zeros((len(queues),1))
    data["topic"].x = torch.zeros((len(topics),1))
    q_idx = {q:i for i,q in enumerate(queues)}
    t_idx = {t:i for i,t in enumerate(topics)}

    # pods / nodes
    pods_by_service = pods_by_service or {s:1 for s in services}
    pod_count = sum(pods_by_service.values())
    data["pod"].x = torch.zeros((pod_count,1))
    node_names = [n.get("name","node") for n in (nodes or [])]
    data["node"].x = torch.zeros((len(node_names),1))
    n_idx = {n:i for i,n in enumerate(node_names)}

    # edges
    # service->service
    s2s_src, s2s_dst = [], []
    for d in deps.get("services", []):
        u = d["name"]
        if u not in svc_index: 
            continue
        for v in d.get("calls", []):
            if v in svc_index:
                s2s_src.append(svc_index[u]); s2s_dst.append(svc_index[v])
    if s2s_src:
        data[("service","calls","service")].edge_index = torch.tensor([s2s_src, s2s_dst])

    # service->queue
    s2q_src, s2q_dst = [], []
    for d in deps.get("services", []):
        u = d["name"]
        for q in d.get("queues_out", []):
            if u in svc_index and q in q_idx:
                s2q_src.append(svc_index[u]); s2q_dst.append(q_idx[q])
    if s2q_src:
        data[("service","produces","queue")].edge_index = torch.tensor([s2q_src, s2q_dst])

    # queue->topic
    q2t_src, q2t_dst = [], []
    for q in deps.get("queues", []):
        qn = q["name"]
        for t in q.get("topics_out", []):
            if qn in q_idx and t in t_idx:
                q2t_src.append(q_idx[qn]); q2t_dst.append(t_idx[t])
    if q2t_src:
        data[("queue","publishes","topic")].edge_index = torch.tensor([q2t_src, q2t_dst])

    # service->pod
    s2p_src, s2p_dst = [], []
    cursor = 0
    for s in services:
        k = pods_by_service.get(s,1)
        for j in range(k):
            s2p_src.append(svc_index[s]); s2p_dst.append(cursor+j)
        cursor += k
    if s2p_src:
        data[("service","owns","pod")].edge_index = torch.tensor([s2p_src, s2p_dst])

    # pod->node (round-robin)
    if len(node_names) > 0 and pod_count > 0:
        p2n_src = list(range(pod_count))
        p2n_dst = [i % len(node_names) for i in range(pod_count)]
        data[("pod","scheduled_on","node")].edge_index = torch.tensor([p2n_src, p2n_dst])

    return data, services
