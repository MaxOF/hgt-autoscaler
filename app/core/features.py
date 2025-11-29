from prometheus_api_client import PrometheusConnect
import pandas as pd
from typing import Dict

class FeatureCollector:
    def __init__(self, url: str):
        self.prom = PrometheusConnect(url=url, disable_ssl=True)

    def _query(self, ql: str) -> pd.DataFrame:
        res = self.prom.custom_query(ql)
        rows = []
        for r in res:
            metric = r.get("metric", {})
            svc = metric.get("service") or metric.get("app") or metric.get("pod") or "unknown"
            value = float(r["value"][1])
            rows.append({"service": svc, "value": value})
        return pd.DataFrame(rows)

    def collect(self, queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        out = {"prometheus": {}}
        for key, ql in queries.items():
            res = self._query(ql)  # DataFrame с колонками (зависит от ответа)
            val = None
            if "value" in res.columns and len(res):     # vector
                # если несколько строк (по handler/dialer_name), суммируем
                val = float(res["value"].sum())
            elif "histogram" in ql or "histogram_quantile" in ql or "quantile_over_time" in ql:
                # _query уже вернёт vector → см. выше
                val = float(res["value"].sum()) if len(res) else 0.0
            else:
                val = float(res["value"].sum()) if "value" in res and len(res) else 0.0

            out["prometheus"][key] = 0.0 if val is None else val

        print(out)
        return out

