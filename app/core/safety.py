import numpy as np
from typing import Dict, List

class SafetyPolicy:
    def __init__(self, bounds, hysteresis_windows=2, max_replica_step=2, cooldown_steps=1):
        self.b = bounds
        self.H = hysteresis_windows
        self.step = max_replica_step
        self.cool = cooldown_steps
        self.mem: Dict[str, Dict] = {}

    def _clip_all(self, r, cpu, mem):
        r = int(max(self.b["replicas"]["min"], min(self.b["replicas"]["max"], round(r))))
        cpu = int(max(self.b["cpu_mcores"]["min"], min(self.b["cpu_mcores"]["max"], cpu)))
        mem = int(max(self.b["mem_mib"]["min"],   min(self.b["mem_mib"]["max"],   mem)))
        return r, cpu, mem

    def decide(self, services: List[str], pred_tensor):
        out: Dict[str, Dict[str,int]] = {}
        for i, name in enumerate(services):
            r, cpu, mem = [float(x) for x in pred_tensor[i].tolist()]
            r, cpu, mem = self._clip_all(r, cpu, mem)
            st = self.mem.setdefault(name, {"last":[r,cpu,mem], "cnt":0, "cool":0})
            last = st["last"]

            if st["cool"] > 0:
                st["cool"] -= 1
                out[name] = {"replicas": last[0], "cpu_mcores": last[1], "mem_mib": last[2]}
                continue

            changed = [r,cpu,mem] != last
            st["cnt"] = st["cnt"] + 1 if changed else 0

            if st["cnt"] >= self.H:
                # ограничение шага по репликам
                delta = r - last[0]
                if abs(delta) > self.step:
                    r = last[0] + int(np.sign(delta) * self.step)
                st["last"] = [r,cpu,mem]
                st["cnt"] = 0
                st["cool"] = self.cool

            out[name] = {"replicas": st["last"][0], "cpu_mcores": st["last"][1], "mem_mib": st["last"][2]}
        return out
