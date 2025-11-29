# HGT-Autoscaler  
Heterogeneous Graph Transformer for Kubernetes Autoscaling

---

## 📌 Overview

**HGT-Autoscaler** — это продвинутый autoscaling-контроллер для Kubernetes,  
использующий **Heterogeneous Graph Transformer (HGT)** для прогнозирования:

- оптимального числа реплик,
- CPU-лимитов,
- Memory-лимитов

для каждого микросервиса.  
Модель учитывает **граф зависимостей микросервисов**: сервисы → очереди → топики  
и использует **Prometheus-метрики как признаки узлов и ребер графа**.

Проект включает:

- FastAPI сервис (`/train`, `/predict`, `/apply`)
- HGT-модель на PyTorch Geometric
- CSV-тренировку на синтетических или реальных метриках
- K8s actuator (создание масштабирования через Kubernetes API)

---

## 🏗 Архитектура


::contentReference[oaicite:0]{index=0}


*(Вставьте сюда свой файл: `docs/system-architecture.pdf`)*

Архитектура состоит из шести ключевых подсистем:

1. **Prometheus Metrics Collector**  
   Получает метрики сервисов, очередей, маршрутов, p95 latency и др.

2. **DTO Normalization Layer**  
   Преобразует метрики в нормализованные векторы.

3. **Graph Builder**  
   Создаёт гетерогенный граф:
   - узлы: сервисы, очереди, топики  
   - связи: calls, produces, publishes, subscribes, consumes  
   - веса рёбер: RPS, throughput, backlog

4. **HGT Inference Engine**  
   PyTorch-модель на базе HGT.

5. **Decision & Safety Layer**  
   - hysteresis  
   - cooldown  
   - bounding  
   - prediction smoothing

6. **K8s Actuator**  
   Применяет пересчёт ресурсов через *Kubernetes Python Client*.

---

## 📊 Features

- Полная поддержка **гетерогенных графов** (service–queue–topic)
- Прогнозирование:
  - replicas
  - CPU mCores
  - Memory MiB
- Реал-тайм inference каждые 5 минут
- Поддержка synthetic workload генератора
- CSV training loader
- Мягкая интеграция с Kubernetes API:
  - Scale (apps/v1)
  - Patch resources (SSA)

---

## 📂 Project Structure

```text
app/
 ├─ api/
 │   ├─ controller.py
 │   └─ dto/
 ├─ core/
 │   ├─ model.py         # HGT model
 │   └─ layers.py
 ├─ service/
 │   ├─ hgt_service.py   # training + predict logic
 │   ├─ k8s_actuator.py  # scaling actuator
 │   └─ graph_builder.py
 ├─ utils/
 │   ├─ csv_loader.py
 │   └─ normalizer.py
config/
 ├─ settings.py
 └─ dependencies.py
synthetic/
 └─ generate_metrics.py


##Installation

```bash
git clone https://github.com/MaxOF/repo.git
cd hgt-autoscaler

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Running the Autoscaler

```bash
uvicorn app.main:app --reload
```