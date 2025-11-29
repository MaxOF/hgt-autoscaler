import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_metrics():
    """Генерация синтетических метрик за 1 месяц"""
    
    # Начальная дата - 1 месяц назад
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    # Создаем временной ряд с интервалом 5 минут (как decision_interval_seconds)
    date_range = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    data = []
    
    for timestamp in date_range:
        hour = timestamp.hour
        day_type = 1.0 if timestamp.weekday() < 5 else 0.7  # Будни/выходные
        
        # Базовая нагрузка + пики
        base_load = 10.0
        peak_multiplier = 1.0
        
        # Пиковые часы: 14-16 и 19-23
        if 10 <= hour < 11:
            peak_multiplier = 3.5  # Дневной пик
        elif 19 <= hour < 23:
            peak_multiplier = 4.0  # Вечерний пик
        elif 8 <= hour < 10:
            peak_multiplier = 1.8  # Утренний подъем
        elif 12 <= hour < 14:
            peak_multiplier = 2.5  # Обеденное время
        
        # Сезонность - более высокая нагрузка в определенные дни
        day_of_month_effect = 1.0 + 0.2 * np.sin(timestamp.day * 2 * np.pi / 30)
        
        # Случайные всплески (5% вероятности)
        random_spike = 1.0
        if random.random() < 0.05:
            random_spike = 2.0 + random.random() * 3.0
        
        total_multiplier = peak_multiplier * day_type * day_of_month_effect * random_spike
        
        # Генерация метрик для каждого сервиса
        for service in ['orders-service', 'payments-service', 'products-service']:
            
            # Базовая нагрузка зависит от сервиса
            if service == 'orders-service':
                base_rps = 50.0
                latency_base = 45.0
            elif service == 'payments-service':
                base_rps = 30.0
                latency_base = 80.0
            else:  # products-service
                base_rps = 20.0
                latency_base = 25.0
            
            # RPS метрики
            total_rps = base_rps * total_multiplier * (0.8 + random.random() * 0.4)
            topk_rps = total_rps * (0.6 + random.random() * 0.3)  # 60-90% нагрузки в топ-5
            long_tail = max(0, total_rps - topk_rps)
            active_routes = max(1, int(5 + random.random() * 10))  # 5-15 активных маршрутов
            
            # Латентность зависит от нагрузки
            latency_multiplier = 1.0 + (total_rps / (base_rps * 4)) * 2.0
            p95_latency = latency_base * latency_multiplier * (0.8 + random.random() * 0.4)
            
            # Failed connections - редкие ошибки
            failed_connections = random.random() * 0.1  # 0-0.1 RPS
            
            # Текущие ресурсы (будут меняться реже)
            current_replicas = max(1, int(total_rps / (base_rps * 1.5)) + random.randint(-1, 1))
            current_cpu = max(100, int(total_rps * 10 + random.random() * 200))
            current_memory = max(128, int(total_rps * 5 + random.random() * 100))
            
            # Оптимальные ресурсы (target для обучения)
            optimal_replicas = max(1, int(total_rps / (base_rps * 0.8)) + 1)
            optimal_cpu = max(100, int(total_rps * 8 + random.random() * 150))
            optimal_memory = max(128, int(total_rps * 4 + random.random() * 80))
            
            data.append({
                'timestamp': timestamp,
                'service': service,
                'svc_total_rps': total_rps,
                'svc_rps_topk_sum': topk_rps,
                'svc_rps_long_tail': long_tail,
                'svc_active_routes': active_routes,
                'svc_http_p95_ms': p95_latency,
                'dial_failed_rps': failed_connections,
                'current_replicas': current_replicas,
                'current_cpu_mcores': current_cpu,
                'current_mem_mib': current_memory,
                'optimal_replicas': optimal_replicas,
                'optimal_cpu_mcores': optimal_cpu,
                'optimal_mem_mib': optimal_memory,
                'hour': hour,
                'day_of_week': timestamp.weekday(),
                'is_peak_hours': 1 if (10 <= hour < 11) or (19 <= hour < 23) else 0
            })
    
    return pd.DataFrame(data)

def add_anomalies(df):
    """Добавление аномалий для реалистичности"""
    
    # 1. Периоды простоя (ночью)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    df.loc[df['is_night'] == 1, 'svc_total_rps'] *= 0.1  # 90% снижение ночью
    df.loc[df['is_night'] == 1, 'svc_rps_topk_sum'] *= 0.1
    df.loc[df['is_night'] == 1, 'svc_rps_long_tail'] *= 0.1
    df.loc[df['is_night'] == 1, 'svc_active_routes'] = 1
    
    # 2. Внезапные всплески (1% данных)
    spike_mask = np.random.random(len(df)) < 0.01
    df.loc[spike_mask, 'svc_total_rps'] *= 10
    df.loc[spike_mask, 'svc_rps_topk_sum'] *= 10
    df.loc[spike_mask, 'svc_rps_long_tail'] *= 10
    df.loc[spike_mask, 'svc_http_p95_ms'] *= 3
    
    # 3. Периоды высоких ошибок (0.5% данных)
    error_mask = np.random.random(len(df)) < 0.005
    df.loc[error_mask, 'dial_failed_rps'] = 5.0 + np.random.random() * 10
    df.loc[error_mask, 'svc_http_p95_ms'] *= 2
    
    return df

def generate_training_csv():
    """Генерация CSV файла с тренировочными данными"""
    print("Генерация синтетических данных за 1 месяц...")
    
    # Генерация данных
    df = generate_synthetic_metrics()
    
    # Добавление аномалий
    df = add_anomalies(df)
    
    # Округление значений
    float_columns = ['svc_total_rps', 'svc_rps_topk_sum', 'svc_rps_long_tail', 
                    'svc_http_p95_ms', 'dial_failed_rps']
    for col in float_columns:
        df[col] = df[col].round(3)
    
    # Сохранение в CSV
    filename = f"synthetic_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    print(f"Данные сохранены в {filename}")
    print(f"Всего записей: {len(df)}")
    print(f"Период: {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"Сервисы: {df['service'].unique().tolist()}")
    
    # Статистика по пиковым часам
    peak_data = df[df['is_peak_hours'] == 1]
    print(f"\nПиковая нагрузка (14-16, 19-23):")
    print(f" - Записей: {len(peak_data)}")
    print(f" - Средний RPS orders: {peak_data[peak_data['service'] == 'orders-service']['svc_total_rps'].mean():.1f}")
    print(f" - Средний RPS payments: {peak_data[peak_data['service'] == 'payments-service']['svc_total_rps'].mean():.1f}")
    
    return df

if __name__ == "__main__":
    df = generate_training_csv()
    
    # Дополнительная визуализация статистики
    print("\nОбщая статистика по сервисам:")
    for service in df['service'].unique():
        service_data = df[df['service'] == service]
        print(f"\n{service}:")
        print(f"  - Средний RPS: {service_data['svc_total_rps'].mean():.1f}")
        print(f"  - Макс RPS: {service_data['svc_total_rps'].max():.1f}")
        print(f"  - Средняя латентность: {service_data['svc_http_p95_ms'].mean():.1f} мс")
        print(f"  - Средние реплики: {service_data['current_replicas'].mean():.1f}")