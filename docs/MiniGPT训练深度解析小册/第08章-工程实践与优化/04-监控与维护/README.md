# 04 监控与维护

> **从被动响应到主动预防：构建智能化的系统监控与维护体系**

## 核心思想

监控与维护是保障语言模型生产系统长期稳定运行的关键环节。与传统的被动监控不同，现代智能监控系统需要具备预测性分析、自动化响应和持续优化的能力。通过建立全方位的监控体系，我们可以在问题发生前进行预警，在故障发生时快速恢复，在系统运行中持续优化。

**关键洞察**：
- **全栈监控**：从基础设施到业务指标的端到端可观测性
- **智能告警**：基于机器学习的异常检测和动态阈值调整
- **自愈能力**：自动化的故障检测、诊断和恢复机制
- **持续优化**：基于监控数据的系统持续改进和模型演进

从数学角度看，监控系统是一个实时状态估计和控制问题，需要在不确定性环境中维持系统的最优运行状态。

## 4.1 系统监控的数学理论

### 时间序列异常检测的统计模型

**ARIMA模型的状态空间表示**：
对于时间序列 $\{y_t\}$，ARIMA(p,d,q)模型可表示为：
$$(1-\phi_1B-...-\phi_pB^p)(1-B)^d y_t = (1+\theta_1B+...+\theta_qB^q)\epsilon_t$$

其中 $B$ 是滞后算子，$\epsilon_t \sim N(0, \sigma^2)$。

**卡尔曼滤波的状态估计**：
状态方程：$\mathbf{x}_t = \mathbf{A}\mathbf{x}_{t-1} + \mathbf{w}_t$
观测方程：$\mathbf{y}_t = \mathbf{H}\mathbf{x}_t + \mathbf{v}_t$

预测步骤：
$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{A}\hat{\mathbf{x}}_{t-1|t-1}$$
$$\mathbf{P}_{t|t-1} = \mathbf{A}\mathbf{P}_{t-1|t-1}\mathbf{A}^T + \mathbf{Q}$$

更新步骤：
$$\mathbf{K}_t = \mathbf{P}_{t|t-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}^T + \mathbf{R})^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t(\mathbf{y}_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1})$$

**异常检测的统计量**：
基于马哈拉诺比斯距离的异常得分：
$$\text{Anomaly Score}_t = (\mathbf{y}_t - \hat{\mathbf{y}}_t)^T \mathbf{\Sigma}^{-1} (\mathbf{y}_t - \hat{\mathbf{y}}_t)$$

其中 $\hat{\mathbf{y}}_t$ 是预测值，$\mathbf{\Sigma}$ 是协方差矩阵。

### SLA可用性的概率模型

**系统可用性计算**：
对于串联系统，总可用性为：
$$A_{total} = \prod_{i=1}^n A_i$$

对于并联系统，总可用性为：
$$A_{total} = 1 - \prod_{i=1}^n (1 - A_i)$$

**MTTR和MTBF的关系**：
系统可用性可表示为：
$$A = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}$$

其中MTBF是平均无故障时间，MTTR是平均恢复时间。

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time
import threading
import asyncio
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# 科学计算和机器学习
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support

# 监控和指标库
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
import psutil
import GPUtil

# 异常检测库
try:
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
except ImportError:
    print("PyOD库未安装，部分异常检测功能将不可用")

class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """告警状态"""
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    unit: str = ""
    aggregation_interval: int = 60  # 聚合间隔(秒)

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # 例如: "value > 0.8"
    severity: AlertSeverity
    duration: int = 300  # 持续时间(秒)
    description: str = ""
    runbook_url: str = ""

@dataclass
class Alert:
    """告警实例"""
    rule_name: str
    metric_name: str
    value: float
    threshold: float
    severity: AlertSeverity
    status: AlertStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

class MetricCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.data_points: Dict[str, deque] = {}
        self.lock = threading.Lock()
        
        # 初始化基础指标
        self._initialize_basic_metrics()
    
    def _initialize_basic_metrics(self):
        """初始化基础系统指标"""
        
        basic_metrics = [
            MetricDefinition("cpu_usage_percent", MetricType.GAUGE, "CPU使用率", unit="%"),
            MetricDefinition("memory_usage_percent", MetricType.GAUGE, "内存使用率", unit="%"),
            MetricDefinition("disk_usage_percent", MetricType.GAUGE, "磁盘使用率", unit="%"),
            MetricDefinition("network_bytes_sent", MetricType.COUNTER, "网络发送字节数", unit="bytes"),
            MetricDefinition("network_bytes_recv", MetricType.COUNTER, "网络接收字节数", unit="bytes"),
            MetricDefinition("gpu_usage_percent", MetricType.GAUGE, "GPU使用率", unit="%"),
            MetricDefinition("gpu_memory_usage_percent", MetricType.GAUGE, "GPU内存使用率", unit="%"),
        ]
        
        for metric_def in basic_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition):
        """注册指标"""
        with self.lock:
            self.metric_definitions[metric_def.name] = metric_def
            self.data_points[metric_def.name] = deque(maxlen=10000)  # 保留最近10000个数据点
            
            # 创建Prometheus指标对象
            if metric_def.type == MetricType.COUNTER:
                self.metrics[metric_def.name] = Counter(
                    metric_def.name, metric_def.description, metric_def.labels
                )
            elif metric_def.type == MetricType.GAUGE:
                self.metrics[metric_def.name] = Gauge(
                    metric_def.name, metric_def.description, metric_def.labels
                )
            elif metric_def.type == MetricType.HISTOGRAM:
                self.metrics[metric_def.name] = Histogram(
                    metric_def.name, metric_def.description, metric_def.labels
                )
            elif metric_def.type == MetricType.SUMMARY:
                self.metrics[metric_def.name] = Summary(
                    metric_def.name, metric_def.description, metric_def.labels
                )
    
    def record_metric(self, 
                     metric_name: str, 
                     value: float, 
                     labels: Optional[Dict[str, str]] = None,
                     timestamp: Optional[datetime] = None):
        """记录指标值"""
        
        if metric_name not in self.metrics:
            logging.warning(f"未注册的指标: {metric_name}")
            return
        
        timestamp = timestamp or datetime.now()
        
        with self.lock:
            # 记录历史数据点
            data_point = {
                'timestamp': timestamp,
                'value': value,
                'labels': labels or {}
            }
            self.data_points[metric_name].append(data_point)
            
            # 更新Prometheus指标
            metric = self.metrics[metric_name]
            metric_def = self.metric_definitions[metric_name]
            
            if labels and metric_def.labels:
                if metric_def.type == MetricType.COUNTER:
                    metric.labels(**labels).inc(value)
                elif metric_def.type == MetricType.GAUGE:
                    metric.labels(**labels).set(value)
                elif metric_def.type == MetricType.HISTOGRAM:
                    metric.labels(**labels).observe(value)
                elif metric_def.type == MetricType.SUMMARY:
                    metric.labels(**labels).observe(value)
            else:
                if metric_def.type == MetricType.COUNTER:
                    metric.inc(value)
                elif metric_def.type == MetricType.GAUGE:
                    metric.set(value)
                elif metric_def.type == MetricType.HISTOGRAM:
                    metric.observe(value)
                elif metric_def.type == MetricType.SUMMARY:
                    metric.observe(value)
    
    def collect_system_metrics(self):
        """收集系统指标"""
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric("cpu_usage_percent", cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.record_metric("memory_usage_percent", memory.percent)
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.record_metric("disk_usage_percent", disk_percent)
        
        # 网络统计
        network = psutil.net_io_counters()
        self.record_metric("network_bytes_sent", network.bytes_sent)
        self.record_metric("network_bytes_recv", network.bytes_recv)
        
        # GPU指标（如果可用）
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                labels = {"gpu_id": str(i), "gpu_name": gpu.name}
                self.record_metric("gpu_usage_percent", gpu.load * 100, labels)
                self.record_metric("gpu_memory_usage_percent", gpu.memoryUtil * 100, labels)
        except Exception as e:
            logging.debug(f"GPU指标收集失败: {e}")
    
    def get_metric_history(self, 
                          metric_name: str, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict]:
        """获取指标历史数据"""
        
        if metric_name not in self.data_points:
            return []
        
        with self.lock:
            data_points = list(self.data_points[metric_name])
        
        # 时间过滤
        if start_time or end_time:
            filtered_data = []
            for point in data_points:
                timestamp = point['timestamp']
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_data.append(point)
            return filtered_data
        
        return data_points
    
    def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式的指标"""
        return generate_latest()

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.training_data: Dict[str, deque] = {}
        self.lock = threading.Lock()
        
        # 异常检测统计
        self.anomaly_stats = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    
    def train_detector(self, metric_name: str, data: List[float], method: str = "isolation_forest"):
        """训练异常检测模型"""
        
        if len(data) < self.window_size:
            logging.warning(f"数据点不足，无法训练检测器: {metric_name}")
            return
        
        # 数据预处理
        X = np.array(data).reshape(-1, 1)
        
        with self.lock:
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[metric_name] = scaler
            
            # 选择检测模型
            if method == "isolation_forest":
                model = IsolationForest(contamination=self.contamination, random_state=42)
            elif method == "local_outlier_factor":
                from sklearn.neighbors import LocalOutlierFactor
                model = LocalOutlierFactor(contamination=self.contamination, novelty=True)
            elif method == "one_class_svm":
                from sklearn.svm import OneClassSVM
                model = OneClassSVM(gamma='auto', nu=self.contamination)
            else:
                model = IsolationForest(contamination=self.contamination, random_state=42)
            
            # 训练模型
            if hasattr(model, 'fit_predict'):
                model.fit(X_scaled)
            else:
                model.fit(X_scaled)
            
            self.models[metric_name] = model
            
            # 存储训练数据用于持续学习
            self.training_data[metric_name] = deque(data, maxlen=self.window_size * 10)
        
        logging.info(f"异常检测模型训练完成: {metric_name}")
    
    def detect_anomaly(self, metric_name: str, value: float) -> Dict:
        """检测异常"""
        
        if metric_name not in self.models:
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'message': '未训练的检测器'
            }
        
        with self.lock:
            model = self.models[metric_name]
            scaler = self.scalers[metric_name]
        
        # 数据预处理
        X = np.array([[value]])
        X_scaled = scaler.transform(X)
        
        # 异常检测
        try:
            if hasattr(model, 'predict'):
                prediction = model.predict(X_scaled)[0]
                is_anomaly = prediction == -1
                
                # 计算异常得分
                if hasattr(model, 'decision_function'):
                    score = model.decision_function(X_scaled)[0]
                    confidence = abs(score)
                elif hasattr(model, 'score_samples'):
                    score = model.score_samples(X_scaled)[0]
                    confidence = abs(score)
                else:
                    confidence = 1.0 if is_anomaly else 0.0
            else:
                is_anomaly = False
                confidence = 0.0
        except Exception as e:
            logging.error(f"异常检测失败: {e}")
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'message': f'检测失败: {e}'
            }
        
        # 更新统计
        if is_anomaly:
            self.anomaly_stats['total_detections'] += 1
        
        # 持续学习：将新数据点加入训练集
        if metric_name in self.training_data:
            self.training_data[metric_name].append(value)
            
            # 定期重新训练
            if len(self.training_data[metric_name]) % 100 == 0:
                self._retrain_model(metric_name)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'value': value,
            'timestamp': datetime.now(),
            'model_type': type(model).__name__
        }
    
    def _retrain_model(self, metric_name: str):
        """重新训练模型"""
        try:
            data = list(self.training_data[metric_name])
            self.train_detector(metric_name, data)
            logging.info(f"模型重新训练完成: {metric_name}")
        except Exception as e:
            logging.error(f"模型重新训练失败: {e}")
    
    def batch_detect_anomalies(self, 
                              metric_name: str, 
                              values: List[float]) -> List[Dict]:
        """批量异常检测"""
        
        results = []
        for value in values:
            result = self.detect_anomaly(metric_name, value)
            results.append(result)
        
        return results
    
    def get_anomaly_statistics(self) -> Dict:
        """获取异常检测统计"""
        
        total = self.anomaly_stats['total_detections']
        if total == 0:
            return {
                'total_detections': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        tp = self.anomaly_stats['true_positives']
        fp = self.anomaly_stats['false_positives']
        fn = self.anomaly_stats['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'total_detections': total,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: List[Callable] = []
        self.lock = threading.Lock()
        
        # 告警统计
        self.alert_stats = {
            'total_alerts': 0,
            'resolved_alerts': 0,
            'average_resolution_time': 0.0
        }
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self.lock:
            self.alert_rules[rule.name] = rule
        logging.info(f"告警规则已添加: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        with self.lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logging.info(f"告警规则已移除: {rule_name}")
    
    def evaluate_rules(self, metric_name: str, value: float):
        """评估告警规则"""
        
        current_time = datetime.now()
        
        with self.lock:
            rules_to_evaluate = [
                rule for rule in self.alert_rules.values() 
                if rule.metric_name == metric_name
            ]
        
        for rule in rules_to_evaluate:
            try:
                # 评估条件
                condition_met = self._evaluate_condition(rule.condition, value)
                alert_key = f"{rule.name}:{metric_name}"
                
                if condition_met:
                    if alert_key not in self.active_alerts:
                        # 创建新告警
                        alert = Alert(
                            rule_name=rule.name,
                            metric_name=metric_name,
                            value=value,
                            threshold=self._extract_threshold(rule.condition),
                            severity=rule.severity,
                            status=AlertStatus.FIRING,
                            start_time=current_time,
                            annotations={
                                'description': rule.description,
                                'runbook_url': rule.runbook_url
                            }
                        )
                        
                        with self.lock:
                            self.active_alerts[alert_key] = alert
                            self.alert_history.append(alert)
                            self.alert_stats['total_alerts'] += 1
                        
                        # 发送通知
                        self._send_notification(alert)
                        
                        logging.warning(f"新告警: {rule.name} - {metric_name}={value}")
                    else:
                        # 更新现有告警
                        with self.lock:
                            self.active_alerts[alert_key].value = value
                
                else:
                    if alert_key in self.active_alerts:
                        # 解决告警
                        with self.lock:
                            alert = self.active_alerts[alert_key]
                            alert.status = AlertStatus.RESOLVED
                            alert.end_time = current_time
                            
                            # 计算解决时间
                            resolution_time = (current_time - alert.start_time).total_seconds()
                            self._update_resolution_stats(resolution_time)
                            
                            del self.active_alerts[alert_key]
                        
                        # 发送解决通知
                        self._send_resolution_notification(alert)
                        
                        logging.info(f"告警已解决: {rule.name} - {metric_name}={value}")
                        
            except Exception as e:
                logging.error(f"告警规则评估失败: {rule.name}, 错误: {e}")
    
    def _evaluate_condition(self, condition: str, value: float) -> bool:
        """评估告警条件"""
        
        # 简单的条件解析和评估
        # 支持的操作符: >, <, >=, <=, ==, !=
        try:
            # 替换condition中的'value'为实际值
            safe_condition = condition.replace('value', str(value))
            
            # 安全评估（仅允许数字和基本操作符）
            allowed_chars = set('0123456789.><!=+-*/()')
            if not all(c in allowed_chars or c.isspace() for c in safe_condition):
                return False
            
            return eval(safe_condition)
        except:
            return False
    
    def _extract_threshold(self, condition: str) -> float:
        """从条件中提取阈值"""
        import re
        # 简单提取数字作为阈值
        numbers = re.findall(r'[-+]?\d*\.?\d+', condition)
        return float(numbers[0]) if numbers else 0.0
    
    def _send_notification(self, alert: Alert):
        """发送告警通知"""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logging.error(f"通知发送失败: {e}")
    
    def _send_resolution_notification(self, alert: Alert):
        """发送解决通知"""
        for channel in self.notification_channels:
            try:
                if hasattr(channel, 'send_resolution'):
                    channel.send_resolution(alert)
                else:
                    # 标记为解决状态后发送
                    channel(alert)
            except Exception as e:
                logging.error(f"解决通知发送失败: {e}")
    
    def _update_resolution_stats(self, resolution_time: float):
        """更新解决时间统计"""
        with self.lock:
            self.alert_stats['resolved_alerts'] += 1
            total_resolved = self.alert_stats['resolved_alerts']
            current_avg = self.alert_stats['average_resolution_time']
            
            # 计算新的平均解决时间
            new_avg = ((current_avg * (total_resolved - 1)) + resolution_time) / total_resolved
            self.alert_stats['average_resolution_time'] = new_avg
    
    def add_notification_channel(self, channel: Callable):
        """添加通知渠道"""
        self.notification_channels.append(channel)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """获取告警历史"""
        
        with self.lock:
            alerts = list(self.alert_history)
        
        # 过滤条件
        filtered_alerts = []
        for alert in alerts:
            if start_time and alert.start_time < start_time:
                continue
            if end_time and alert.start_time > end_time:
                continue
            if severity and alert.severity != severity:
                continue
            filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def get_alert_statistics(self) -> Dict:
        """获取告警统计"""
        with self.lock:
            return dict(self.alert_stats)

class ModelDriftDetector:
    """模型漂移检测器"""
    
    def __init__(self, reference_window: int = 1000, detection_window: int = 100):
        self.reference_window = reference_window
        self.detection_window = detection_window
        self.reference_data: Dict[str, deque] = {}
        self.current_data: Dict[str, deque] = {}
        self.drift_history: List[Dict] = []
        self.lock = threading.Lock()
    
    def set_reference_data(self, feature_name: str, data: List[float]):
        """设置参考数据"""
        with self.lock:
            self.reference_data[feature_name] = deque(data, maxlen=self.reference_window)
    
    def add_current_data(self, feature_name: str, value: float):
        """添加当前数据"""
        with self.lock:
            if feature_name not in self.current_data:
                self.current_data[feature_name] = deque(maxlen=self.detection_window)
            self.current_data[feature_name].append(value)
    
    def detect_drift(self, feature_name: str) -> Dict:
        """检测数据漂移"""
        
        if feature_name not in self.reference_data:
            return {'error': '缺少参考数据'}
        
        if feature_name not in self.current_data:
            return {'error': '缺少当前数据'}
        
        with self.lock:
            reference = list(self.reference_data[feature_name])
            current = list(self.current_data[feature_name])
        
        if len(current) < self.detection_window:
            return {'error': '当前数据不足'}
        
        # Kolmogorov-Smirnov检验
        ks_statistic, ks_p_value = stats.ks_2samp(reference, current)
        
        # Jensen-Shannon散度
        js_divergence = self._jensen_shannon_divergence(reference, current)
        
        # Population Stability Index (PSI)
        psi = self._population_stability_index(reference, current)
        
        # 综合漂移得分
        drift_score = (ks_statistic + js_divergence + psi) / 3.0
        
        # 判断是否发生漂移
        is_drift = (ks_p_value < 0.05) or (psi > 0.2) or (js_divergence > 0.1)
        
        drift_result = {
            'feature_name': feature_name,
            'is_drift': is_drift,
            'drift_score': drift_score,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            'js_divergence': js_divergence,
            'psi': psi,
            'timestamp': datetime.now(),
            'reference_stats': {
                'mean': np.mean(reference),
                'std': np.std(reference),
                'count': len(reference)
            },
            'current_stats': {
                'mean': np.mean(current),
                'std': np.std(current),
                'count': len(current)
            }
        }
        
        # 记录漂移历史
        self.drift_history.append(drift_result)
        
        return drift_result
    
    def _jensen_shannon_divergence(self, X: List[float], Y: List[float], bins: int = 10) -> float:
        """计算Jensen-Shannon散度"""
        
        # 创建直方图
        x_min, x_max = min(min(X), min(Y)), max(max(X), max(Y))
        x_bins = np.linspace(x_min, x_max, bins + 1)
        
        p, _ = np.histogram(X, bins=x_bins, density=True)
        q, _ = np.histogram(Y, bins=x_bins, density=True)
        
        # 归一化
        p = p / np.sum(p) + 1e-10  # 避免零值
        q = q / np.sum(q) + 1e-10
        
        # 计算JS散度
        m = (p + q) / 2
        js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
        
        return js_div
    
    def _population_stability_index(self, reference: List[float], current: List[float], bins: int = 10) -> float:
        """计算Population Stability Index"""
        
        # 基于参考数据创建分箱
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # 计算各分箱的比例
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        ref_percents = ref_counts / len(reference) + 1e-10
        cur_percents = cur_counts / len(current) + 1e-10
        
        # 计算PSI
        psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))
        
        return psi
    
    def get_drift_history(self, feature_name: Optional[str] = None) -> List[Dict]:
        """获取漂移历史"""
        
        if feature_name is None:
            return self.drift_history
        
        return [drift for drift in self.drift_history if drift['feature_name'] == feature_name]

class AutomatedResponseSystem:
    """自动化响应系统"""
    
    def __init__(self):
        self.response_rules: Dict[str, Callable] = {}
        self.response_history: List[Dict] = []
        self.lock = threading.Lock()
        
        # 初始化基础响应策略
        self._initialize_basic_responses()
    
    def _initialize_basic_responses(self):
        """初始化基础响应策略"""
        
        # CPU使用率过高的响应
        self.register_response("high_cpu_usage", self._handle_high_cpu)
        
        # 内存使用率过高的响应
        self.register_response("high_memory_usage", self._handle_high_memory)
        
        # 服务不可用的响应
        self.register_response("service_unavailable", self._handle_service_unavailable)
        
        # 模型漂移的响应
        self.register_response("model_drift", self._handle_model_drift)
    
    def register_response(self, trigger_name: str, response_func: Callable):
        """注册响应策略"""
        with self.lock:
            self.response_rules[trigger_name] = response_func
        logging.info(f"响应策略已注册: {trigger_name}")
    
    def trigger_response(self, trigger_name: str, context: Dict) -> Dict:
        """触发自动化响应"""
        
        if trigger_name not in self.response_rules:
            return {'error': f'未知触发器: {trigger_name}'}
        
        try:
            response_func = self.response_rules[trigger_name]
            result = response_func(context)
            
            # 记录响应历史
            response_record = {
                'trigger_name': trigger_name,
                'context': context,
                'result': result,
                'timestamp': datetime.now()
            }
            
            with self.lock:
                self.response_history.append(response_record)
            
            logging.info(f"自动化响应执行: {trigger_name}")
            return result
            
        except Exception as e:
            error_result = {'error': f'响应执行失败: {e}'}
            logging.error(f"自动化响应失败: {trigger_name}, 错误: {e}")
            return error_result
    
    def _handle_high_cpu(self, context: Dict) -> Dict:
        """处理CPU使用率过高"""
        
        cpu_usage = context.get('cpu_usage', 0)
        
        actions_taken = []
        
        # 1. 记录详细的系统状态
        process_info = self._get_top_processes()
        actions_taken.append(f"记录了{len(process_info)}个高CPU进程")
        
        # 2. 如果CPU使用率极高，考虑重启高消耗进程
        if cpu_usage > 95:
            # 这里应该实现实际的进程管理逻辑
            actions_taken.append("CPU使用率>95%，已标记为需要人工干预")
        
        # 3. 发送详细告警
        actions_taken.append("已发送详细CPU告警")
        
        return {
            'success': True,
            'actions_taken': actions_taken,
            'process_info': process_info,
            'recommendation': '建议检查高CPU进程并考虑优化或重启'
        }
    
    def _handle_high_memory(self, context: Dict) -> Dict:
        """处理内存使用率过高"""
        
        memory_usage = context.get('memory_usage', 0)
        
        actions_taken = []
        
        # 1. 强制垃圾回收
        import gc
        gc.collect()
        actions_taken.append("执行了垃圾回收")
        
        # 2. 清理临时文件
        temp_cleaned = self._cleanup_temp_files()
        actions_taken.append(f"清理了{temp_cleaned}MB临时文件")
        
        # 3. 如果内存极高，建议重启服务
        if memory_usage > 90:
            actions_taken.append("内存使用率>90%，建议重启服务")
        
        return {
            'success': True,
            'actions_taken': actions_taken,
            'temp_cleaned_mb': temp_cleaned,
            'recommendation': '内存使用率过高，建议检查内存泄漏'
        }
    
    def _handle_service_unavailable(self, context: Dict) -> Dict:
        """处理服务不可用"""
        
        service_name = context.get('service_name', 'unknown')
        
        actions_taken = []
        
        # 1. 尝试健康检查
        health_status = self._perform_health_check(service_name)
        actions_taken.append(f"执行健康检查: {health_status}")
        
        # 2. 尝试重启服务
        restart_result = self._restart_service(service_name)
        actions_taken.append(f"尝试重启服务: {restart_result}")
        
        # 3. 检查依赖服务
        dependency_status = self._check_dependencies(service_name)
        actions_taken.append(f"检查依赖服务: {dependency_status}")
        
        return {
            'success': True,
            'actions_taken': actions_taken,
            'health_status': health_status,
            'restart_result': restart_result,
            'dependency_status': dependency_status
        }
    
    def _handle_model_drift(self, context: Dict) -> Dict:
        """处理模型漂移"""
        
        feature_name = context.get('feature_name', 'unknown')
        drift_score = context.get('drift_score', 0)
        
        actions_taken = []
        
        # 1. 记录漂移详情
        actions_taken.append(f"记录了特征{feature_name}的漂移，得分: {drift_score}")
        
        # 2. 如果漂移严重，触发模型重训练流程
        if drift_score > 0.5:
            retrain_triggered = self._trigger_model_retrain(feature_name)
            actions_taken.append(f"触发模型重训练: {retrain_triggered}")
        
        # 3. 更新监控阈值
        threshold_updated = self._update_drift_thresholds(feature_name, drift_score)
        actions_taken.append(f"更新漂移阈值: {threshold_updated}")
        
        return {
            'success': True,
            'actions_taken': actions_taken,
            'retrain_needed': drift_score > 0.5,
            'recommendation': '检查数据分布变化并考虑模型更新'
        }
    
    def _get_top_processes(self, top_n: int = 10) -> List[Dict]:
        """获取CPU使用率最高的进程"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                processes.append(proc.info)
            
            # 按CPU使用率降序排序
            top_processes = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:top_n]
            return top_processes
        except Exception as e:
            logging.error(f"获取进程信息失败: {e}")
            return []
    
    def _cleanup_temp_files(self) -> float:
        """清理临时文件"""
        # 模拟清理，实际应该实现真正的清理逻辑
        import random
        cleaned_mb = random.uniform(10, 100)
        return round(cleaned_mb, 2)
    
    def _perform_health_check(self, service_name: str) -> str:
        """执行健康检查"""
        # 模拟健康检查
        return "健康检查完成"
    
    def _restart_service(self, service_name: str) -> str:
        """重启服务"""
        # 模拟服务重启
        return f"服务{service_name}重启尝试完成"
    
    def _check_dependencies(self, service_name: str) -> str:
        """检查依赖服务"""
        # 模拟依赖检查
        return "依赖服务状态正常"
    
    def _trigger_model_retrain(self, feature_name: str) -> str:
        """触发模型重训练"""
        # 模拟重训练触发
        return f"特征{feature_name}的重训练任务已创建"
    
    def _update_drift_thresholds(self, feature_name: str, drift_score: float) -> str:
        """更新漂移阈值"""
        # 模拟阈值更新
        return f"特征{feature_name}的漂移阈值已更新"
    
    def get_response_history(self, 
                           trigger_name: Optional[str] = None,
                           start_time: Optional[datetime] = None) -> List[Dict]:
        """获取响应历史"""
        
        with self.lock:
            history = list(self.response_history)
        
        # 过滤条件
        if trigger_name:
            history = [h for h in history if h['trigger_name'] == trigger_name]
        
        if start_time:
            history = [h for h in history if h['timestamp'] >= start_time]
        
        return history

class MonitoringSystem:
    """综合监控系统"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.metric_collector = MetricCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.drift_detector = ModelDriftDetector()
        self.response_system = AutomatedResponseSystem()
        
        # 监控线程
        self.monitoring_thread = None
        self.is_running = False
        
        # 设置基础告警规则
        self._setup_basic_alert_rules()
        
        # 设置通知渠道
        self._setup_notification_channels()
        
        logging.info("监控系统初始化完成")
    
    def _setup_basic_alert_rules(self):
        """设置基础告警规则"""
        
        basic_rules = [
            AlertRule(
                name="高CPU使用率",
                metric_name="cpu_usage_percent",
                condition="value > 80",
                severity=AlertSeverity.WARNING,
                duration=300,
                description="CPU使用率超过80%"
            ),
            AlertRule(
                name="极高CPU使用率",
                metric_name="cpu_usage_percent",
                condition="value > 95",
                severity=AlertSeverity.CRITICAL,
                duration=60,
                description="CPU使用率超过95%"
            ),
            AlertRule(
                name="高内存使用率",
                metric_name="memory_usage_percent",
                condition="value > 85",
                severity=AlertSeverity.WARNING,
                duration=300,
                description="内存使用率超过85%"
            ),
            AlertRule(
                name="磁盘空间不足",
                metric_name="disk_usage_percent",
                condition="value > 90",
                severity=AlertSeverity.ERROR,
                duration=60,
                description="磁盘使用率超过90%"
            )
        ]
        
        for rule in basic_rules:
            self.alert_manager.add_alert_rule(rule)
    
    def _setup_notification_channels(self):
        """设置通知渠道"""
        
        # 添加日志通知渠道
        def log_notification(alert: Alert):
            logging.warning(f"告警通知: {alert.rule_name} - {alert.metric_name}={alert.value}")
        
        self.alert_manager.add_notification_channel(log_notification)
    
    def start_monitoring(self, interval: int = 30):
        """启动监控"""
        
        if self.is_running:
            logging.warning("监控系统已在运行")
            return
        
        self.is_running = True
        
        def monitoring_loop():
            while self.is_running:
                try:
                    # 收集系统指标
                    self.metric_collector.collect_system_metrics()
                    
                    # 获取最新指标并进行异常检测和告警评估
                    for metric_name in ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent']:
                        history = self.metric_collector.get_metric_history(metric_name)
                        if history:
                            latest_value = history[-1]['value']
                            
                            # 异常检测
                            if len(history) > 100:  # 有足够数据时才进行异常检测
                                values = [point['value'] for point in history[-100:]]
                                
                                # 训练异常检测器（如果还没有）
                                if metric_name not in self.anomaly_detector.models:
                                    self.anomaly_detector.train_detector(metric_name, values)
                                
                                # 检测异常
                                anomaly_result = self.anomaly_detector.detect_anomaly(metric_name, latest_value)
                                if anomaly_result['is_anomaly']:
                                    logging.warning(f"检测到异常: {metric_name}={latest_value}, 置信度={anomaly_result['confidence']}")
                            
                            # 告警评估
                            self.alert_manager.evaluate_rules(metric_name, latest_value)
                    
                    # 检查是否有告警需要自动化响应
                    active_alerts = self.alert_manager.get_active_alerts()
                    for alert in active_alerts:
                        self._handle_alert_response(alert)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logging.error(f"监控循环错误: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logging.info(f"监控系统已启动，监控间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logging.info("监控系统已停止")
    
    def _handle_alert_response(self, alert: Alert):
        """处理告警响应"""
        
        # 根据告警类型触发自动化响应
        if alert.metric_name == 'cpu_usage_percent' and alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            context = {
                'cpu_usage': alert.value,
                'alert': alert
            }
            self.response_system.trigger_response('high_cpu_usage', context)
        
        elif alert.metric_name == 'memory_usage_percent' and alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            context = {
                'memory_usage': alert.value,
                'alert': alert
            }
            self.response_system.trigger_response('high_memory_usage', context)
    
    def add_custom_metric(self, metric_def: MetricDefinition):
        """添加自定义指标"""
        self.metric_collector.register_metric(metric_def)
    
    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict] = None):
        """记录指标值"""
        self.metric_collector.record_metric(metric_name, value, labels)
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_manager.add_alert_rule(rule)
    
    def get_system_status(self) -> Dict:
        """获取系统状态概览"""
        
        # 获取最新系统指标
        metrics_status = {}
        for metric_name in ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent']:
            history = self.metric_collector.get_metric_history(metric_name)
            if history:
                latest = history[-1]
                metrics_status[metric_name] = {
                    'value': latest['value'],
                    'timestamp': latest['timestamp']
                }
        
        # 获取活跃告警
        active_alerts = self.alert_manager.get_active_alerts()
        
        # 获取异常检测统计
        anomaly_stats = self.anomaly_detector.get_anomaly_statistics()
        
        # 获取告警统计
        alert_stats = self.alert_manager.get_alert_statistics()
        
        return {
            'monitoring_status': 'running' if self.is_running else 'stopped',
            'metrics': metrics_status,
            'active_alerts_count': len(active_alerts),
            'active_alerts': [
                {
                    'rule_name': alert.rule_name,
                    'severity': alert.severity.value,
                    'value': alert.value,
                    'start_time': alert.start_time
                } for alert in active_alerts
            ],
            'anomaly_detection': anomaly_stats,
            'alert_statistics': alert_stats,
            'timestamp': datetime.now()
        }
    
    def get_dashboard_data(self) -> Dict:
        """获取仪表板数据"""
        
        dashboard_data = {}
        
        # 获取各种指标的时间序列数据
        for metric_name in ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent']:
            history = self.metric_collector.get_metric_history(
                metric_name, 
                start_time=datetime.now() - timedelta(hours=24)
            )
            
            dashboard_data[metric_name] = {
                'timestamps': [point['timestamp'] for point in history],
                'values': [point['value'] for point in history],
                'latest_value': history[-1]['value'] if history else 0
            }
        
        # 获取告警趋势
        alert_history = self.alert_manager.get_alert_history(
            start_time=datetime.now() - timedelta(hours=24)
        )
        
        # 按小时统计告警数量
        hourly_alerts = defaultdict(int)
        for alert in alert_history:
            hour_key = alert.start_time.strftime('%Y-%m-%d %H:00')
            hourly_alerts[hour_key] += 1
        
        dashboard_data['alert_trend'] = {
            'hours': list(hourly_alerts.keys()),
            'counts': list(hourly_alerts.values())
        }
        
        return dashboard_data

## 应用指导

### 监控最佳实践

1. **分层监控策略**：
   - 基础设施层：CPU、内存、磁盘、网络
   - 应用层：请求延迟、错误率、吞吐量
   - 业务层：用户体验、业务指标

2. **告警规则设计**：
   - 基于历史数据设定动态阈值
   - 实施告警分级和升级机制
   - 避免告警风暴和疲劳

3. **异常检测优化**：
   - 选择合适的检测算法
   - 定期重训练检测模型
   - 结合业务知识调整检测策略

4. **自动化响应**：
   - 建立响应策略库
   - 实施渐进式自动化
   - 保持人工干预能力

## 性能与扩展性

### 监控系统性能优化

- **数据采样**：对高频指标使用采样策略
- **存储优化**：使用时间序列数据库
- **计算分布**：将复杂计算分布到多个节点
- **缓存策略**：缓存计算结果和查询结果

### 可扩展性设计

1. **水平扩展**：
   - 监控组件的分布式部署
   - 负载均衡和故障转移
   - 数据分片和复制

2. **插件化架构**：
   - 可插拔的监控源
   - 可扩展的告警渠道
   - 自定义响应策略

## 扩展阅读

- 《Site Reliability Engineering》- 谷歌SRE实践
- 《Monitoring and Observability》- 监控与可观测性
- 《The Art of Monitoring》- 监控的艺术
- 《Chaos Engineering》- 混沌工程实践

---

*"监控不是目的，而是保障服务可靠性的手段。好的监控系统应该是透明的、智能的、可操作的。"* 📊