# 03 部署与生产系统

> **从模型到服务：构建可靠、可扩展的生产级语言模型系统**

## 核心思想

部署与生产系统是将训练好的语言模型转化为实际服务的关键环节。与实验室环境不同，生产系统需要考虑高可用性、可扩展性、安全性、成本控制等多个维度的问题。一个成功的生产系统不仅要保证模型的性能表现，更要确保系统的稳定运行和用户体验。

**关键洞察**：
- **服务化抽象**：将模型能力封装为标准化的服务接口
- **弹性架构**：支持动态扩缩容和故障自愈的系统设计
- **多层缓存**：通过智能缓存策略提升响应速度和降低成本
- **渐进式部署**：通过A/B测试和金丝雀发布确保系统可靠性

从数学角度看，生产系统设计是一个多目标优化问题，需要在性能、可用性、成本、延迟等多个约束条件下寻找帕累托最优解。

## 3.1 服务架构设计的数学建模

### 微服务架构的排队论分析

**服务请求的到达过程**：
假设请求按泊松过程到达，强度为 $\lambda$，则在时间区间 $[0, t]$ 内到达 $k$ 个请求的概率为：
$$P(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}$$

**服务时间分布**：
设模型推理时间服从参数为 $\mu$ 的指数分布，则服务时间的概率密度函数为：
$$f(t) = \mu e^{-\mu t}, \quad t \geq 0$$

**M/M/c 排队模型**：
对于 $c$ 个并行服务实例的系统，稳态概率为：
$$\pi_n = \begin{cases}
\frac{\rho^n}{n!} \pi_0 & \text{if } n \leq c \\
\frac{\rho^n}{c! c^{n-c}} \pi_0 & \text{if } n > c
\end{cases}$$

其中 $\rho = \lambda/\mu$ 是系统负载强度。

**平均响应时间**：
根据利特尔定理和排队论，平均响应时间为：
$$W = \frac{1}{\mu} + \frac{W_q}{\lambda} = \frac{1}{\mu} + \frac{C(c, \rho)}{c\mu - \lambda}$$

其中 $C(c, \rho)$ 是厄朗C公式。

### 负载均衡策略的优化理论

**加权轮询的数学表示**：
设有 $n$ 个服务实例，权重分别为 $w_1, w_2, ..., w_n$，则请求分配概率为：
$$p_i = \frac{w_i}{\sum_{j=1}^n w_j}$$

**最小连接数策略**：
$$\text{选择实例} = \arg\min_i \frac{c_i}{w_i}$$

其中 $c_i$ 是实例 $i$ 当前的连接数。

**一致性哈希的数学原理**：
对于请求键 $k$ 和服务节点 $s_i$，使用哈希函数 $h$：
$$\text{映射节点} = \arg\min_{s_i} \{h(s_i) : h(s_i) \geq h(k)\}$$

```python
import torch
import torch.nn as nn
import asyncio
import aiohttp
from aiohttp import web
import json
import time
import logging
import hashlib
import bisect
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import docker
import kubernetes
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis
import uvloop
import grpc
from grpc import aio as aio_grpc
import websockets
import ssl
from contextlib import asynccontextmanager

class ServiceType(Enum):
    """服务类型枚举"""
    HTTP_REST = "http_rest"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    ASYNC_HTTP = "async_http"

class LoadBalanceStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"

@dataclass
class ServiceConfig:
    """服务配置"""
    # 基础配置
    service_name: str = "minigpt-service"
    service_type: ServiceType = ServiceType.HTTP_REST
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    
    # 模型配置
    model_path: str = ""
    tokenizer_path: str = ""
    max_batch_size: int = 32
    max_sequence_length: int = 512
    
    # 性能配置
    enable_batching: bool = True
    batch_timeout_ms: int = 10
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # 监控配置
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # 安全配置
    enable_ssl: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""

@dataclass
class ServerInstance:
    """服务实例"""
    instance_id: str
    host: str
    port: int
    weight: float = 1.0
    current_connections: int = 0
    health_status: bool = True
    last_health_check: float = 0.0
    performance_score: float = 1.0

class ConsistentHashRing:
    """一致性哈希环"""
    
    def __init__(self, replicas: int = 150):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        
    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    
    def add_node(self, node: str):
        """添加节点"""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            key = self._hash(virtual_key)
            self.ring[key] = node
            bisect.insort(self.sorted_keys, key)
    
    def remove_node(self, node: str):
        """移除节点"""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            key = self._hash(virtual_key)
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)
    
    def get_node(self, key: str) -> Optional[str]:
        """获取节点"""
        if not self.ring:
            return None
            
        hash_key = self._hash(key)
        
        # 找到第一个大于等于hash_key的节点
        idx = bisect.bisect_right(self.sorted_keys, hash_key)
        if idx == len(self.sorted_keys):
            idx = 0
            
        return self.ring[self.sorted_keys[idx]]

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.instances: List[ServerInstance] = []
        self.current_index = 0
        self.hash_ring = ConsistentHashRing()
        self.instance_weights = {}
        self.lock = threading.Lock()
        
        # 性能监控
        self.request_count = Counter('lb_requests_total', 'Total requests', ['instance'])
        self.response_time = Histogram('lb_response_time_seconds', 'Response time', ['instance'])
        self.active_connections = Gauge('lb_active_connections', 'Active connections', ['instance'])
        
    def add_instance(self, instance: ServerInstance):
        """添加服务实例"""
        with self.lock:
            self.instances.append(instance)
            if self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
                self.hash_ring.add_node(f"{instance.host}:{instance.port}")
            self.instance_weights[instance.instance_id] = instance.weight
    
    def remove_instance(self, instance_id: str):
        """移除服务实例"""
        with self.lock:
            instance = next((inst for inst in self.instances if inst.instance_id == instance_id), None)
            if instance:
                self.instances.remove(instance)
                if self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
                    self.hash_ring.remove_node(f"{instance.host}:{instance.port}")
                if instance_id in self.instance_weights:
                    del self.instance_weights[instance_id]
    
    def select_instance(self, request_key: Optional[str] = None) -> Optional[ServerInstance]:
        """选择服务实例"""
        if not self.instances:
            return None
        
        healthy_instances = [inst for inst in self.instances if inst.health_status]
        if not healthy_instances:
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_instances)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_instances)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_instances)
        elif self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(healthy_instances, request_key)
        else:  # RANDOM
            return np.random.choice(healthy_instances)
    
    def _round_robin_select(self, instances: List[ServerInstance]) -> ServerInstance:
        """轮询选择"""
        with self.lock:
            instance = instances[self.current_index % len(instances)]
            self.current_index += 1
            return instance
    
    def _weighted_round_robin_select(self, instances: List[ServerInstance]) -> ServerInstance:
        """加权轮询选择"""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return instances[0]
        
        # 生成累积权重
        cumulative_weights = []
        cumulative = 0
        for inst in instances:
            cumulative += inst.weight
            cumulative_weights.append(cumulative)
        
        # 随机选择
        r = np.random.uniform(0, total_weight)
        for i, cum_weight in enumerate(cumulative_weights):
            if r <= cum_weight:
                return instances[i]
        
        return instances[-1]
    
    def _least_connections_select(self, instances: List[ServerInstance]) -> ServerInstance:
        """最少连接选择"""
        return min(instances, key=lambda x: x.current_connections / x.weight)
    
    def _consistent_hash_select(self, instances: List[ServerInstance], key: str) -> Optional[ServerInstance]:
        """一致性哈希选择"""
        if not key:
            return self._round_robin_select(instances)
        
        node_key = self.hash_ring.get_node(key)
        if not node_key:
            return instances[0]
        
        # 找到对应的实例
        for inst in instances:
            if f"{inst.host}:{inst.port}" == node_key:
                return inst
        
        return instances[0]
    
    def update_instance_metrics(self, instance_id: str, connections: int, response_time: float):
        """更新实例指标"""
        instance = next((inst for inst in self.instances if inst.instance_id == instance_id), None)
        if instance:
            instance.current_connections = connections
            instance.performance_score = self._calculate_performance_score(instance, response_time)
            
            # 更新监控指标
            self.active_connections.labels(instance=instance_id).set(connections)
            self.response_time.labels(instance=instance_id).observe(response_time)
    
    def _calculate_performance_score(self, instance: ServerInstance, response_time: float) -> float:
        """计算性能评分"""
        # 基于响应时间和连接数的综合评分
        time_score = max(0, 1.0 - response_time / 10.0)  # 假设10秒是最差响应时间
        load_score = max(0, 1.0 - instance.current_connections / 100.0)  # 假设100是最大连接数
        return (time_score + load_score) / 2.0

class RequestBatcher:
    """请求批处理器"""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = deque()
        self.batch_queue = asyncio.Queue()
        self.processing_lock = asyncio.Lock()
        
        # 性能监控
        self.batch_size_histogram = Histogram('batch_size', 'Batch size distribution')
        self.batch_wait_time = Histogram('batch_wait_time_seconds', 'Batch wait time')
        
    async def add_request(self, request_data: Dict, response_future: asyncio.Future):
        """添加请求到批处理队列"""
        async with self.processing_lock:
            self.pending_requests.append({
                'data': request_data,
                'future': response_future,
                'timestamp': time.time()
            })
            
            # 检查是否需要创建批次
            if len(self.pending_requests) >= self.max_batch_size:
                await self._create_batch()
    
    async def _create_batch(self):
        """创建批次"""
        if not self.pending_requests:
            return
        
        batch_size = min(len(self.pending_requests), self.max_batch_size)
        batch_requests = []
        
        for _ in range(batch_size):
            batch_requests.append(self.pending_requests.popleft())
        
        # 计算等待时间
        current_time = time.time()
        wait_times = [current_time - req['timestamp'] for req in batch_requests]
        avg_wait_time = sum(wait_times) / len(wait_times)
        
        # 更新监控指标
        self.batch_size_histogram.observe(batch_size)
        self.batch_wait_time.observe(avg_wait_time)
        
        await self.batch_queue.put(batch_requests)
    
    async def get_batch(self) -> List[Dict]:
        """获取批次"""
        return await self.batch_queue.get()
    
    async def start_timeout_monitor(self):
        """启动超时监控"""
        while True:
            await asyncio.sleep(self.timeout_ms / 1000.0)
            async with self.processing_lock:
                if self.pending_requests:
                    oldest_request = self.pending_requests[0]
                    if time.time() - oldest_request['timestamp'] >= self.timeout_ms / 1000.0:
                        await self._create_batch()

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
        self.local_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        
        # 监控指标
        self.cache_hits = Counter('cache_hits_total', 'Cache hits')
        self.cache_misses = Counter('cache_misses_total', 'Cache misses')
        self.cache_size = Gauge('cache_size_bytes', 'Cache size in bytes')
    
    def _generate_cache_key(self, request_data: Dict) -> str:
        """生成缓存键"""
        # 创建标准化的缓存键
        normalized_data = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(normalized_data.encode()).hexdigest()
    
    async def get(self, request_data: Dict) -> Optional[Dict]:
        """获取缓存"""
        cache_key = self._generate_cache_key(request_data)
        
        # 先检查本地缓存
        if cache_key in self.local_cache:
            self.cache_stats['hits'] += 1
            self.cache_hits.inc()
            return self.local_cache[cache_key]
        
        # 检查Redis缓存
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                # 更新本地缓存
                self.local_cache[cache_key] = result
                self.cache_stats['hits'] += 1
                self.cache_hits.inc()
                return result
        except Exception as e:
            logging.warning(f"Redis缓存读取失败: {e}")
        
        self.cache_stats['misses'] += 1
        self.cache_misses.inc()
        return None
    
    async def set(self, request_data: Dict, response_data: Dict):
        """设置缓存"""
        cache_key = self._generate_cache_key(request_data)
        
        # 更新本地缓存
        self.local_cache[cache_key] = response_data
        
        # 更新Redis缓存
        try:
            self.redis_client.setex(
                cache_key, 
                self.ttl, 
                json.dumps(response_data, ensure_ascii=False)
            )
            self.cache_stats['sets'] += 1
        except Exception as e:
            logging.warning(f"Redis缓存写入失败: {e}")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'total_sets': self.cache_stats['sets'],
            'local_cache_size': len(self.local_cache)
        }

class ModelService:
    """模型服务核心类"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.load_balancer = LoadBalancer()
        self.request_batcher = RequestBatcher(
            max_batch_size=config.max_batch_size,
            timeout_ms=config.batch_timeout_ms
        )
        self.cache_manager = CacheManager() if config.enable_caching else None
        
        # 性能监控
        self.request_count = Counter('model_requests_total', 'Total model requests')
        self.inference_time = Histogram('model_inference_seconds', 'Model inference time')
        self.queue_size = Gauge('model_queue_size', 'Model processing queue size')
        
        # 初始化模型
        self._load_model()
        
        # 启动后台任务
        self.background_tasks = []
    
    def _load_model(self):
        """加载模型"""
        try:
            # 这里应该加载实际的MiniGPT模型
            logging.info(f"加载模型: {self.config.model_path}")
            # self.model = torch.load(self.config.model_path)
            # self.tokenizer = load_tokenizer(self.config.tokenizer_path)
            logging.info("模型加载完成")
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise
    
    async def predict(self, input_text: str, **kwargs) -> Dict:
        """模型预测"""
        start_time = time.time()
        
        # 构建请求数据
        request_data = {
            'input_text': input_text,
            'kwargs': kwargs,
            'timestamp': start_time
        }
        
        # 检查缓存
        if self.cache_manager:
            cached_result = await self.cache_manager.get(request_data)
            if cached_result:
                return cached_result
        
        # 创建响应Future
        response_future = asyncio.Future()
        
        # 添加到批处理队列
        if self.config.enable_batching:
            await self.request_batcher.add_request(request_data, response_future)
            result = await response_future
        else:
            result = await self._single_predict(request_data)
        
        # 更新缓存
        if self.cache_manager and result:
            await self.cache_manager.set(request_data, result)
        
        # 更新监控指标
        inference_time = time.time() - start_time
        self.request_count.inc()
        self.inference_time.observe(inference_time)
        
        return result
    
    async def _single_predict(self, request_data: Dict) -> Dict:
        """单个预测"""
        # 模拟模型推理
        await asyncio.sleep(0.1)  # 模拟推理时间
        
        return {
            'output_text': f"Response to: {request_data['input_text']}",
            'model_info': {
                'name': self.config.service_name,
                'timestamp': time.time()
            }
        }
    
    async def _batch_predict(self, batch_requests: List[Dict]) -> List[Dict]:
        """批量预测"""
        # 提取输入文本
        input_texts = [req['data']['input_text'] for req in batch_requests]
        
        # 模拟批量推理
        await asyncio.sleep(0.05 * len(input_texts))  # 模拟批量推理时间
        
        # 生成结果
        results = []
        for i, req in enumerate(batch_requests):
            result = {
                'output_text': f"Batch response to: {input_texts[i]}",
                'batch_info': {
                    'batch_size': len(batch_requests),
                    'position': i,
                    'timestamp': time.time()
                }
            }
            results.append(result)
        
        return results
    
    async def start_batch_processor(self):
        """启动批处理器"""
        if not self.config.enable_batching:
            return
        
        async def batch_worker():
            while True:
                try:
                    batch_requests = await self.request_batcher.get_batch()
                    self.queue_size.set(len(batch_requests))
                    
                    # 批量推理
                    results = await self._batch_predict(batch_requests)
                    
                    # 返回结果
                    for req, result in zip(batch_requests, results):
                        if not req['future'].done():
                            req['future'].set_result(result)
                            
                except Exception as e:
                    logging.error(f"批处理错误: {e}")
                    # 处理错误情况
                    if 'batch_requests' in locals():
                        for req in batch_requests:
                            if not req['future'].done():
                                req['future'].set_exception(e)
        
        # 启动批处理工作者
        for _ in range(self.config.workers):
            task = asyncio.create_task(batch_worker())
            self.background_tasks.append(task)
        
        # 启动超时监控
        timeout_task = asyncio.create_task(self.request_batcher.start_timeout_monitor())
        self.background_tasks.append(timeout_task)

## 3.2 容器化与编排技术

### Docker化部署的数学模型

**资源利用率优化**：
设容器 $i$ 的CPU需求为 $c_i$，内存需求为 $m_i$，则资源分配问题为：
$$\max \sum_{i=1}^n x_i \quad \text{s.t.} \begin{cases}
\sum_{i=1}^n x_i c_i \leq C \\
\sum_{i=1}^n x_i m_i \leq M \\
x_i \in \{0, 1\}
\end{cases}$$

**容器调度的图论模型**：
将调度问题建模为二分图匹配：$G = (V_{containers} \cup V_{nodes}, E)$，目标是找到最大权重匹配。

**服务发现的概率模型**：
在分布式环境中，服务发现成功的概率为：
$$P(\text{discovery}) = 1 - \prod_{i=1}^n (1 - p_i)$$

其中 $p_i$ 是第 $i$ 个注册中心的可用概率。

```python
import docker
import yaml
from kubernetes import client, config
from kubernetes.client import V1Pod, V1Service, V1Deployment
import subprocess
import os
from pathlib import Path

class DockerManager:
    """Docker管理器"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.logger = logging.getLogger(__name__)
    
    def build_image(self, 
                   dockerfile_path: str, 
                   image_name: str, 
                   tag: str = "latest",
                   build_args: Optional[Dict] = None) -> str:
        """构建Docker镜像"""
        
        full_image_name = f"{image_name}:{tag}"
        
        try:
            self.logger.info(f"开始构建镜像: {full_image_name}")
            
            # 构建镜像
            image, logs = self.client.images.build(
                path=str(Path(dockerfile_path).parent),
                dockerfile=Path(dockerfile_path).name,
                tag=full_image_name,
                buildargs=build_args or {},
                rm=True,
                forcerm=True
            )
            
            # 打印构建日志
            for log in logs:
                if 'stream' in log:
                    self.logger.info(log['stream'].strip())
            
            self.logger.info(f"镜像构建完成: {full_image_name}")
            return image.id
            
        except Exception as e:
            self.logger.error(f"镜像构建失败: {e}")
            raise
    
    def run_container(self, 
                     image_name: str,
                     container_name: str,
                     ports: Optional[Dict] = None,
                     environment: Optional[Dict] = None,
                     volumes: Optional[Dict] = None,
                     command: Optional[str] = None) -> str:
        """运行容器"""
        
        try:
            self.logger.info(f"启动容器: {container_name}")
            
            container = self.client.containers.run(
                image=image_name,
                name=container_name,
                ports=ports or {},
                environment=environment or {},
                volumes=volumes or {},
                command=command,
                detach=True,
                remove=False,
                restart_policy={"Name": "unless-stopped"}
            )
            
            self.logger.info(f"容器启动成功: {container.id}")
            return container.id
            
        except Exception as e:
            self.logger.error(f"容器启动失败: {e}")
            raise
    
    def stop_container(self, container_name: str):
        """停止容器"""
        try:
            container = self.client.containers.get(container_name)
            container.stop()
            self.logger.info(f"容器已停止: {container_name}")
        except Exception as e:
            self.logger.error(f"停止容器失败: {e}")
    
    def create_dockerfile(self, 
                         service_config: ServiceConfig,
                         output_path: str = "./Dockerfile"):
        """创建Dockerfile"""
        
        dockerfile_content = f"""
# MiniGPT服务Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# 复制模型文件 (如果本地有的话)
# COPY models/ ./models/

# 设置环境变量
ENV PYTHONPATH=/app
ENV SERVICE_NAME={service_config.service_name}
ENV SERVICE_PORT={service_config.port}
ENV MODEL_PATH={service_config.model_path}
ENV TOKENIZER_PATH={service_config.tokenizer_path}

# 暴露端口
EXPOSE {service_config.port}
EXPOSE {service_config.metrics_port}

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{service_config.port}/health || exit 1

# 启动命令
CMD ["python", "scripts/service.py"]
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content.strip())
        
        self.logger.info(f"Dockerfile已创建: {output_path}")

class KubernetesManager:
    """Kubernetes管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            config.load_kube_config(config_file=config_path)
        else:
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api() 
        self.logger = logging.getLogger(__name__)
    
    def create_deployment(self, 
                         service_config: ServiceConfig,
                         image_name: str,
                         replicas: int = 3,
                         namespace: str = "default") -> str:
        """创建Deployment"""
        
        deployment_name = f"{service_config.service_name}-deployment"
        
        # 定义容器
        container = client.V1Container(
            name=service_config.service_name,
            image=image_name,
            ports=[
                client.V1ContainerPort(container_port=service_config.port),
                client.V1ContainerPort(container_port=service_config.metrics_port, name="metrics")
            ],
            env=[
                client.V1EnvVar(name="SERVICE_NAME", value=service_config.service_name),
                client.V1EnvVar(name="SERVICE_PORT", value=str(service_config.port)),
                client.V1EnvVar(name="MODEL_PATH", value=service_config.model_path),
                client.V1EnvVar(name="TOKENIZER_PATH", value=service_config.tokenizer_path),
            ],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "500m", "memory": "1Gi"},
                limits={"cpu": "2", "memory": "4Gi"}
            ),
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path="/health",
                    port=service_config.port
                ),
                initial_delay_seconds=30,
                period_seconds=10
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path="/ready",
                    port=service_config.port
                ),
                initial_delay_seconds=5,
                period_seconds=5
            )
        )
        
        # 定义Pod模板
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"app": service_config.service_name}
            ),
            spec=client.V1PodSpec(containers=[container])
        )
        
        # 定义Deployment规格
        spec = client.V1DeploymentSpec(
            replicas=replicas,
            selector=client.V1LabelSelector(
                match_labels={"app": service_config.service_name}
            ),
            template=template,
            strategy=client.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=client.V1RollingUpdateDeployment(
                    max_surge="25%",
                    max_unavailable="25%"
                )
            )
        )
        
        # 创建Deployment
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=spec
        )
        
        try:
            self.apps_v1.create_namespaced_deployment(
                body=deployment,
                namespace=namespace
            )
            self.logger.info(f"Deployment创建成功: {deployment_name}")
            return deployment_name
        except Exception as e:
            self.logger.error(f"Deployment创建失败: {e}")
            raise
    
    def create_service(self, 
                      service_config: ServiceConfig,
                      namespace: str = "default",
                      service_type: str = "ClusterIP") -> str:
        """创建Service"""
        
        service_name = f"{service_config.service_name}-service"
        
        # 定义Service规格
        spec = client.V1ServiceSpec(
            selector={"app": service_config.service_name},
            ports=[
                client.V1ServicePort(
                    name="http",
                    port=80,
                    target_port=service_config.port,
                    protocol="TCP"
                ),
                client.V1ServicePort(
                    name="metrics",
                    port=9090,
                    target_port=service_config.metrics_port,
                    protocol="TCP"
                )
            ],
            type=service_type
        )
        
        # 创建Service
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=service_name,
                labels={"app": service_config.service_name}
            ),
            spec=spec
        )
        
        try:
            self.v1.create_namespaced_service(
                body=service,
                namespace=namespace
            )
            self.logger.info(f"Service创建成功: {service_name}")
            return service_name
        except Exception as e:
            self.logger.error(f"Service创建失败: {e}")
            raise
    
    def create_horizontal_pod_autoscaler(self,
                                       deployment_name: str,
                                       min_replicas: int = 2,
                                       max_replicas: int = 10,
                                       target_cpu_percent: int = 70,
                                       namespace: str = "default") -> str:
        """创建水平Pod自动扩缩容器"""
        
        hpa_name = f"{deployment_name}-hpa"
        
        # 使用autoscaling/v2 API
        autoscaling_v2 = client.AutoscalingV2Api()
        
        hpa = client.V2HorizontalPodAutoscaler(
            api_version="autoscaling/v2",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(name=hpa_name),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deployment_name
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=target_cpu_percent
                            )
                        )
                    )
                ]
            )
        )
        
        try:
            autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                body=hpa,
                namespace=namespace
            )
            self.logger.info(f"HPA创建成功: {hpa_name}")
            return hpa_name
        except Exception as e:
            self.logger.error(f"HPA创建失败: {e}")
            raise
    
    def create_config_map(self, 
                         name: str,
                         data: Dict[str, str],
                         namespace: str = "default") -> str:
        """创建ConfigMap"""
        
        config_map = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(name=name),
            data=data
        )
        
        try:
            self.v1.create_namespaced_config_map(
                body=config_map,
                namespace=namespace
            )
            self.logger.info(f"ConfigMap创建成功: {name}")
            return name
        except Exception as e:
            self.logger.error(f"ConfigMap创建失败: {e}")
            raise

## 3.3 A/B测试框架的统计学原理

### 实验设计的数学基础

**功效分析**：
对于比较两个比例的A/B测试，所需样本量为：
$$n = \frac{(z_{\alpha/2} + z_\beta)^2 (p_1(1-p_1) + p_2(1-p_2))}{(p_1 - p_2)^2}$$

其中 $z_{\alpha/2}$ 是显著性水平对应的临界值，$z_\beta$ 是统计功效对应的临界值。

**多重比较校正**：
使用Bonferroni校正，调整后的显著性水平为：
$$\alpha_{adjusted} = \frac{\alpha}{k}$$

其中 $k$ 是比较次数。

**贝叶斯A/B测试**：
后验概率更新：
$$P(p_A > p_B | \text{data}) = \int_{0}^{1} \int_{0}^{p_A} \text{Beta}(p_A | \alpha_A, \beta_A) \text{Beta}(p_B | \alpha_B, \beta_B) dp_B dp_A$$

```python
import numpy as np
from scipy import stats
from scipy.stats import beta
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import hashlib
import json

class ExperimentType(Enum):
    """实验类型"""
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"

class MetricType(Enum):
    """指标类型"""
    CONVERSION = "conversion"  # 转化率
    CONTINUOUS = "continuous"  # 连续变量
    COUNT = "count"            # 计数
    TIME_TO_EVENT = "time_to_event"  # 生存分析

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str
    experiment_type: ExperimentType
    metric_type: MetricType
    variants: List[str]
    traffic_allocation: Dict[str, float]
    
    # 统计参数
    significance_level: float = 0.05
    statistical_power: float = 0.8
    minimum_detectable_effect: float = 0.05
    
    # 实验设置
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_duration_days: int = 30
    min_sample_size: int = 1000

@dataclass  
class ExperimentResult:
    """实验结果"""
    variant: str
    metric_value: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    p_value: float
    is_significant: bool

class TrafficSplitter:
    """流量分割器"""
    
    def __init__(self, allocation: Dict[str, float]):
        self.allocation = allocation
        self.cumulative_weights = self._calculate_cumulative_weights()
    
    def _calculate_cumulative_weights(self) -> List[Tuple[str, float]]:
        """计算累积权重"""
        cumulative = 0
        cumulative_weights = []
        
        for variant, weight in self.allocation.items():
            cumulative += weight
            cumulative_weights.append((variant, cumulative))
        
        return cumulative_weights
    
    def assign_variant(self, user_id: str) -> str:
        """分配变体"""
        # 使用一致性哈希确保用户总是被分配到相同变体
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000.0
        
        for variant, cumulative_weight in self.cumulative_weights:
            if random_value <= cumulative_weight:
                return variant
        
        # 默认返回最后一个变体
        return self.cumulative_weights[-1][0]

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def analyze_conversion_rate(self, 
                              control_conversions: int,
                              control_total: int,
                              treatment_conversions: int,
                              treatment_total: int) -> Dict:
        """分析转化率差异"""
        
        # 计算转化率
        control_rate = control_conversions / control_total
        treatment_rate = treatment_conversions / treatment_total
        
        # 计算标准误差
        control_se = np.sqrt(control_rate * (1 - control_rate) / control_total)
        treatment_se = np.sqrt(treatment_rate * (1 - treatment_rate) / treatment_total)
        
        # 计算差异的标准误差
        diff_se = np.sqrt(control_se**2 + treatment_se**2)
        
        # 计算z分数
        difference = treatment_rate - control_rate
        z_score = difference / diff_se if diff_se > 0 else 0
        
        # 计算p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # 计算置信区间
        margin_of_error = stats.norm.ppf(1 - self.significance_level/2) * diff_se
        ci_lower = difference - margin_of_error
        ci_upper = difference + margin_of_error
        
        # 计算统计功效
        effect_size = abs(difference) / np.sqrt(control_rate * (1 - control_rate))
        power = self._calculate_power(effect_size, control_total + treatment_total)
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'difference': difference,
            'relative_improvement': difference / control_rate if control_rate > 0 else 0,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'confidence_interval': (ci_lower, ci_upper),
            'statistical_power': power,
            'sample_size': control_total + treatment_total
        }
    
    def analyze_continuous_metric(self,
                                control_values: List[float],
                                treatment_values: List[float]) -> Dict:
        """分析连续指标差异"""
        
        # 基本统计量
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)
        
        # t检验
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        # 计算效应大小 (Cohen's d)
        pooled_std = np.sqrt(((len(control_values) - 1) * control_std**2 + 
                             (len(treatment_values) - 1) * treatment_std**2) / 
                            (len(control_values) + len(treatment_values) - 2))
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        # 置信区间
        se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
        df = len(control_values) + len(treatment_values) - 2
        t_critical = stats.t.ppf(1 - self.significance_level/2, df)
        margin_of_error = t_critical * se_diff
        
        difference = treatment_mean - control_mean
        ci_lower = difference - margin_of_error
        ci_upper = difference + margin_of_error
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'difference': difference,
            'relative_improvement': difference / control_mean if control_mean != 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'confidence_interval': (ci_lower, ci_upper),
            'cohens_d': cohens_d,
            'sample_size': len(control_values) + len(treatment_values)
        }
    
    def bayesian_ab_test(self,
                        control_conversions: int,
                        control_total: int,
                        treatment_conversions: int,
                        treatment_total: int,
                        prior_alpha: float = 1,
                        prior_beta: float = 1) -> Dict:
        """贝叶斯A/B测试分析"""
        
        # 后验分布参数
        control_alpha = prior_alpha + control_conversions
        control_beta = prior_beta + control_total - control_conversions
        treatment_alpha = prior_alpha + treatment_conversions
        treatment_beta = prior_beta + treatment_total - treatment_conversions
        
        # 后验均值
        control_mean = control_alpha / (control_alpha + control_beta)
        treatment_mean = treatment_alpha / (treatment_alpha + treatment_beta)
        
        # 蒙特卡罗模拟计算P(treatment > control)
        n_simulations = 100000
        control_samples = beta.rvs(control_alpha, control_beta, size=n_simulations)
        treatment_samples = beta.rvs(treatment_alpha, treatment_beta, size=n_simulations)
        
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        
        # 计算可信区间
        treatment_ci = beta.interval(1 - self.significance_level, treatment_alpha, treatment_beta)
        control_ci = beta.interval(1 - self.significance_level, control_alpha, control_beta)
        
        # 预期损失
        def expected_loss(samples_a, samples_b):
            diff = samples_b - samples_a
            return np.mean(np.maximum(0, -diff))
        
        loss_if_choose_treatment = expected_loss(treatment_samples, control_samples)
        loss_if_choose_control = expected_loss(control_samples, treatment_samples)
        
        return {
            'control_posterior_mean': control_mean,
            'treatment_posterior_mean': treatment_mean,
            'prob_treatment_better': prob_treatment_better,
            'control_credible_interval': control_ci,
            'treatment_credible_interval': treatment_ci,
            'expected_loss_treatment': loss_if_choose_treatment,
            'expected_loss_control': loss_if_choose_control,
            'posterior_params': {
                'control': (control_alpha, control_beta),
                'treatment': (treatment_alpha, treatment_beta)
            }
        }
    
    def _calculate_power(self, effect_size: float, total_sample_size: int) -> float:
        """计算统计功效"""
        critical_value = stats.norm.ppf(1 - self.significance_level/2)
        se = np.sqrt(2 / total_sample_size)  # 简化的标准误差
        z_beta = critical_value - effect_size / se
        power = 1 - stats.norm.cdf(z_beta)
        return max(0, min(1, power))  # 确保在[0,1]范围内
    
    def sample_size_calculation(self,
                              baseline_rate: float,
                              minimum_detectable_effect: float,
                              significance_level: float = 0.05,
                              power: float = 0.8) -> int:
        """样本量计算"""
        
        # 计算效应大小
        treatment_rate = baseline_rate * (1 + minimum_detectable_effect)
        
        # 使用正态近似
        z_alpha = stats.norm.ppf(1 - significance_level/2)
        z_beta = stats.norm.ppf(power)
        
        # 计算所需样本量
        pooled_p = (baseline_rate + treatment_rate) / 2
        n = (z_alpha + z_beta)**2 * 2 * pooled_p * (1 - pooled_p) / (treatment_rate - baseline_rate)**2
        
        return int(np.ceil(n))

class ABTestFramework:
    """A/B测试框架"""
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_data: Dict[str, List[Dict]] = {}
        self.traffic_splitters: Dict[str, TrafficSplitter] = {}
        self.analyzer = StatisticalAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """创建实验"""
        
        # 验证流量分配
        total_allocation = sum(config.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 1e-6:
            raise ValueError(f"流量分配总和必须为1.0，当前为{total_allocation}")
        
        # 存储实验配置
        self.experiments[config.name] = config
        self.experiment_data[config.name] = []
        
        # 创建流量分割器
        self.traffic_splitters[config.name] = TrafficSplitter(config.traffic_allocation)
        
        self.logger.info(f"实验创建成功: {config.name}")
        return config.name
    
    def assign_user_to_variant(self, experiment_name: str, user_id: str) -> str:
        """为用户分配变体"""
        
        if experiment_name not in self.traffic_splitters:
            raise ValueError(f"实验不存在: {experiment_name}")
        
        variant = self.traffic_splitters[experiment_name].assign_variant(user_id)
        return variant
    
    def record_event(self,
                    experiment_name: str,
                    user_id: str,
                    variant: str,
                    metric_value: float,
                    timestamp: Optional[datetime] = None):
        """记录事件"""
        
        if experiment_name not in self.experiment_data:
            raise ValueError(f"实验不存在: {experiment_name}")
        
        event = {
            'user_id': user_id,
            'variant': variant,
            'metric_value': metric_value,
            'timestamp': timestamp or datetime.now()
        }
        
        self.experiment_data[experiment_name].append(event)
    
    def analyze_experiment(self, experiment_name: str) -> Dict:
        """分析实验结果"""
        
        if experiment_name not in self.experiments:
            raise ValueError(f"实验不存在: {experiment_name}")
        
        config = self.experiments[experiment_name]
        data = self.experiment_data[experiment_name]
        
        if not data:
            return {'error': '无实验数据'}
        
        # 按变体分组数据
        variant_data = defaultdict(list)
        for event in data:
            variant_data[event['variant']].append(event['metric_value'])
        
        results = {}
        
        if config.metric_type == MetricType.CONVERSION:
            # 转化率分析
            if len(config.variants) == 2:
                control_variant = config.variants[0]
                treatment_variant = config.variants[1]
                
                control_data = variant_data[control_variant]
                treatment_data = variant_data[treatment_variant]
                
                # 假设转化用1表示，未转化用0表示
                control_conversions = sum(control_data)
                treatment_conversions = sum(treatment_data)
                
                analysis = self.analyzer.analyze_conversion_rate(
                    control_conversions, len(control_data),
                    treatment_conversions, len(treatment_data)
                )
                
                results['analysis'] = analysis
        
        elif config.metric_type == MetricType.CONTINUOUS:
            # 连续指标分析
            if len(config.variants) == 2:
                control_variant = config.variants[0]
                treatment_variant = config.variants[1]
                
                control_data = variant_data[control_variant]
                treatment_data = variant_data[treatment_variant]
                
                analysis = self.analyzer.analyze_continuous_metric(
                    control_data, treatment_data
                )
                
                results['analysis'] = analysis
        
        # 添加实验元信息
        results['experiment_config'] = {
            'name': config.name,
            'variants': config.variants,
            'metric_type': config.metric_type.value,
            'total_samples': len(data)
        }
        
        # 变体分布统计
        results['variant_distribution'] = {
            variant: len(values) for variant, values in variant_data.items()
        }
        
        return results
    
    def get_experiment_status(self, experiment_name: str) -> Dict:
        """获取实验状态"""
        
        if experiment_name not in self.experiments:
            return {'error': '实验不存在'}
        
        config = self.experiments[experiment_name]
        data = self.experiment_data[experiment_name]
        
        # 计算运行时间
        if data:
            first_event = min(event['timestamp'] for event in data)
            last_event = max(event['timestamp'] for event in data)
            duration = last_event - first_event
        else:
            duration = timedelta(0)
        
        # 计算样本量
        variant_counts = defaultdict(int)
        for event in data:
            variant_counts[event['variant']] += 1
        
        # 判断是否达到最小样本量
        min_samples_reached = all(
            count >= config.min_sample_size for count in variant_counts.values()
        )
        
        return {
            'experiment_name': experiment_name,
            'total_samples': len(data),
            'variant_samples': dict(variant_counts),
            'duration_days': duration.days,
            'min_samples_reached': min_samples_reached,
            'is_active': True,  # 简化实现
            'config': {
                'significance_level': config.significance_level,
                'statistical_power': config.statistical_power,
                'minimum_detectable_effect': config.minimum_detectable_effect
            }
        }

## 应用指导

### 部署最佳实践

1. **微服务架构设计**：
   - 按业务边界拆分服务
   - 实现服务间的松耦合
   - 建立标准化的API契约
   - 实施服务版本管理

2. **容器化策略**：
   - 构建最小化镜像
   - 实现多阶段构建优化
   - 配置健康检查和优雅关闭
   - 实施安全最佳实践

3. **Kubernetes部署**：
   - 使用资源限制和请求
   - 配置水平自动扩缩容
   - 实现滚动更新策略
   - 建立监控和告警体系

4. **A/B测试实施**：
   - 建立严格的实验设计流程
   - 确保统计显著性
   - 实施多重比较校正
   - 建立实验结果决策框架

## 性能与可扩展性

### 关键指标

- **服务延迟**：P50, P95, P99响应时间
- **吞吐量**：每秒请求数(RPS)和每秒事务数(TPS)  
- **可用性**：系统正常运行时间百分比
- **资源利用率**：CPU、内存、网络使用率

### 扩展策略

1. **水平扩缩容**：
   - 基于CPU/内存使用率自动扩缩容
   - 基于自定义指标(如队列长度)扩缩容
   - 预测性扩缩容

2. **负载均衡优化**：
   - 智能路由策略
   - 会话亲和性
   - 跨可用区负载分配

3. **缓存策略**：
   - 多级缓存架构
   - 缓存预热和失效策略
   - 缓存一致性保证

## 扩展阅读

- 《Microservices Patterns》- 微服务架构模式
- 《Kubernetes in Action》- Kubernetes实战指南
- 《Building Microservices》- 微服务设计
- 《Site Reliability Engineering》- 站点可靠性工程

---

*"部署不是终点，而是新的起点。真正的挑战在于构建一个能够持续演进、稳定可靠的生产系统。"* 🚀