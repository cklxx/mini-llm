# 01 训练基础设施与可扩展性

> **从单机到集群：构建可扩展的大规模训练系统**

## 核心思想

训练基础设施是支撑大规模语言模型训练的技术基石。随着模型规模的指数级增长，单机训练已无法满足需求，分布式训练成为必然选择。然而，分布式系统的复杂性远超单机环境——从数据分割到梯度同步，从故障处理到资源调度，每个环节都需要精心设计和优化。

**关键洞察**：
- **并行化策略**：数据并行、模型并行、流水线并行的数学原理与工程实现
- **通信优化**：减少通信开销，提高带宽利用率
- **容错机制**：在分布式环境中保证训练的稳定性和可恢复性
- **资源管理**：智能调度和高效利用昂贵的GPU资源

从数学角度看，分布式训练是将单机优化问题分解为多个子问题的并行求解，其核心挑战在于保证收敛性的同时最大化并行效率。

## 1.1 分布式训练架构的数学理论

### 数据并行的数学建模

**数据并行的基本思想**：
将批数据 $\mathcal{B}$ 分割为 $N$ 个子批次：$\mathcal{B} = \bigcup_{i=1}^{N} \mathcal{B}_i$，每个设备计算局部梯度：
$$\mathbf{g}_i = \frac{1}{|\mathcal{B}_i|} \sum_{x \in \mathcal{B}_i} \nabla_\theta L(f_\theta(x), y)$$

**全局梯度聚合**：
$$\mathbf{g}_{global} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{g}_i$$

**收敛性分析**：
设真实梯度为 $\mathbf{g}^*$，数据并行梯度为 $\mathbf{g}_{dp}$，则方差为：
$$\text{Var}[\mathbf{g}_{dp}] = \frac{1}{N} \text{Var}[\mathbf{g}^*] + \frac{N-1}{N} \text{Bias}^2$$

其中偏差项来源于不同设备上数据分布的差异。

### 模型并行的张量分割理论

**线性层的分割**：
对于线性变换 $\mathbf{y} = \mathbf{x} \mathbf{W}$，可以按行或列分割权重矩阵：

**列并行**：$\mathbf{W} = [\mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_N]$
$$\mathbf{y} = \sum_{i=1}^{N} \mathbf{x} \mathbf{W}_i$$

**行并行**：$\mathbf{W} = \begin{bmatrix} \mathbf{W}_1 \\ \mathbf{W}_2 \\ \vdots \\ \mathbf{W}_N \end{bmatrix}$
$$\mathbf{y}_i = \mathbf{x}_i \mathbf{W}_i, \quad \mathbf{y} = \text{concat}([\mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_N])$$

**注意力机制的分割**：
多头注意力天然支持并行：
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \mathbf{W}^O$$

每个头可以分配到不同设备并行计算。

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import os
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import GPUtil

class ParallelismType(Enum):
    """并行类型枚举"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"

@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_size: str = "small"
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    
    # 分布式配置
    world_size: int = 1
    local_rank: int = 0
    parallelism_type: ParallelismType = ParallelismType.DATA_PARALLEL
    
    # 系统配置
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # 故障恢复配置
    checkpoint_interval: int = 1000
    max_checkpoints: int = 3

class DistributedTrainingManager:
    """分布式训练管理器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.device = None
        self.process_group = None
        self.performance_metrics = {
            'throughput': [],
            'memory_usage': [],
            'communication_time': [],
            'computation_time': []
        }
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Rank {self.config.local_rank}] %(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def initialize_distributed(self, backend: str = 'nccl'):
        """初始化分布式环境"""
        
        self.logger.info(f"初始化分布式训练: backend={backend}, world_size={self.config.world_size}")
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.local_rank)
        
        # 初始化进程组
        if self.config.world_size > 1:
            dist.init_process_group(
                backend=backend,
                rank=self.config.local_rank,
                world_size=self.config.world_size
            )
            self.process_group = dist.group.WORLD
        
        # 设置设备
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.config.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        self.logger.info(f"分布式环境初始化完成: device={self.device}")
    
    def create_data_parallel_model(self, model: nn.Module) -> nn.Module:
        """创建数据并行模型"""
        
        model = model.to(self.device)
        
        if self.config.world_size > 1:
            # 使用DistributedDataParallel
            model = DDP(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=False
            )
            self.logger.info("使用DistributedDataParallel包装模型")
        
        return model
    
    def create_model_parallel_model(self, model: nn.Module) -> nn.Module:
        """创建模型并行模型"""
        
        # 简化实现：将模型的不同层分配到不同设备
        if self.config.world_size > 1:
            layers_per_device = len(model.layers) // self.config.world_size
            start_layer = self.config.local_rank * layers_per_device
            end_layer = min((self.config.local_rank + 1) * layers_per_device, 
                           len(model.layers))
            
            # 只保留分配给当前设备的层
            device_layers = model.layers[start_layer:end_layer]
            
            class ModelParallelWrapper(nn.Module):
                def __init__(self, layers, device, rank, world_size):
                    super().__init__()
                    self.layers = nn.ModuleList(layers)
                    self.device = device
                    self.rank = rank
                    self.world_size = world_size
                    
                def forward(self, x):
                    x = x.to(self.device)
                    
                    for layer in self.layers:
                        x = layer(x)
                    
                    # 发送到下一个设备
                    if self.rank < self.world_size - 1:
                        dist.send(x.contiguous(), dst=self.rank + 1)
                        return None
                    else:
                        return x
            
            wrapped_model = ModelParallelWrapper(
                device_layers, self.device, 
                self.config.local_rank, self.config.world_size
            )
            
            self.logger.info(f"模型并行: 设备{self.config.local_rank}负责层{start_layer}-{end_layer}")
            return wrapped_model
        
        return model.to(self.device)

class PipelineParallelTrainer:
    """流水线并行训练器"""
    
    def __init__(self, config: TrainingConfig, num_microbatches: int = 4):
        self.config = config
        self.num_microbatches = num_microbatches
        self.logger = logging.getLogger(__name__)
        
    def create_pipeline_schedule(self, num_stages: int, num_microbatches: int) -> List[List[str]]:
        """创建流水线调度"""
        
        # GPipe调度策略
        schedule = [[] for _ in range(num_stages)]
        
        # Forward pass
        for micro_batch in range(num_microbatches):
            for stage in range(num_stages):
                schedule[stage].append(f'F{micro_batch}')
        
        # Backward pass (反向)
        for micro_batch in range(num_microbatches-1, -1, -1):
            for stage in range(num_stages-1, -1, -1):
                schedule[stage].append(f'B{micro_batch}')
        
        return schedule
    
    def execute_pipeline_stage(self, 
                             stage_model: nn.Module,
                             schedule: List[str],
                             input_queue: queue.Queue,
                             output_queue: queue.Queue):
        """执行流水线阶段"""
        
        activations = {}  # 保存激活值用于反向传播
        
        for operation in schedule:
            op_type = operation[0]  # 'F' 或 'B'
            micro_batch_id = int(operation[1:])
            
            if op_type == 'F':
                # Forward pass
                if not input_queue.empty():
                    input_data = input_queue.get()
                    
                    with torch.no_grad() if not stage_model.training else torch.enable_grad():
                        output = stage_model(input_data)
                        activations[micro_batch_id] = output.detach()
                        
                        if not output_queue.full():
                            output_queue.put(output)
            
            elif op_type == 'B':
                # Backward pass
                if micro_batch_id in activations:
                    activation = activations[micro_batch_id]
                    # 执行反向传播
                    activation.backward()
                    del activations[micro_batch_id]
    
    def analyze_pipeline_efficiency(self, 
                                  num_stages: int, 
                                  num_microbatches: int,
                                  stage_times: List[float]) -> Dict:
        """分析流水线效率"""
        
        # 计算理论最优时间
        max_stage_time = max(stage_times)
        sequential_time = sum(stage_times) * num_microbatches
        
        # 流水线执行时间
        pipeline_time = (num_stages - 1) * max_stage_time + num_microbatches * max_stage_time
        
        # 效率分析
        speedup = sequential_time / pipeline_time
        efficiency = speedup / num_stages
        
        # 气泡时间分析
        bubble_time = (num_stages - 1) * max_stage_time
        bubble_ratio = bubble_time / pipeline_time
        
        return {
            'speedup': speedup,
            'efficiency': efficiency,
            'bubble_ratio': bubble_ratio,
            'pipeline_time': pipeline_time,
            'sequential_time': sequential_time,
            'max_stage_time': max_stage_time
        }

## 1.2 通信优化与带宽管理

### All-Reduce算法的数学分析

**Ring All-Reduce算法**：
对于 $N$ 个设备，每个设备有参数向量 $\mathbf{p}_i$，Ring All-Reduce的通信复杂度为：
$$T_{comm} = 2 \cdot \frac{N-1}{N} \cdot \frac{M}{B}$$

其中 $M$ 是参数总量，$B$ 是带宽。

**Butterfly All-Reduce算法**：
通信复杂度为：
$$T_{comm} = \log_2(N) \cdot \frac{M}{B}$$

适用于设备数量较少的场景。

class CommunicationOptimizer:
    """通信优化器"""
    
    def __init__(self, world_size: int, bandwidth_mbps: float = 1000):
        self.world_size = world_size
        self.bandwidth_mbps = bandwidth_mbps
        self.communication_stats = {
            'total_bytes': 0,
            'total_time': 0.0,
            'operation_count': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def estimate_allreduce_time(self, 
                              tensor_size_bytes: int,
                              algorithm: str = 'ring') -> float:
        """估算All-Reduce通信时间"""
        
        bandwidth_bps = self.bandwidth_mbps * 1024 * 1024  # 转换为bps
        
        if algorithm == 'ring':
            # Ring All-Reduce: 2 * (N-1)/N * M/B
            comm_time = 2 * (self.world_size - 1) / self.world_size * tensor_size_bytes / bandwidth_bps
        
        elif algorithm == 'butterfly':
            # Butterfly All-Reduce: log2(N) * M/B
            comm_time = np.log2(self.world_size) * tensor_size_bytes / bandwidth_bps
        
        elif algorithm == 'tree':
            # Tree All-Reduce: 2 * log2(N) * M/B
            comm_time = 2 * np.log2(self.world_size) * tensor_size_bytes / bandwidth_bps
        
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        return comm_time
    
    def optimize_gradient_communication(self, 
                                      gradients: Dict[str, torch.Tensor],
                                      compression_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """优化梯度通信"""
        
        start_time = time.time()
        
        optimized_gradients = {}
        total_original_size = 0
        total_compressed_size = 0
        
        for name, grad in gradients.items():
            total_original_size += grad.numel() * grad.element_size()
            
            # 梯度压缩
            if compression_ratio < 1.0:
                compressed_grad = self._compress_gradient(grad, compression_ratio)
                optimized_gradients[name] = compressed_grad
                total_compressed_size += compressed_grad.numel() * compressed_grad.element_size()
            else:
                optimized_gradients[name] = grad
                total_compressed_size += grad.numel() * grad.element_size()
        
        # 执行All-Reduce
        for name, grad in optimized_gradients.items():
            if self.world_size > 1:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad.div_(self.world_size)
        
        comm_time = time.time() - start_time
        
        # 更新统计信息
        self.communication_stats['total_bytes'] += total_compressed_size
        self.communication_stats['total_time'] += comm_time
        self.communication_stats['operation_count'] += 1
        
        compression_rate = total_compressed_size / total_original_size
        self.logger.info(f"梯度通信完成: 压缩率={compression_rate:.3f}, 时间={comm_time:.3f}s")
        
        return optimized_gradients
    
    def _compress_gradient(self, gradient: torch.Tensor, ratio: float) -> torch.Tensor:
        """梯度压缩"""
        
        # Top-K压缩
        if ratio < 1.0:
            k = max(1, int(gradient.numel() * ratio))
            
            # 展平梯度
            flat_grad = gradient.view(-1)
            
            # 选择top-k元素
            _, top_k_indices = torch.topk(torch.abs(flat_grad), k)
            
            # 创建稀疏梯度
            compressed_grad = torch.zeros_like(flat_grad)
            compressed_grad[top_k_indices] = flat_grad[top_k_indices]
            
            return compressed_grad.view(gradient.shape)
        
        return gradient
    
    def analyze_communication_pattern(self) -> Dict:
        """分析通信模式"""
        
        if self.communication_stats['operation_count'] == 0:
            return {}
        
        avg_bytes_per_op = self.communication_stats['total_bytes'] / self.communication_stats['operation_count']
        avg_time_per_op = self.communication_stats['total_time'] / self.communication_stats['operation_count']
        effective_bandwidth = avg_bytes_per_op / avg_time_per_op if avg_time_per_op > 0 else 0
        
        return {
            'total_operations': self.communication_stats['operation_count'],
            'total_bytes_transferred': self.communication_stats['total_bytes'],
            'total_communication_time': self.communication_stats['total_time'],
            'average_bytes_per_operation': avg_bytes_per_op,
            'average_time_per_operation': avg_time_per_op,
            'effective_bandwidth_mbps': effective_bandwidth / (1024 * 1024),
            'bandwidth_utilization': (effective_bandwidth / (1024 * 1024)) / self.bandwidth_mbps
        }

## 1.3 资源管理与调度系统

### GPU集群资源建模

**资源约束优化**：
$$\max \sum_{i=1}^{N} u_i \cdot x_i \quad \text{s.t.} \begin{cases}
\sum_{i=1}^{N} r_i \cdot x_i \leq R \\
\sum_{i=1}^{N} m_i \cdot x_i \leq M \\
x_i \in \{0, 1\}
\end{cases}$$

其中 $u_i$ 是任务效用，$r_i, m_i$ 是资源需求，$R, M$ 是总资源。

**负载均衡指标**：
$$\text{Load Balance} = 1 - \frac{\text{std}(\{L_1, L_2, ..., L_N\})}{\text{mean}(\{L_1, L_2, ..., L_N\})}$$

其中 $L_i$ 是第 $i$ 个设备的负载。

class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.gpu_info = self._get_gpu_info()
        self.resource_history = []
        self.allocation_strategy = "greedy"
        self.logger = logging.getLogger(__name__)
        
    def _get_gpu_info(self) -> List[Dict]:
        """获取GPU信息"""
        
        gpu_info = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_available': torch.cuda.get_device_properties(i).total_memory,
                    'compute_capability': torch.cuda.get_device_properties(i).major,
                    'utilization': 0.0,
                    'temperature': 0.0
                }
                gpu_info.append(info)
        
        return gpu_info
    
    def monitor_resource_usage(self) -> Dict:
        """监控资源使用情况"""
        
        usage_info = {
            'timestamp': time.time(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'gpu_usage': []
        }
        
        # 获取GPU使用信息
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_usage = {
                    'device_id': gpu.id,
                    'utilization': gpu.load * 100,
                    'memory_usage': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature,
                    'power_usage': getattr(gpu, 'powerDraw', 0)
                }
                usage_info['gpu_usage'].append(gpu_usage)
        except:
            # 如果GPUtil不可用，使用简化信息
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                
                gpu_usage = {
                    'device_id': i,
                    'utilization': 0.0,  # 无法获取
                    'memory_usage': (memory_allocated / memory_total) * 100,
                    'temperature': 0.0,  # 无法获取
                    'power_usage': 0.0   # 无法获取
                }
                usage_info['gpu_usage'].append(gpu_usage)
        
        # 记录历史
        self.resource_history.append(usage_info)
        
        # 保持历史记录在合理范围内
        if len(self.resource_history) > 1000:
            self.resource_history = self.resource_history[-1000:]
        
        return usage_info
    
    def allocate_resources(self, 
                         task_requirements: List[Dict],
                         strategy: str = "first_fit") -> Dict:
        """资源分配"""
        
        self.logger.info(f"开始资源分配: {len(task_requirements)}个任务, 策略={strategy}")
        
        allocation_result = {
            'allocated_tasks': [],
            'failed_tasks': [],
            'resource_utilization': {}
        }
        
        available_resources = {
            'gpu_memory': [gpu['memory_available'] for gpu in self.gpu_info],
            'gpu_utilization': [gpu['utilization'] for gpu in self.gpu_info]
        }
        
        for task_id, requirements in enumerate(task_requirements):
            allocated = False
            
            if strategy == "first_fit":
                # 第一个满足条件的GPU
                for gpu_id, gpu in enumerate(self.gpu_info):
                    if self._can_allocate(gpu_id, requirements, available_resources):
                        self._allocate_to_gpu(task_id, gpu_id, requirements, available_resources)
                        allocation_result['allocated_tasks'].append({
                            'task_id': task_id,
                            'gpu_id': gpu_id,
                            'requirements': requirements
                        })
                        allocated = True
                        break
            
            elif strategy == "best_fit":
                # 选择最适合的GPU（剩余资源最少但足够）
                best_gpu = -1
                min_waste = float('inf')
                
                for gpu_id, gpu in enumerate(self.gpu_info):
                    if self._can_allocate(gpu_id, requirements, available_resources):
                        waste = available_resources['gpu_memory'][gpu_id] - requirements.get('memory', 0)
                        if waste < min_waste:
                            min_waste = waste
                            best_gpu = gpu_id
                
                if best_gpu != -1:
                    self._allocate_to_gpu(task_id, best_gpu, requirements, available_resources)
                    allocation_result['allocated_tasks'].append({
                        'task_id': task_id,
                        'gpu_id': best_gpu,
                        'requirements': requirements
                    })
                    allocated = True
            
            elif strategy == "load_balance":
                # 负载均衡：选择利用率最低的可用GPU
                best_gpu = -1
                min_utilization = float('inf')
                
                for gpu_id, gpu in enumerate(self.gpu_info):
                    if (self._can_allocate(gpu_id, requirements, available_resources) and
                        available_resources['gpu_utilization'][gpu_id] < min_utilization):
                        min_utilization = available_resources['gpu_utilization'][gpu_id]
                        best_gpu = gpu_id
                
                if best_gpu != -1:
                    self._allocate_to_gpu(task_id, best_gpu, requirements, available_resources)
                    allocation_result['allocated_tasks'].append({
                        'task_id': task_id,
                        'gpu_id': best_gpu,
                        'requirements': requirements
                    })
                    allocated = True
            
            if not allocated:
                allocation_result['failed_tasks'].append({
                    'task_id': task_id,
                    'requirements': requirements,
                    'reason': 'insufficient_resources'
                })
        
        # 计算资源利用率
        total_memory = sum(gpu['memory_total'] for gpu in self.gpu_info)
        used_memory = total_memory - sum(available_resources['gpu_memory'])
        allocation_result['resource_utilization'] = {
            'memory_utilization': used_memory / total_memory if total_memory > 0 else 0,
            'gpu_utilization': sum(available_resources['gpu_utilization']) / len(self.gpu_info)
        }
        
        self.logger.info(f"资源分配完成: {len(allocation_result['allocated_tasks'])}个任务成功分配")
        
        return allocation_result
    
    def _can_allocate(self, 
                     gpu_id: int, 
                     requirements: Dict, 
                     available_resources: Dict) -> bool:
        """检查是否可以分配"""
        
        required_memory = requirements.get('memory', 0)
        available_memory = available_resources['gpu_memory'][gpu_id]
        
        # 简化检查：只检查内存
        return available_memory >= required_memory
    
    def _allocate_to_gpu(self, 
                        task_id: int, 
                        gpu_id: int, 
                        requirements: Dict, 
                        available_resources: Dict):
        """分配任务到GPU"""
        
        required_memory = requirements.get('memory', 0)
        required_utilization = requirements.get('utilization', 10.0)
        
        # 更新可用资源
        available_resources['gpu_memory'][gpu_id] -= required_memory
        available_resources['gpu_utilization'][gpu_id] += required_utilization
    
    def optimize_resource_allocation(self, 
                                   tasks: List[Dict],
                                   optimization_objective: str = "utilization") -> Dict:
        """优化资源分配"""
        
        if optimization_objective == "utilization":
            return self._optimize_for_utilization(tasks)
        elif optimization_objective == "latency":
            return self._optimize_for_latency(tasks)
        elif optimization_objective == "throughput":
            return self._optimize_for_throughput(tasks)
        else:
            raise ValueError(f"不支持的优化目标: {optimization_objective}")
    
    def _optimize_for_utilization(self, tasks: List[Dict]) -> Dict:
        """优化GPU利用率"""
        
        # 使用贪心算法：按资源需求降序排序，优先分配大任务
        sorted_tasks = sorted(enumerate(tasks), 
                            key=lambda x: x[1].get('memory', 0), 
                            reverse=True)
        
        reordered_tasks = [task for _, task in sorted_tasks]
        
        return self.allocate_resources(reordered_tasks, strategy="best_fit")
    
    def _optimize_for_latency(self, tasks: List[Dict]) -> Dict:
        """优化延迟"""
        
        # 负载均衡策略：尽量均匀分配
        return self.allocate_resources(tasks, strategy="load_balance")
    
    def _optimize_for_throughput(self, tasks: List[Dict]) -> Dict:
        """优化吞吐量"""
        
        # 先分配小任务，提高并行度
        sorted_tasks = sorted(enumerate(tasks), 
                            key=lambda x: x[1].get('memory', 0))
        
        reordered_tasks = [task for _, task in sorted_tasks]
        
        return self.allocate_resources(reordered_tasks, strategy="first_fit")

## 1.4 容错与恢复机制

### 故障检测的数学模型

**故障概率建模**：
设设备 $i$ 在时间 $t$ 内发生故障的概率为：
$$P_{\text{fail}}(i, t) = 1 - e^{-\lambda_i t}$$

其中 $\lambda_i$ 是设备 $i$ 的故障率。

**系统可靠性**：
对于有 $N$ 个设备的系统，至少有 $k$ 个设备正常工作的概率：
$$R(t, k) = \sum_{i=k}^{N} \binom{N}{i} p^i (1-p)^{N-i}$$

其中 $p = e^{-\lambda t}$ 是单设备可靠性。

class FaultToleranceManager:
    """容错管理器"""
    
    def __init__(self, 
                 checkpoint_interval: int = 1000,
                 max_checkpoints: int = 3,
                 backup_strategy: str = "redundant"):
        
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.backup_strategy = backup_strategy
        self.checkpoint_history = []
        self.failure_history = []
        self.logger = logging.getLogger(__name__)
        
    def create_checkpoint(self, 
                         model: nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         step: int,
                         checkpoint_dir: str = "./checkpoints") -> str:
        """创建检查点"""
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        
        # 确保目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_data = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': time.time(),
            'random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        
        # 保存检查点
        start_time = time.time()
        torch.save(checkpoint_data, checkpoint_path)
        save_time = time.time() - start_time
        
        # 记录检查点信息
        checkpoint_info = {
            'path': checkpoint_path,
            'step': step,
            'timestamp': checkpoint_data['timestamp'],
            'save_time': save_time,
            'file_size': os.path.getsize(checkpoint_path)
        }
        
        self.checkpoint_history.append(checkpoint_info)
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"检查点已保存: {checkpoint_path}, 耗时{save_time:.2f}s")
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer) -> int:
        """加载检查点"""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        start_time = time.time()
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        load_time = time.time() - start_time
        
        # 恢复模型和优化器状态
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # 恢复随机数状态
        torch.set_rng_state(checkpoint_data['random_state'])
        if checkpoint_data['cuda_random_state'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint_data['cuda_random_state'])
        
        step = checkpoint_data['step']
        
        self.logger.info(f"检查点已加载: {checkpoint_path}, 恢复到步骤{step}, 耗时{load_time:.2f}s")
        
        return step
    
    def detect_failure(self, 
                      devices: List[int],
                      timeout: float = 30.0) -> List[int]:
        """检测设备故障"""
        
        failed_devices = []
        
        for device_id in devices:
            try:
                # 健康检查：尝试在设备上执行简单操作
                if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                    with torch.cuda.device(device_id):
                        # 创建测试张量
                        test_tensor = torch.randn(100, 100, device=device_id)
                        result = test_tensor @ test_tensor.T
                        
                        # 检查结果是否正常
                        if torch.isnan(result).any() or torch.isinf(result).any():
                            failed_devices.append(device_id)
                            self.logger.warning(f"设备{device_id}计算结果异常")
                
            except Exception as e:
                failed_devices.append(device_id)
                self.logger.error(f"设备{device_id}故障检测失败: {e}")
                
                # 记录故障
                failure_info = {
                    'device_id': device_id,
                    'timestamp': time.time(),
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                self.failure_history.append(failure_info)
        
        if failed_devices:
            self.logger.warning(f"检测到故障设备: {failed_devices}")
        
        return failed_devices
    
    def handle_device_failure(self, 
                            failed_devices: List[int],
                            available_devices: List[int]) -> Dict:
        """处理设备故障"""
        
        recovery_plan = {
            'strategy': self.backup_strategy,
            'device_mapping': {},
            'recovery_actions': []
        }
        
        if self.backup_strategy == "redundant":
            # 冗余策略：使用备用设备替换故障设备
            for failed_device in failed_devices:
                if available_devices:
                    backup_device = available_devices.pop(0)
                    recovery_plan['device_mapping'][failed_device] = backup_device
                    recovery_plan['recovery_actions'].append(
                        f"将设备{failed_device}的任务迁移到设备{backup_device}"
                    )
                else:
                    recovery_plan['recovery_actions'].append(
                        f"无可用备用设备替换故障设备{failed_device}"
                    )
        
        elif self.backup_strategy == "checkpoint_restart":
            # 检查点重启策略：从最近的检查点恢复
            if self.checkpoint_history:
                latest_checkpoint = self.checkpoint_history[-1]
                recovery_plan['recovery_actions'].append(
                    f"从检查点{latest_checkpoint['path']}恢复训练"
                )
            else:
                recovery_plan['recovery_actions'].append(
                    "没有可用检查点，需要重新开始训练"
                )
        
        elif self.backup_strategy == "graceful_degradation":
            # 优雅降级：调整批大小或模型并行度
            recovery_plan['recovery_actions'].append(
                "调整训练配置以适应减少的设备数量"
            )
        
        self.logger.info(f"故障恢复计划: {recovery_plan}")
        
        return recovery_plan
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        
        if len(self.checkpoint_history) > self.max_checkpoints:
            # 删除最旧的检查点
            old_checkpoints = self.checkpoint_history[:-self.max_checkpoints]
            
            for checkpoint_info in old_checkpoints:
                try:
                    if os.path.exists(checkpoint_info['path']):
                        os.remove(checkpoint_info['path'])
                        self.logger.info(f"已删除旧检查点: {checkpoint_info['path']}")
                except Exception as e:
                    self.logger.error(f"删除检查点失败: {e}")
            
            # 更新历史记录
            self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]
    
    def analyze_failure_patterns(self) -> Dict:
        """分析故障模式"""
        
        if not self.failure_history:
            return {}
        
        # 故障频率分析
        device_failure_counts = Counter([f['device_id'] for f in self.failure_history])
        error_type_counts = Counter([f['error_type'] for f in self.failure_history])
        
        # 时间分析
        timestamps = [f['timestamp'] for f in self.failure_history]
        if len(timestamps) > 1:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
        else:
            avg_interval = 0
            std_interval = 0
        
        return {
            'total_failures': len(self.failure_history),
            'device_failure_counts': dict(device_failure_counts),
            'error_type_counts': dict(error_type_counts),
            'failure_rate': len(self.failure_history) / (time.time() - timestamps[0]) if timestamps else 0,
            'average_failure_interval': avg_interval,
            'failure_interval_std': std_interval,
            'most_unreliable_device': device_failure_counts.most_common(1)[0][0] if device_failure_counts else None,
            'most_common_error': error_type_counts.most_common(1)[0][0] if error_type_counts else None
        }

def create_distributed_training_system(config: TrainingConfig):
    """创建分布式训练系统"""
    
    # 初始化管理器
    training_manager = DistributedTrainingManager(config)
    resource_manager = ResourceManager()
    comm_optimizer = CommunicationOptimizer(config.world_size)
    fault_manager = FaultToleranceManager(
        checkpoint_interval=config.checkpoint_interval,
        max_checkpoints=config.max_checkpoints
    )
    
    return {
        'training_manager': training_manager,
        'resource_manager': resource_manager,
        'communication_optimizer': comm_optimizer,
        'fault_tolerance_manager': fault_manager
    }

# 演示完整的分布式训练基础设施
def demonstrate_distributed_infrastructure():
    """演示分布式训练基础设施"""
    
    print("=== MiniGPT分布式训练基础设施演示 ===\n")
    
    # 创建训练配置
    config = TrainingConfig(
        model_size="small",
        batch_size=64,
        world_size=4,
        parallelism_type=ParallelismType.DATA_PARALLEL,
        mixed_precision=True,
        checkpoint_interval=100
    )
    
    # 创建分布式训练系统
    system = create_distributed_training_system(config)
    
    # 1. 初始化分布式环境
    print("1. 初始化分布式训练环境")
    system['training_manager'].initialize_distributed()
    
    # 2. 资源监控和分配
    print("\n2. 资源监控和管理")
    
    # 监控资源使用
    resource_usage = system['resource_manager'].monitor_resource_usage()
    print(f"当前CPU使用率: {resource_usage['cpu_usage']:.1f}%")
    print(f"当前内存使用率: {resource_usage['memory_usage']:.1f}%")
    
    # 模拟任务分配
    tasks = [
        {'memory': 2048, 'utilization': 20.0},  # 2GB内存需求
        {'memory': 4096, 'utilization': 35.0},  # 4GB内存需求
        {'memory': 1024, 'utilization': 15.0},  # 1GB内存需求
        {'memory': 3072, 'utilization': 25.0},  # 3GB内存需求
    ]
    
    allocation = system['resource_manager'].allocate_resources(tasks, strategy="best_fit")
    print(f"成功分配任务数: {len(allocation['allocated_tasks'])}")
    print(f"资源利用率: {allocation['resource_utilization']['memory_utilization']:.3f}")
    
    # 3. 通信优化分析
    print("\n3. 通信优化分析")
    
    # 模拟梯度张量
    gradients = {
        'layer1.weight': torch.randn(512, 256),
        'layer1.bias': torch.randn(512),
        'layer2.weight': torch.randn(256, 128),
        'layer2.bias': torch.randn(256)
    }
    
    # 估算通信时间
    total_size = sum(grad.numel() * grad.element_size() for grad in gradients.values())
    estimated_time = system['communication_optimizer'].estimate_allreduce_time(total_size, 'ring')
    print(f"估算Ring All-Reduce时间: {estimated_time:.4f}s")
    
    # 4. 容错机制测试
    print("\n4. 容错和恢复机制")
    
    # 模拟故障检测
    devices = list(range(config.world_size))
    failed_devices = system['fault_tolerance_manager'].detect_failure(devices)
    
    if failed_devices:
        recovery_plan = system['fault_tolerance_manager'].handle_device_failure(
            failed_devices, [4, 5, 6]  # 可用的备用设备
        )
        print(f"故障恢复计划: {recovery_plan['strategy']}")
    else:
        print("所有设备运行正常")
    
    # 5. 性能分析
    print("\n5. 系统性能分析")
    
    # 流水线效率分析
    pipeline_trainer = PipelineParallelTrainer(config, num_microbatches=8)
    stage_times = [0.1, 0.12, 0.11, 0.13]  # 各阶段时间
    efficiency = pipeline_trainer.analyze_pipeline_efficiency(
        num_stages=4, num_microbatches=8, stage_times=stage_times
    )
    
    print(f"流水线加速比: {efficiency['speedup']:.2f}")
    print(f"流水线效率: {efficiency['efficiency']:.2f}")
    print(f"气泡时间比例: {efficiency['bubble_ratio']:.2f}")
    
    return {
        'config': config,
        'system': system,
        'resource_usage': resource_usage,
        'allocation_result': allocation,
        'pipeline_efficiency': efficiency
    }

# 运行演示
if __name__ == "__main__":
    results = demonstrate_distributed_infrastructure()
    
    print("\n=== 分布式训练基础设施评估完成 ===")
    print(f"系统配置总结:")
    print(f"- 并行策略: {results['config'].parallelism_type.value}")
    print(f"- 设备数量: {results['config'].world_size}")
    print(f"- 批大小: {results['config'].batch_size}")
    print(f"- 混合精度: {results['config'].mixed_precision}")
    print(f"- 流水线效率: {results['pipeline_efficiency']['efficiency']:.2f}")
```

## 理论总结

### 1.5 分布式训练的统一理论框架

**扩展性定律**：
理想情况下，使用 $N$ 个设备的加速比为：
$$S(N) = \frac{T_1}{T_N} = \frac{T_1}{T_1/N + T_{comm} + T_{sync}}$$

其中 $T_{comm}$ 是通信时间，$T_{sync}$ 是同步时间。

**Amdahl定律的推广**：
$$S(N) = \frac{1}{f + \frac{1-f}{N}}$$

其中 $f$ 是不可并行部分的比例。

**通信复杂度理论**：
对于参数量为 $P$ 的模型，不同并行策略的通信复杂度：
- 数据并行：$O(P)$
- 模型并行：$O(\sqrt{P})$
- 流水线并行：$O(1)$

## 应用指导

### 实践建议

1. **并行策略选择**：
   - 小模型：优先数据并行
   - 大模型：混合并行策略
   - 超大模型：3D并行（数据+模型+流水线）

2. **通信优化**：
   - 使用高速互联（InfiniBand、NVLink）
   - 实施梯度压缩和量化
   - 重叠计算与通信

3. **容错设计**：
   - 定期保存检查点
   - 实现快速故障检测
   - 设计优雅降级机制

训练基础设施是大规模语言模型成功的关键支撑，需要在性能、可靠性、可扩展性之间找到最优平衡。

## 扩展阅读

- 《Distributed Deep Learning: A Survey》- 分布式深度学习综述
- 《Efficient Large-Scale Language Model Training》- 大规模语言模型训练技术
- 《PaLM: Scaling Language Modeling with Pathways》- Google PaLM训练经验
- 《GPipe: Efficient Training of Giant Neural Networks》- 流水线并行技术

---

*"基础设施决定上层建筑。在语言模型训练中，可靠高效的基础设施是实现技术突破的重要保障。"* 🏗️