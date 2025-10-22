"""
训练配置文件 - NVIDIA GPU 优化版本
支持 PyTorch 2.4 和 NVIDIA GPU 自动检测优化
"""
import os

import torch


def _parse_bool_env(key: str, default: bool) -> bool:
    """Parse boolean environment variables safely."""

    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_cpu_reserve(default: int = 2) -> int:
    """Parse the number of CPU cores to reserve for the system.

    The user can override the default via the ``MINIGPT_CPU_RESERVE`` environment
    variable. Invalid overrides fall back to ``default``.
    """

    reserve_env = os.environ.get("MINIGPT_CPU_RESERVE")
    if reserve_env is None:
        return max(0, default)

    try:
        return max(0, int(reserve_env))
    except (TypeError, ValueError):
        return max(0, default)


def _get_available_cpu_count(reserve_default: int = 2) -> tuple[int, int]:
    """Return a tuple of (total_cpus, available_cpus) respecting reservations."""

    total = os.cpu_count() or 1
    reserve = _parse_cpu_reserve(reserve_default)
    available = max(1, total - reserve)
    return total, available


def get_gpu_info():
    """获取 GPU 详细信息"""
    if not torch.cuda.is_available():
        return None

    gpu_info = {
        'count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'devices': []
    }

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            'id': i,
            'name': props.name,
            'memory_total': props.total_memory / 1024**3,  # GB
            'memory_allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
            'memory_reserved': torch.cuda.memory_reserved(i) / 1024**3,  # GB
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processor_count': props.multi_processor_count
        }
        gpu_info['devices'].append(device_info)

    return gpu_info


def setup_cuda_optimizations():
    """设置 CUDA 优化"""
    if torch.cuda.is_available():
        # 启用 TensorFloat-32 (适用于 Ampere 架构)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 启用 CuDNN benchmark
        torch.backends.cudnn.benchmark = True

        # 设置 CUDA 内存分配策略
        torch.cuda.empty_cache()

        print("已启用 CUDA 优化:")
        print(f"  - TensorFloat-32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        return True
    return False


def get_device():
    """自动检测最佳设备并优化配置"""
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        setup_cuda_optimizations()

        primary_gpu = gpu_info['devices'][0]
        print(f"检测到 NVIDIA GPU: {primary_gpu['name']}")
        print(f"显存总量: {primary_gpu['memory_total']:.1f} GB")
        print(f"计算能力: {primary_gpu['compute_capability']}")
        print(f"多处理器数量: {primary_gpu['multi_processor_count']}")

        return "cuda", gpu_info
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("检测到 Apple Silicon GPU (MPS)")
        return "mps", None
    else:
        print("使用 CPU")
        return "cpu", None


class BaseConfig:
    """基础配置类 - NVIDIA GPU 优化"""
    def __init__(self):
        # 基础路径
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.manifest_dir = os.path.join(self.project_root, "configs", "data")
        self.data_dir = os.path.join(self.project_root, "data")
        self.data_search_dirs = [
            self.data_dir,
            os.path.join(self.data_dir, "final")
        ]
        self.checkpoint_dir = os.path.join(self.project_root, "checkpoints")
        self.log_dir = os.path.join(self.project_root, "logs")

        # Tokenizer configuration (MiniMind alignment)
        self.tokenizer_type = "huggingface"
        self.tokenizer_json_path = os.path.join(
            self.project_root, "tokenizers", "minimind", "tokenizer.json"
        )
        self.tokenizer_config_path = os.path.join(
            self.project_root, "tokenizers", "minimind", "tokenizer_config.json"
        )

        # TensorBoard 配置
        # 检测是否在云GPU环境（OpenBayes）
        cloud_tb_dir = "/openbayes/home/tf_dir"
        if os.path.exists("/openbayes/home") and os.access("/openbayes/home", os.W_OK):
            # 云GPU环境：使用固定路径以便平台自动检测
            self.tensorboard_dir = cloud_tb_dir
            os.makedirs(cloud_tb_dir, exist_ok=True)
            print(f"🌐 检测到云GPU环境，TensorBoard日志: {cloud_tb_dir}")
        else:
            # 本地环境：使用项目内的runs目录
            self.tensorboard_dir = os.path.join(self.project_root, "runs")

        self.enable_tensorboard = True  # 默认启用TensorBoard
        self.tensorboard_flush_secs = 30  # 每30秒刷新一次

        # 设备配置
        self.device, self.gpu_info = get_device()
        self.cpu_total_cores, self.cpu_available_cores = _get_available_cpu_count()
        self.cpu_reserved_cores = max(0, self.cpu_total_cores - self.cpu_available_cores)
        if self.cpu_reserved_cores > 0:
            print(
                "🧮 CPU 核心: 总计"
                f" {self.cpu_total_cores}, 预留 {self.cpu_reserved_cores}, 可用于数据处理 {self.cpu_available_cores}"
            )
        else:
            print(f"🧮 CPU 核心: 总计 {self.cpu_total_cores}, 全部可用于数据处理")

        # 默认启用可扩展显存分段，缓解CUDA内存碎片问题
        if self.device == "cuda":
            alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            if alloc_conf is None:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                print("🧠 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 以减少显存碎片")

        # 数据集路径
        self.pretrain_data_path = os.path.join(self.data_dir, "pretrain_hq.jsonl")
        self.sft_data_path = os.path.join(self.data_dir, "sft_mini_512.jsonl")
        self.dpo_data_path = os.path.join(self.data_dir, "dpo.jsonl")

        # 身份认同和能力增强数据集路径
        self.alex_identity_data_path = os.path.join(self.data_dir, "alex_identity.jsonl")
        self.ultra_think_data_path = os.path.join(self.data_dir, "ultra_think.jsonl")
        self.lora_identity_data_path = os.path.join(self.data_dir, "lora_identity.jsonl")

        # 训练可重复性与验证集设置
        self.random_seed = int(os.environ.get("MINIGPT_TRAIN_SEED", 42))
        self.validation_split = float(os.environ.get("MINIGPT_VAL_SPLIT", 0.05))
        self.validation_min_samples = 80
        self.early_stopping_patience = int(os.environ.get("MINIGPT_EARLY_STOP_PATIENCE", 4))
        self.early_stopping_delta = float(os.environ.get("MINIGPT_EARLY_STOP_DELTA", 0.0))
        self.label_smoothing = float(os.environ.get("MINIGPT_LABEL_SMOOTHING", 0.05))
        self.dpo_beta = float(os.environ.get("MINIGPT_DPO_BETA", 0.1))

        # 全局数据采样控制
        global_ratio_env = os.environ.get("MINIGPT_GLOBAL_SAMPLE_RATIO")
        try:
            self.dataset_global_sample_ratio = (
                max(0.0, float(global_ratio_env)) if global_ratio_env is not None else 1.0
            )
        except ValueError:
            self.dataset_global_sample_ratio = 1.0

        def _parse_sample_ratio(env_key: str, default: float) -> float:
            value = os.environ.get(env_key)
            if value is None:
                return default
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return default

        def _parse_max_samples(env_key: str, default: int | None) -> int | None:
            value = os.environ.get(env_key)
            if value is None:
                return default
            value = str(value).strip()
            if value == "":
                return default
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                return default

        # 数据采样与验证划分策略（按文件名匹配）
        self.dataset_sampling = {
            "default": {
                "sample_ratio": 1.0,
                "max_samples": None,
                "val_split": self.validation_split
            },
            "sft_mini_512.cleaned.jsonl": {
                "sample_ratio": float(os.environ.get("MINIGPT_SFT_MAIN_RATIO", 0.2)),
                "max_samples": int(os.environ.get("MINIGPT_SFT_MAIN_MAX", 150000)),
                "val_split": self.validation_split
            },
            "sft_mini_512.jsonl": {
                "sample_ratio": float(os.environ.get("MINIGPT_SFT_MAIN_RATIO", 0.2)),
                "max_samples": int(os.environ.get("MINIGPT_SFT_MAIN_MAX", 150000)),
                "val_split": self.validation_split
            },
            "alex_identity.jsonl": {
                "sample_ratio": 0.25,
                "max_samples": 3000,
                "val_split": 0.1
            },
            "minigpt_identity.jsonl": {
                "sample_ratio": 0.25,
                "max_samples": 3000,
                "val_split": 0.1
            },
            "ultra_think.jsonl": {
                "sample_ratio": 0.5,
                "max_samples": 6000,
                "val_split": 0.05
            },
            "dpo.jsonl": {
                "sample_ratio": _parse_sample_ratio("MINIGPT_DPO_RATIO", 1.0),
                "max_samples": _parse_max_samples("MINIGPT_DPO_MAX", None),
                "val_split": 0.0
            },
            "wiki_zh_full.simdedup.jsonl": {
                "sample_ratio": _parse_sample_ratio("MINIGPT_PRETRAIN_WIKI_RATIO", 1.0),
                "max_samples": _parse_max_samples("MINIGPT_PRETRAIN_WIKI_MAX", None),
                "val_split": self.validation_split
            },
            "chinacorpus_full.simdedup.jsonl": {
                "sample_ratio": _parse_sample_ratio("MINIGPT_PRETRAIN_CHINA_RATIO", 1.0),
                "max_samples": _parse_max_samples("MINIGPT_PRETRAIN_CHINA_MAX", None),
                "val_split": self.validation_split
            },
            "pretrain_hq.cleaned.jsonl": {
                "sample_ratio": _parse_sample_ratio("MINIGPT_PRETRAIN_HQ_RATIO", 1.0),
                "max_samples": _parse_max_samples("MINIGPT_PRETRAIN_HQ_MAX", None),
                "val_split": self.validation_split
            },
        }

        # 对话角色标记
        self.role_tokens = {
            "system": "<|system|>",
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "turn_separator": "<|endofturn|>"
        }

        # 对话数据增强配置
        self.conversation_augmentation = {
            "turn_truncate_prob": float(os.environ.get("MINIGPT_TURN_TRUNCATE_PROB", 0.1)),
            "max_turn_truncate": int(os.environ.get("MINIGPT_MAX_TURN_TRUNCATE", 1))
        }

        # 性能与日志控制
        self.enable_performance_diagnostics = (
            os.environ.get("MINIGPT_PERF_DIAGNOSTICS", "1") == "1"
        )
        self.step_log_interval = max(
            1, int(os.environ.get("MINIGPT_STEP_LOG_INTERVAL", 10))
        )
        self.performance_window_updates = max(
            1,
            int(
                os.environ.get(
                    "MINIGPT_PERF_WINDOW_UPDATES", self.step_log_interval
                )
            ),
        )

        # 数据加载性能优化
        self.use_high_performance_data_loading = (
            os.environ.get("MINIGPT_FAST_DATA_LOADING", "1") == "1"
        )
        cache_root = os.environ.get(
            "MINIGPT_DATA_CACHE_DIR",
            os.path.join(self.project_root, "cache", "data_loader"),
        )
        self.data_cache_dir = cache_root
        os.makedirs(self.data_cache_dir, exist_ok=True)
        self.data_cache_enabled = os.environ.get("MINIGPT_DATA_CACHE", "1") == "1"
        self.data_cache_force_rebuild = (
            os.environ.get("MINIGPT_DATA_CACHE_REBUILD", "0") == "1"
        )
        self.data_streaming_enabled = (
            os.environ.get("MINIGPT_DATA_STREAMING", "0") == "1"
        )
        default_chunk = 20000
        default_buffer = 100000
        self.data_chunk_size = int(
            os.environ.get("MINIGPT_DATA_CHUNK_SIZE", default_chunk)
        )
        self.data_buffer_size = int(
            os.environ.get("MINIGPT_DATA_BUFFER_SIZE", default_buffer)
        )
        default_parallel = max(1, min(32, self.cpu_available_cores))
        self.data_max_parallel_workers = int(
            os.environ.get("MINIGPT_DATA_MAX_WORKERS", default_parallel)
        )

        # 训练内存监控与优化
        self.memory_monitor_enabled = os.environ.get("MINIGPT_MEMORY_MONITOR", "1") == "1"
        self.memory_pressure_threshold = float(os.environ.get("MINIGPT_MEMORY_THRESHOLD", 0.92))
        self.memory_cleanup_interval = int(os.environ.get("MINIGPT_MEMORY_CLEANUP_INTERVAL", 200))
        self.memory_log_interval = int(os.environ.get("MINIGPT_MEMORY_LOG_INTERVAL", 200))

        # 推理与评测通用配置（参考 MiniMind 推理脚本）
        inference_dtype_env = (os.environ.get("MINIGPT_INFERENCE_DTYPE") or "auto").strip().lower()
        if inference_dtype_env in {"bf16", "bfloat16"} and torch.cuda.is_available():
            self.inference_autocast_dtype = torch.bfloat16
        elif inference_dtype_env in {"fp16", "float16", "half"} and torch.cuda.is_available():
            self.inference_autocast_dtype = torch.float16
        else:
            self.inference_autocast_dtype = None

        self.inference_temperature = float(os.environ.get("MINIGPT_INFERENCE_TEMPERATURE", 0.85))
        self.inference_top_p = float(os.environ.get("MINIGPT_INFERENCE_TOP_P", 0.85))
        self.inference_top_k = int(os.environ.get("MINIGPT_INFERENCE_TOP_K", 0))
        self.inference_repetition_penalty = float(
            os.environ.get("MINIGPT_INFERENCE_REPETITION_PENALTY", 1.05)
        )
        default_new_tokens = int(os.environ.get("MINIGPT_INFERENCE_MAX_NEW_TOKENS", 0))
        if default_new_tokens <= 0:
            default_new_tokens = getattr(self, "max_seq_len", 512)
        self.inference_max_new_tokens = max(8, default_new_tokens)
        self.inference_history_turns = max(
            0, int(os.environ.get("MINIGPT_INFERENCE_HISTORY_TURNS", 0))
        )
        self.inference_use_chat_template = _parse_bool_env(
            "MINIGPT_INFERENCE_USE_CHAT_TEMPLATE", True
        )
        self.inference_enable_rope_scaling = _parse_bool_env(
            "MINIGPT_INFERENCE_ENABLE_ROPE_SCALING", False
        )

        # 训练后回归评估配置
        self.regression_eval_enabled = os.environ.get("MINIGPT_REGRESSION_EVAL", "1") == "1"
        self.regression_eval_prompts = os.path.join(self.data_dir, "eval", "regression_prompts.jsonl")
        self.regression_eval_interval = int(os.environ.get("MINIGPT_REGRESSION_INTERVAL", 500))
        default_regression_new = int(os.environ.get("MINIGPT_REGRESSION_MAX_NEW", 0))
        if default_regression_new <= 0:
            default_regression_new = getattr(self, "inference_max_new_tokens", 96)
        self.regression_eval_max_new_tokens = max(4, default_regression_new)
        self.regression_eval_temperature = float(
            os.environ.get("MINIGPT_REGRESSION_TEMPERATURE", self.inference_temperature)
        )
        self.regression_eval_top_p = float(
            os.environ.get("MINIGPT_REGRESSION_TOP_P", self.inference_top_p)
        )

        # 行业评测基准配置（默认使用 WikiText-2、LAMBADA、HellaSwag、ARC 等行业通用任务）
        self.benchmark_eval_enabled = os.environ.get("MINIGPT_BENCHMARK_ENABLED", "1") == "1"

        tasks_env = os.environ.get("MINIGPT_BENCHMARK_TASKS")
        default_tasks = [
            "wikitext2",
            "lambada_openai",
            "hellaswag",
            "arc_challenge",
            "winogrande",
            "piqa",
            "boolq",
        ]
        if tasks_env:
            parsed_tasks = [task.strip() for task in tasks_env.split(",") if task.strip()]
            self.benchmark_eval_tasks = parsed_tasks or default_tasks
        else:
            self.benchmark_eval_tasks = default_tasks

        def _normalize_env(key: str):
            value = os.environ.get(key)
            if value is None:
                return None
            value = value.strip()
            return value or None

        self.benchmark_eval_overrides = {
            "dataset_name": _normalize_env("MINIGPT_BENCHMARK_DATASET"),
            "dataset_config": _normalize_env("MINIGPT_BENCHMARK_DATASET_CONFIG"),
            "split": _normalize_env("MINIGPT_BENCHMARK_SPLIT"),
            "text_column": _normalize_env("MINIGPT_BENCHMARK_TEXT_COLUMN"),
        }

        self.benchmark_eval_frequency = int(
            os.environ.get("MINIGPT_BENCHMARK_FREQUENCY", 500)
        )
        self.benchmark_eval_max_samples = int(
            os.environ.get("MINIGPT_BENCHMARK_MAX_SAMPLES", 256)
        )
        self.benchmark_eval_batch_size = int(
            os.environ.get("MINIGPT_BENCHMARK_BATCH_SIZE", 4)
        )
        benchmark_max_len = int(os.environ.get("MINIGPT_BENCHMARK_MAX_LENGTH", 0))
        self.benchmark_eval_max_length = benchmark_max_len if benchmark_max_len > 0 else None

        self.benchmark_eval_auto_download = (
            os.environ.get("MINIGPT_BENCHMARK_AUTO_DOWNLOAD", "1") == "1"
        )
        cache_dir_override = _normalize_env("MINIGPT_BENCHMARK_CACHE_DIR")
        if cache_dir_override:
            self.benchmark_eval_cache_dir = cache_dir_override
        else:
            self.benchmark_eval_cache_dir = os.path.join(
                self.data_dir, "eval", "benchmarks"
            )
        if self.benchmark_eval_cache_dir:
            os.makedirs(self.benchmark_eval_cache_dir, exist_ok=True)

        # NVIDIA GPU 优化设置
        self.mixed_precision = self.device == "cuda"
        self.compile_model = self.device == "cuda"  # PyTorch 2.4 编译优化
        self.gradient_checkpointing = True
        self.flash_attention = self.device == "cuda"

        # 数据加载优化 - 针对RTX 4090和多核CPU优化
        available_workers = self.cpu_available_cores
        if self.device == "cuda":
            # 根据可用CPU核心动态调整，默认预留系统核心
            self.num_workers = max(1, min(16, available_workers))
            self.prefetch_factor = 8 if self.num_workers > 1 else 2
        else:
            self.num_workers = max(1, min(4, available_workers))
            self.prefetch_factor = 4 if self.num_workers > 1 else 2
        self.pin_memory = self.device == "cuda"
        self.persistent_workers = True if self.num_workers > 0 else False

        # 创建必要目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)


class TinyConfig(BaseConfig):
    """超小型模型配置 (~1M参数) - 快速实验"""
    def __init__(self):
        super().__init__()

        # 模型标识
        self.model_size = "tiny"

        # 模型参数
        self.vocab_size = 6400
        self.d_model = 128
        self.n_heads = 4
        self.n_layers = 8
        self.d_ff = 384
        self.max_seq_len = 512
        self.dropout = 0.1

        # 训练参数
        self.batch_size = 32
        self.gradient_accumulation_steps = 2
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.warmup_steps = 500
        self.max_steps = 10000
        self.eval_steps = 500
        self.save_steps = 1000

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 512
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class SmallConfig(BaseConfig):
    """小型模型配置 (~25M参数) - 瘦长架构优化内存"""
    def __init__(self):
        super().__init__()

        # 模型标识
        self.model_size = "small"

        # 模型参数 - 瘦长架构：更窄但更深，降低内存峰值
        self.vocab_size = 6400
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 8
        self.d_ff = 2048
        self.max_seq_len = 512
        self.dropout = 0.0

        # 训练参数 - 优化内存使用和GPU利用率
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            gpu_name = self.gpu_info['devices'][0]['name'].lower() if self.gpu_info else ""

            if gpu_memory >= 22 or "4090" in gpu_name or "ada" in gpu_name:
                # RTX 4090/A6000: 降低单次batch显存峰值
                self.batch_size = 32
                self.gradient_accumulation_steps = 6  # 有效batch = 32*6 = 192
            elif gpu_memory >= 16:
                self.batch_size = 48
                self.gradient_accumulation_steps = 5
            elif gpu_memory >= 12:
                self.batch_size = 32
                self.gradient_accumulation_steps = 8
            else:
                self.batch_size = 32
                self.gradient_accumulation_steps = 8
        else:
            self.batch_size = 16
            self.gradient_accumulation_steps = 16

        self.learning_rate = 5e-4
        self.weight_decay = 0.01
        self.warmup_steps = 2000
        self.max_steps = 60000
        self.eval_steps = 2000
        self.save_steps = 4000

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 1024
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class Small30MConfig(BaseConfig):
    """30M参数小型模型配置"""
    def __init__(self):
        super().__init__()

        # 模型标识
        self.model_size = "small_30m"

        # 模型参数
        self.vocab_size = 6400
        self.d_model = 384
        self.n_heads = 12
        self.n_layers = 13
        self.d_ff = 1408
        self.max_seq_len = 2048
        self.dropout = 0.1

        # 训练参数
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            if gpu_memory >= 24:
                self.batch_size = 24
                self.gradient_accumulation_steps = 6
            elif gpu_memory >= 12:
                self.batch_size = 12
                self.gradient_accumulation_steps = 12
            else:
                self.batch_size = 6
                self.gradient_accumulation_steps = 24
        else:
            self.batch_size = 6
            self.gradient_accumulation_steps = 24

        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.warmup_steps = 1500
        self.max_steps = 60000
        self.eval_steps = 1500
        self.save_steps = 3000

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 1024
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class MediumConfig(BaseConfig):
    """中等模型配置 (~80M参数) - GPU优化 + Flash Attention"""
    def __init__(self):
        super().__init__()

        # 模型标识
        self.model_size = "medium"

        # 模型参数 (与 src/model/config.py 中的 medium 预设保持一致)
        self.vocab_size = 6400
        self.d_model = 384       # 瘦长架构：降低宽度
        self.n_heads = 12
        self.n_layers = 20       # 提升深度以补足表达能力
        self.d_ff = 1536         # 4 × hidden 的 FFN 设计
        self.max_seq_len = 2048
        self.dropout = 0.1

        # GPU优化训练参数 - 针对A6000优化 (保守配置以避免OOM)
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            if gpu_memory >= 40:  # A6000等高端卡 (48GB)
                # 保守配置：降低batch_size，增加梯度累积
                # 这样可以减少单次前向传播的内存峰值
                self.batch_size = 16  # 降低到16 (之前32)
                self.gradient_accumulation_steps = 8  # 增加到8，有效批量 = 16 * 8 = 128
            elif gpu_memory >= 24:  # RTX 3090/4090
                self.batch_size = 8
                self.gradient_accumulation_steps = 16  # 有效批量 = 128
            elif gpu_memory >= 12:  # RTX 3060Ti/4060Ti
                self.batch_size = 4
                self.gradient_accumulation_steps = 32
            else:
                self.batch_size = 2
                self.gradient_accumulation_steps = 64
        else:
            self.batch_size = 2
            self.gradient_accumulation_steps = 64

        # 采用与 MiniMind 相近的较高学习率，加速瘦长模型收敛
        self.learning_rate = 5e-4
        self.weight_decay = 0.01
        # 使用 3% warmup 比例，平衡训练稳定性和收敛速度
        self.warmup_steps = 3000
        self.max_steps = 100000
        self.eval_steps = 2000
        self.save_steps = 5000

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 1024
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class FoundationConfig(BaseConfig):
    """基础模型配置 (~200M参数) - 中型规模训练"""
    def __init__(self):
        super().__init__()

        # 模型标识
        self.model_size = "foundation"

        # 模型参数
        self.vocab_size = 6400
        self.d_model = 768
        self.n_heads = 16
        self.n_layers = 24
        self.d_ff = 2688
        self.max_seq_len = 4096
        self.dropout = 0.1

        # 训练参数 - GPU优化
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            if gpu_memory >= 40:  # A6000等高端卡
                self.batch_size = 12
                self.gradient_accumulation_steps = 12
            elif gpu_memory >= 24:  # RTX 3090/4090
                self.batch_size = 6
                self.gradient_accumulation_steps = 24
            elif gpu_memory >= 12:
                self.batch_size = 3
                self.gradient_accumulation_steps = 48
            else:
                self.batch_size = 2
                self.gradient_accumulation_steps = 64
        else:
            self.batch_size = 2
            self.gradient_accumulation_steps = 64

        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.warmup_steps = 3000
        self.max_steps = 150000
        self.eval_steps = 3000
        self.save_steps = 8000

        # 强制启用内存优化
        self.gradient_checkpointing = True

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 2048
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class LargeConfig(BaseConfig):
    """大模型配置 (~350M参数) - 高端 GPU 优化"""
    def __init__(self):
        super().__init__()

        # 模型标识
        self.model_size = "large"

        # 模型参数 (更新为与src/model/config.py一致)
        self.vocab_size = 6400
        self.d_model = 768
        self.n_heads = 24
        self.n_layers = 32
        self.d_ff = 3072
        self.max_seq_len = 4096
        self.dropout = 0.1

        # 只在高端 GPU 上运行
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            if gpu_memory >= 40:  # A6000等高端卡
                self.batch_size = 8
                self.gradient_accumulation_steps = 16
            elif gpu_memory >= 24:  # RTX 3090/4090
                self.batch_size = 4
                self.gradient_accumulation_steps = 32
            elif gpu_memory >= 16:  # RTX 4080
                self.batch_size = 2
                self.gradient_accumulation_steps = 64
            else:
                self.batch_size = 1
                self.gradient_accumulation_steps = 128
                print("警告: 显存不足，建议使用更小的模型配置")
        else:
            self.batch_size = 1
            self.gradient_accumulation_steps = 128
            print("警告: 大模型建议使用 CUDA GPU")

        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.warmup_steps = 4000
        self.max_steps = 200000
        self.eval_steps = 5000
        self.save_steps = 10000

        # 强制启用内存优化
        self.gradient_checkpointing = True

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 2048
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class MOEConfig(BaseConfig):
    """MOE (Mixture of Experts) 模型配置"""
    def __init__(self):
        super().__init__()

        # 模型标识
        self.model_size = "moe"

        # 模型参数
        self.vocab_size = 6400
        self.d_model = 384
        self.n_heads = 12
        self.n_layers = 12
        self.d_ff = 1536
        self.max_seq_len = 1024
        self.dropout = 0.1

        # MOE特有参数
        self.use_moe = True
        self.num_experts_per_tok = 2
        self.n_routed_experts = 4
        self.n_shared_experts = 1

        # 训练参数
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            if gpu_memory >= 24:
                self.batch_size = 16
                self.gradient_accumulation_steps = 8
            elif gpu_memory >= 12:
                self.batch_size = 8
                self.gradient_accumulation_steps = 16
            else:
                self.batch_size = 4
                self.gradient_accumulation_steps = 32
        else:
            self.batch_size = 4
            self.gradient_accumulation_steps = 32

        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.max_steps = 50000
        self.eval_steps = 1000
        self.save_steps = 2000

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 1024
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


def get_config(model_size="medium"):
    """获取指定大小的配置"""
    configs = {
        "tiny": TinyConfig,
        "small": SmallConfig,
        "small_30m": Small30MConfig,
        "medium": MediumConfig,
        "foundation": FoundationConfig,
        "large": LargeConfig,
        "moe": MOEConfig
    }

    if model_size not in configs:
        raise ValueError(f"不支持的模型大小: {model_size}. 支持的大小: {list(configs.keys())}")

    config = configs[model_size]()

    # 打印配置信息
    print(f"\n=== {model_size.upper()} 模型配置 ===")
    print(f"设备: {config.device}")
    if config.gpu_info:
        gpu = config.gpu_info['devices'][0]
        print(f"GPU: {gpu['name']} ({gpu['memory_total']:.1f} GB)")
    print(f"批量大小: {config.batch_size}")
    print(f"梯度累积: {config.gradient_accumulation_steps}")
    print(f"有效批量: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"混合精度: {config.mixed_precision}")
    print(f"模型编译: {config.compile_model}")
    print(f"Flash Attention: {config.flash_attention}")
    print(f"梯度检查点: {config.gradient_checkpointing}")

    return config


def get_medium_config():
    return get_config("medium")


def get_large_config():
    return get_config("large")


if __name__ == "__main__":
    # 测试所有配置
    configs = ["tiny", "small", "small_30m", "medium", "foundation", "large", "moe"]

    for config_name in configs:
        print(f"\n{'='*50}")
        config = get_config(config_name)
        print(f"配置测试完成: {config_name}")
