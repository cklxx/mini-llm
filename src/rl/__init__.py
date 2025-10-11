"""
强化学习模块
实现RLHF (Reinforcement Learning from Human Feedback) 相关功能
"""

from . import ppo, reward_model
from .rlhf_pipeline import (
    RLHFConfig,
    RLHFPipeline,
    create_rlhf_pipeline,
    get_default_config,
)

__all__ = [
    'ppo',
    'reward_model',
    'RLHFPipeline',
    'RLHFConfig',
    'create_rlhf_pipeline',
    'get_default_config',
]
