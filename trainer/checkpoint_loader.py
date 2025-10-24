"""Utility module for loading and managing training checkpoints."""
import os
import torch
from pathlib import Path
from typing import Tuple, Optional


class CheckpointLoader:
    """Utility for loading checkpoints from various sources including remote paths."""

    # Standard checkpoint naming patterns
    CHECKPOINT_PATTERNS = {
        'pretrain': 'pretrain_{hidden_size}{moe}.pth',
        'sft': 'full_sft_{hidden_size}{moe}.pth',
        'reasoning': 'reason_{hidden_size}{moe}.pth',
        'rlhf': 'rlhf_{hidden_size}{moe}.pth',
        'ppo_actor': 'ppo_actor_{hidden_size}{moe}.pth',
        'ppo_critic': 'ppo_critic_{hidden_size}{moe}.pth',
        'distillation': 'full_dist_{hidden_size}{moe}.pth',
    }

    @staticmethod
    def get_checkpoint_path(
        stage: str,
        hidden_size: int,
        use_moe: bool = False,
        source: str = 'local',
        local_dir: str = './out',
        remote_dir: str = '/openbayes/home/out',
    ) -> Tuple[str, bool]:
        """
        Get checkpoint path for given stage and configuration.

        Args:
            stage: Training stage ('pretrain', 'sft', 'reasoning', 'rlhf', 'ppo_actor')
            hidden_size: Model hidden size
            use_moe: Whether model uses MoE
            source: 'local' or 'remote' source
            local_dir: Local checkpoint directory
            remote_dir: Remote checkpoint directory (e.g., /openbayes/home/out)

        Returns:
            Tuple of (checkpoint_path, exists)
        """
        moe_suffix = '_moe' if use_moe else ''

        # Get filename from pattern or use default
        if stage in CheckpointLoader.CHECKPOINT_PATTERNS:
            filename = CheckpointLoader.CHECKPOINT_PATTERNS[stage].format(
                hidden_size=hidden_size,
                moe=moe_suffix
            )
        else:
            filename = f'{stage}_{hidden_size}{moe_suffix}.pth'

        # Determine directory
        if source == 'remote':
            ckp_dir = remote_dir
        else:
            ckp_dir = local_dir

        ckp_path = os.path.join(ckp_dir, filename)
        exists = os.path.exists(ckp_path)

        return ckp_path, exists

    @staticmethod
    def load_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        device: str = 'cuda:0',
        strict: bool = False,
        logger=None,
    ) -> bool:
        """
        Load checkpoint state_dict into model.

        Args:
            model: PyTorch model to load into
            checkpoint_path: Path to checkpoint file
            device: Device to map tensors to
            strict: Whether to require exact key matching
            logger: Optional logger function for messages

        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(checkpoint_path):
            if logger:
                logger(f'Checkpoint not found: {checkpoint_path}')
            return False

        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=strict)

            if logger:
                logger(f'Successfully loaded checkpoint from: {checkpoint_path}')
            return True
        except Exception as e:
            if logger:
                logger(f'Error loading checkpoint {checkpoint_path}: {str(e)}')
            raise

    @staticmethod
    def resolve_checkpoint_path(
        stage: Optional[str] = None,
        hidden_size: Optional[int] = None,
        use_moe: bool = False,
        explicit_path: Optional[str] = None,
        env_var: str = 'MINILLM_PRETRAINED_PATH',
        local_dir: str = './out',
        remote_dir: str = '/openbayes/home/out',
        logger=None,
    ) -> Optional[str]:
        """
        Resolve checkpoint path using multiple fallback strategies.

        Priority order:
        1. Explicit path argument
        2. Environment variable
        3. Computed path (stage + hidden_size)
        4. Return None if none found

        Args:
            stage: Training stage name
            hidden_size: Model hidden size
            use_moe: Whether model uses MoE
            explicit_path: Explicitly provided checkpoint path
            env_var: Environment variable name to check
            local_dir: Local checkpoint directory
            remote_dir: Remote checkpoint directory
            logger: Optional logger function

        Returns:
            Path to checkpoint if found, None otherwise
        """
        # Priority 1: Explicit path
        if explicit_path:
            if os.path.exists(explicit_path):
                if logger:
                    logger(f'Using explicit checkpoint path: {explicit_path}')
                return explicit_path
            elif logger:
                logger(f'Explicit checkpoint path not found: {explicit_path}')
            return None

        # Priority 2: Environment variable
        env_path = os.environ.get(env_var)
        if env_path:
            if os.path.exists(env_path):
                if logger:
                    logger(f'Using checkpoint from env var {env_var}: {env_path}')
                return env_path
            elif logger:
                logger(f'Checkpoint path from env var not found: {env_path}')

        # Priority 3: Compute path from stage and hidden_size
        if stage and hidden_size:
            # Try remote first, then local
            remote_path, remote_exists = CheckpointLoader.get_checkpoint_path(
                stage=stage,
                hidden_size=hidden_size,
                use_moe=use_moe,
                source='remote',
                remote_dir=remote_dir,
            )

            if remote_exists:
                if logger:
                    logger(f'Using remote checkpoint: {remote_path}')
                return remote_path

            local_path, local_exists = CheckpointLoader.get_checkpoint_path(
                stage=stage,
                hidden_size=hidden_size,
                use_moe=use_moe,
                source='local',
                local_dir=local_dir,
            )

            if local_exists:
                if logger:
                    logger(f'Using local checkpoint: {local_path}')
                return local_path
            elif logger:
                logger(f'No checkpoint found for stage={stage}, hidden_size={hidden_size}')

        return None
