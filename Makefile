# MiniGPT Training - Makefile
# 快速命令参考

.PHONY: help install dev-install test lint format clean all

# 默认目标
help:
	@echo "MiniGPT Training - 可用命令:"
	@echo ""
	@echo "  环境配置:"
	@echo "    make install        - 安装基础依赖"
	@echo "    make dev-install    - 安装开发环境（包含测试和代码质量工具）"
	@echo "    make gpu-install    - 安装GPU优化依赖（仅x86_64）"
	@echo ""
	@echo "  代码质量:"
	@echo "    make lint           - 运行代码检查（ruff + mypy）"
	@echo "    make format         - 格式化代码（black + ruff --fix）"
	@echo "    make format-check   - 检查代码格式（不修改）"
	@echo ""
	@echo "  测试:"
	@echo "    make test           - 运行所有测试"
	@echo "    make test-fast      - 运行快速测试（排除slow标记）"
	@echo "    make test-structure - 运行结构验证测试"
	@echo "    make test-arch      - 运行架构测试"
	@echo ""
	@echo "  Git和清理:"
	@echo "    make pre-commit     - 安装pre-commit hooks"
	@echo "    make clean          - 清理临时文件"
	@echo "    make clean-all      - 深度清理（包括缓存）"
	@echo ""
	@echo "  训练和推理:"
	@echo "    make train-sft      - 训练SFT模型（small配置，支持自动恢复）"
	@echo "    make train-pretrain - 预训练模型（small配置，支持自动恢复）"
	@echo "    make chat           - 启动交互式聊天"
	@echo ""
	@echo "  TensorBoard监控:"
	@echo "    make tensorboard        - 启动TensorBoard服务"
	@echo "    make tensorboard-stop   - 停止TensorBoard服务"
	@echo "    make tensorboard-status - 查看TensorBoard状态"
	@echo "    make tensorboard-list   - 列出所有训练日志"
	@echo "    make tensorboard-clean  - 清理30天前的旧日志"
	@echo ""
	@echo "  模型评估:"
	@echo "    make eval-quick       - 快速评估（仅自我认知测试）"
	@echo "    make eval-full        - 完整评估（所有测试类别）"
	@echo "    make eval-categories  - 列出所有评估类别"

# 环境配置
install:
	@echo "📦 安装基础依赖..."
	uv sync

dev-install:
	@echo "📦 安装开发环境..."
	uv sync --all-extras

gpu-install:
	@echo "🚀 安装GPU优化依赖（仅x86_64）..."
	uv sync --extra gpu

# 代码质量
lint:
	@echo "🔍 运行代码检查..."
	uv run ruff check src/ scripts/
	@echo "🔍 运行类型检查..."
	uv run mypy src/

format:
	@echo "✨ 格式化代码..."
	uv run black src/ scripts/
	uv run ruff check --fix src/ scripts/

format-check:
	@echo "🔍 检查代码格式..."
	uv run black --check src/ scripts/
	uv run ruff check src/ scripts/

# 测试
test:
	@echo "🧪 运行所有测试..."
	uv run python scripts/test_runner.py

test-fast:
	@echo "⚡ 运行快速测试..."
	uv run pytest scripts/tests/ -v -m "not slow"

test-structure:
	@echo "🏗️ 运行结构验证测试..."
	uv run python scripts/tests/test_code_structure.py

test-arch:
	@echo "🧠 运行架构测试..."
	uv run python scripts/tests/test_architecture.py

# Git和pre-commit
pre-commit:
	@echo "🪝 安装pre-commit hooks..."
	uv run pre-commit install
	@echo "✅ Pre-commit hooks已安装"

pre-commit-run:
	@echo "🔍 运行pre-commit检查..."
	uv run pre-commit run --all-files

# 清理
clean:
	@echo "🧹 清理临时文件..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type f -name "._*" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

clean-all: clean
	@echo "🧹 深度清理..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .venv/ .uv/
	rm -rf checkpoints/ logs/ wandb/

# 训练和推理快捷命令
train-sft:
	@echo "🏋️ 训练SFT模型（small配置，自动恢复）..."
	uv run python scripts/train.py --mode sft --config small --retrain-tokenizer --auto-resume

train-pretrain:
	@echo "🏋️ 预训练模型（small配置，自动恢复）..."
	uv run python scripts/train.py --mode pretrain --config small --auto-resume

train-dpo:
	@echo "🏋️ DPO训练（需要先完成SFT，自动恢复）..."
	uv run python scripts/train.py --mode dpo --config small --resume checkpoints/sft_small/final_model.pt --auto-resume

chat:
	@echo "💬 启动交互式聊天..."
	@if [ -f checkpoints/sft_small/final_model.pt ]; then \
		uv run python scripts/generate.py --model-path checkpoints/sft_small/final_model.pt --mode chat; \
	else \
		echo "❌ 未找到模型文件，请先训练模型：make train-sft"; \
	fi

# TensorBoard监控
tensorboard:
	@echo "📊 启动TensorBoard服务..."
	uv run python scripts/tensorboard_manager.py start

tensorboard-stop:
	@echo "🛑 停止TensorBoard服务..."
	uv run python scripts/tensorboard_manager.py stop

tensorboard-status:
	@echo "🔍 查看TensorBoard状态..."
	uv run python scripts/tensorboard_manager.py status

tensorboard-list:
	@echo "📋 列出TensorBoard日志..."
	uv run python scripts/tensorboard_manager.py list

tensorboard-clean:
	@echo "🧹 清理TensorBoard旧日志..."
	uv run python scripts/tensorboard_manager.py clean --days 30

# 模型评估

eval-quick:
	@echo "🚀 快速评估（自我认知测试）..."
	@if [ -f checkpoints/sft_small/final_model.pt ]; then \
		uv run python scripts/quick_eval.py --model-path checkpoints/sft_small/final_model.pt --quick; \
	else \
		echo "❌ 未找到模型文件，请先训练模型"; \
	fi

eval-full:
	@echo "📊 完整评估（所有测试）..."
	@if [ -f checkpoints/sft_small/final_model.pt ]; then \
		uv run python scripts/quick_eval.py --model-path checkpoints/sft_small/final_model.pt; \
	else \
		echo "❌ 未找到模型文件，请先训练模型"; \
	fi

eval-categories:
	@echo "📋 列出所有评估类别..."
	uv run python scripts/quick_eval.py --list-categories

# 综合命令
all: dev-install pre-commit lint test
	@echo "✅ 所有检查通过！"
