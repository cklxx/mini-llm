#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mac优化训练 - 一键启动脚本
快速验证模型智能效果，避免Mac卡死
"""
import os
import sys
import subprocess
import time

def print_banner():
    """打印横幅"""
    print("=" * 60)
    print("🚀 MiniGPT Mac优化训练 - 智能效果快速验证")
    print("=" * 60)
    print("📦 数据量: 200条高质量对话")
    print("⚡ 训练时间: 10-20分钟")
    print("💾 内存需求: <1GB")
    print("🎯 目标: 快速验证智能效果")
    print("=" * 60)

def check_environment():
    """检查环境"""
    print("🔍 检查环境...")

    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        return False

    # 检查必要的包
    required_packages = ['torch', 'psutil']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")

    if missing_packages:
        print(f"\n📦 安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    # 检查数据文件
    data_file = "data/dataset/minimind_dataset/pretrain_200.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    else:
        print(f"✅ 数据文件存在: {data_file}")

    return True

def show_menu():
    """显示菜单"""
    print("\n🎛️  选择训练配置:")
    print("1. Tiny模型 (推荐首次使用) - 13K参数，10-20分钟")
    print("2. Small模型 (平衡性能) - 66K参数，30-45分钟")
    print("3. 测试配置 (不训练)")
    print("4. 退出")

    while True:
        choice = input("\n请选择 (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return choice
        else:
            print("无效选择，请输入1-4")

def test_config():
    """测试配置"""
    print("\n🧪 测试Mac优化配置...")
    try:
        result = subprocess.run([
            sys.executable, 'config/mac_optimized_config.py'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ 配置测试通过")
            print(result.stdout)
        else:
            print("❌ 配置测试失败")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ 配置测试超时")
    except Exception as e:
        print(f"❌ 配置测试错误: {e}")

def run_training(config_type):
    """运行训练"""
    config_names = {'1': 'tiny', '2': 'small'}
    config = config_names[config_type]

    print(f"\n🏃 开始{config}模型训练...")
    print("💡 提示:")
    print("  - 按 Ctrl+C 可以安全停止训练")
    print("  - 训练会自动保存检查点")
    print("  - 资源使用过高时会自动暂停")

    # 等待用户确认
    input("\n按回车键开始训练...")

    try:
        cmd = [
            sys.executable, 'scripts/train_optimized.py',
            '--config', config
        ]

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\n🎉 训练完成!")
            print("📁 检查点保存在: checkpoints/mac_" + config + "/")
        else:
            print(f"\n❌ 训练异常退出，返回码: {result.returncode}")

    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练错误: {e}")

def show_results():
    """显示结果"""
    print("\n📊 查看训练结果:")

    # 检查检查点目录
    checkpoint_dirs = ['checkpoints/mac_tiny', 'checkpoints/mac_small']

    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            print(f"\n📁 {checkpoint_dir}:")
            files = os.listdir(checkpoint_dir)
            for file in files:
                file_path = os.path.join(checkpoint_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  📄 {file} ({size:.1f}KB)")

def main():
    """主函数"""
    print_banner()

    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        return

    while True:
        choice = show_menu()

        if choice == '1' or choice == '2':
            run_training(choice)
            show_results()
        elif choice == '3':
            test_config()
        elif choice == '4':
            print("\n👋 再见!")
            break

        # 询问是否继续
        if choice != '4':
            continue_choice = input("\n是否继续使用? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\n👋 再见!")
                break

if __name__ == "__main__":
    main()