#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从检查点文件绘制训练损失曲线
独立工具脚本，可以分析现有的检查点并生成损失曲线图
"""
import os
import sys
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 添加项目路径以支持config模块加载
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# 导入config模块以支持检查点加载
try:
    from config.training_config import TrainingConfig
    from config.mac_optimized_config import get_mac_medium_config
except ImportError:
    print("⚠️  无法导入config模块，将尝试简化加载")

def extract_loss_from_checkpoints(checkpoint_dir):
    """从检查点目录提取损失信息"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    
    # 按步数排序
    def get_step_number(filepath):
        filename = os.path.basename(filepath)
        step_str = filename.replace("checkpoint_step_", "").replace(".pt", "")
        try:
            return int(step_str)
        except:
            return 0
    
    checkpoint_files.sort(key=get_step_number)
    
    steps = []
    losses = []
    
    print(f"分析 {len(checkpoint_files)} 个检查点文件...")
    
    for checkpoint_file in checkpoint_files:
        try:
            # 加载检查点，使用安全模式并忽略config对象
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            step = checkpoint.get('step', 0)
            loss = checkpoint.get('loss', 0)
            
            steps.append(step)
            losses.append(loss)
            
            print(f"步骤 {step}: 损失 {loss:.4f}")
            
        except Exception as e:
            # 尝试使用更宽松的加载方式
            try:
                import pickle
                # 先尝试直接读取基本信息，忽略复杂对象
                with open(checkpoint_file, 'rb') as f:
                    # 使用部分加载，只提取我们需要的信息
                    data = torch.load(f, map_location='cpu')
                    if isinstance(data, dict):
                        step = data.get('step', 0)
                        loss = data.get('loss', 0)
                        if step > 0:
                            steps.append(step)
                            losses.append(loss)
                            print(f"步骤 {step}: 损失 {loss:.4f} (部分加载)")
                        else:
                            raise Exception("无效数据")
                    else:
                        raise Exception("数据格式错误")
            except Exception as e2:
                print(f"⚠️  加载检查点失败 {checkpoint_file}: {e2}")
                continue
    
    return steps, losses

def plot_loss_curve(steps, losses, output_dir):
    """绘制损失曲线"""
    if not steps or not losses:
        print("没有找到有效的损失数据")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 主损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8, label='训练损失')
    plt.title('训练损失曲线 (从检查点重建)', fontsize=14, fontweight='bold')
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息
    if len(losses) > 1:
        current_loss = losses[-1]
        min_loss = min(losses)
        max_loss = max(losses)
        avg_loss = sum(losses) / len(losses)
        
        # 计算改善趋势
        if len(losses) >= 2:
            improvement = losses[0] - losses[-1]
            improvement_rate = improvement / losses[0] * 100 if losses[0] != 0 else 0
        else:
            improvement = 0
            improvement_rate = 0
        
        stats_text = f'当前: {current_loss:.4f} | 最小: {min_loss:.4f} | 最大: {max_loss:.4f} | 平均: {avg_loss:.4f}'
        improvement_text = f'改善: {improvement:.4f} ({improvement_rate:.1f}%)'
        
        plt.figtext(0.5, 0.48, stats_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.figtext(0.5, 0.44, improvement_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # 损失改善率曲线
    plt.subplot(2, 1, 2)
    if len(losses) > 1:
        # 计算相对于第一个损失的改善率
        first_loss = losses[0]
        improvement_rates = [(first_loss - loss) / first_loss * 100 for loss in losses]
        plt.plot(steps, improvement_rates, 'r-', linewidth=2, marker='s', markersize=3, label='改善率 (%)')
        plt.title('损失改善率', fontsize=12)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    else:
        plt.plot(steps, losses, 'r-', linewidth=2, marker='s', markersize=3, label='损失值')
        plt.title('损失值', fontsize=12)
    
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('改善率 (%)' if len(losses) > 1 else '损失值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存重建的损失曲线
    plot_path = os.path.join(output_dir, "loss_curve_from_checkpoints.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # 保存为latest
    latest_path = os.path.join(output_dir, "loss_curve_latest.png")
    plt.savefig(latest_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"📊 损失曲线已保存: {plot_path}")
    return plot_path

def analyze_training_progress(steps, losses):
    """分析训练进度"""
    if not steps or not losses:
        return
    
    print("\n" + "="*50)
    print("📈 训练进度分析")
    print("="*50)
    
    print(f"总训练步数: {len(steps)}")
    print(f"当前步数: {steps[-1]}")
    print(f"当前损失: {losses[-1]:.4f}")
    
    if len(losses) > 1:
        initial_loss = losses[0]
        current_loss = losses[-1]
        improvement = initial_loss - current_loss
        improvement_rate = improvement / initial_loss * 100
        
        print(f"初始损失: {initial_loss:.4f}")
        print(f"损失改善: {improvement:.4f} ({improvement_rate:.1f}%)")
        print(f"最佳损失: {min(losses):.4f}")
        print(f"最差损失: {max(losses):.4f}")
        print(f"平均损失: {sum(losses)/len(losses):.4f}")
        
        # 分析最近的趋势
        if len(losses) >= 5:
            recent_losses = losses[-5:]
            recent_trend = recent_losses[-1] - recent_losses[0]
            if recent_trend < 0:
                print(f"最近趋势: 🟢 改善中 ({recent_trend:.4f})")
            elif recent_trend > 0:
                print(f"最近趋势: 🔴 上升中 (+{recent_trend:.4f})")
            else:
                print(f"最近趋势: 🟡 稳定")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从检查点绘制训练损失曲线')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/mac_medium',
                        help='检查点目录路径')
    parser.add_argument('--output-dir', type=str, default='checkpoints/mac_medium/plots',
                        help='输出图片目录')
    
    args = parser.parse_args()
    
    print(f"🔍 分析检查点目录: {args.checkpoint_dir}")
    
    # 提取损失信息
    steps, losses = extract_loss_from_checkpoints(args.checkpoint_dir)
    
    if not steps:
        print("❌ 没有找到有效的检查点文件")
        return
    
    # 分析训练进度
    analyze_training_progress(steps, losses)
    
    # 绘制损失曲线
    print(f"\n📊 绘制损失曲线...")
    plot_path = plot_loss_curve(steps, losses, args.output_dir)
    
    print(f"\n✅ 完成！损失曲线已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()