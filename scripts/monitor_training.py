#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时监控训练进度脚本
监控检查点变化、损失趋势和生成新的损失曲线
"""
import os
import time
import subprocess
from datetime import datetime

def check_training_process():
    """检查训练进程是否还在运行"""
    try:
        result = subprocess.run(['pgrep', '-f', 'continue_training.py'], 
                              capture_output=True, text=True)
        return bool(result.stdout.strip())
    except:
        return False

def get_latest_checkpoint():
    """获取最新的检查点文件"""
    try:
        checkpoint_dir = "checkpoints/mac_medium"
        checkpoints = []
        
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_step_") and filename.endswith(".pt"):
                step_str = filename.replace("checkpoint_step_", "").replace(".pt", "")
                try:
                    step = int(step_str)
                    filepath = os.path.join(checkpoint_dir, filename)
                    mtime = os.path.getmtime(filepath)
                    checkpoints.append((step, filepath, mtime))
                except ValueError:
                    continue
        
        if checkpoints:
            # 按步数排序，返回最新的
            checkpoints.sort(key=lambda x: x[0])
            return checkpoints[-1]
        return None
    except Exception as e:
        print(f"获取检查点信息失败: {e}")
        return None

def monitor_training():
    """监控训练进度"""
    print("🔍 开始监控训练进度...")
    print("=" * 60)
    
    last_checkpoint_step = 0
    
    while True:
        try:
            # 检查训练进程
            is_running = check_training_process()
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 获取最新检查点
            latest_checkpoint = get_latest_checkpoint()
            
            if latest_checkpoint:
                step, filepath, mtime = latest_checkpoint
                checkpoint_time = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
                
                # 如果有新的检查点
                if step > last_checkpoint_step:
                    print(f"\n🎯 [{current_time}] 新检查点生成!")
                    print(f"   步数: {step}")
                    print(f"   时间: {checkpoint_time}")
                    print(f"   进度: {step/8000*100:.1f}%")
                    
                    # 更新损失曲线
                    print("   📊 更新损失曲线...")
                    try:
                        subprocess.run(['python', 'scripts/plot_training_curves.py'], 
                                     capture_output=True, check=True)
                        print("   ✅ 损失曲线已更新")
                    except subprocess.CalledProcessError as e:
                        print(f"   ⚠️  更新损失曲线失败: {e}")
                    
                    last_checkpoint_step = step
                    
                    # 检查是否接近完成
                    if step >= 8000:
                        print(f"\n🎉 训练已完成! 最终步数: {step}")
                        break
                    elif step >= 7600:
                        print(f"   🚀 训练即将完成，还差 {8000-step} 步")
            
            # 显示当前状态
            status = "🟢 运行中" if is_running else "🔴 已停止"
            current_step = latest_checkpoint[0] if latest_checkpoint else "unknown"
            progress = f"{current_step/8000*100:.1f}%" if latest_checkpoint else "unknown"
            
            print(f"\r[{current_time}] 状态: {status} | 步数: {current_step} | 进度: {progress}", end="", flush=True)
            
            # 如果训练停止了
            if not is_running:
                print(f"\n\n⚠️  训练进程已停止")
                print("检查是否正常完成或遇到错误")
                break
            
            # 等待30秒再检查
            time.sleep(30)
            
        except KeyboardInterrupt:
            print(f"\n\n👋 监控已停止")
            break
        except Exception as e:
            print(f"\n❌ 监控错误: {e}")
            time.sleep(10)

def main():
    print("🚀 MiniGPT 训练监控器")
    print("按 Ctrl+C 停止监控")
    print("-" * 60)
    
    monitor_training()
    
    print("\n📊 最终状态检查...")
    latest = get_latest_checkpoint()
    if latest:
        step, _, _ = latest
        print(f"最终步数: {step}")
        print(f"完成度: {step/8000*100:.1f}%")
        
        if step >= 8000:
            print("🎉 训练已完成!")
        else:
            print(f"还需要 {8000-step} 步完成训练")

if __name__ == "__main__":
    main()