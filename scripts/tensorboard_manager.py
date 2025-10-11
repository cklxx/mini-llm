#!/usr/bin/env python3
"""
TensorBoard管理脚本
提供TensorBoard的启动、停止、清理等功能
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class TensorBoardManager:
    """TensorBoard管理器"""

    def __init__(self, tensorboard_dir=None):
        self.project_root = project_root
        self.tensorboard_dir = tensorboard_dir or self.project_root / "runs"
        self.pid_file = self.project_root / ".tensorboard.pid"

    def start(self, port=6006, host="0.0.0.0", logdir=None, reload_interval=30):
        """启动TensorBoard服务

        参数:
            port: 端口号，默认6006
            host: 绑定地址，默认0.0.0.0（允许远程访问）
            logdir: 日志目录，默认使用项目的runs目录
            reload_interval: 重新加载间隔（秒）
        """
        # 检查是否已经运行
        if self.is_running():
            print("❌ TensorBoard已经在运行中")
            print(f"   PID: {self.get_pid()}")
            print("   使用 'python scripts/tensorboard_manager.py stop' 停止服务")
            return False

        logdir = logdir or str(self.tensorboard_dir)

        # 确保日志目录存在
        os.makedirs(logdir, exist_ok=True)

        # 检查日志目录是否有内容
        if not any(Path(logdir).iterdir()):
            print(f"⚠️  警告: 日志目录为空: {logdir}")
            print("   请先运行训练脚本生成TensorBoard日志")
            print("   或指定其他日志目录: --logdir <path>")

        # 构建命令
        cmd = [
            "tensorboard",
            "--logdir", logdir,
            "--port", str(port),
            "--host", host,
            "--reload_interval", str(reload_interval),
            "--bind_all"  # 允许所有网络接口访问
        ]

        print("🚀 启动TensorBoard服务...")
        print(f"   日志目录: {logdir}")
        print(f"   访问地址: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        print(f"   重载间隔: {reload_interval}秒")

        try:
            # 启动TensorBoard进程（后台运行）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            # 保存PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))

            # 等待一下确认启动成功
            time.sleep(2)

            if process.poll() is None:
                print(f"✅ TensorBoard已启动 (PID: {process.pid})")
                print("💡 使用以下命令停止:")
                print("   python scripts/tensorboard_manager.py stop")
                return True
            else:
                print("❌ TensorBoard启动失败")
                if self.pid_file.exists():
                    self.pid_file.unlink()
                return False

        except FileNotFoundError:
            print("❌ 未找到tensorboard命令")
            print("💡 请先安装TensorBoard:")
            print("   pip install tensorboard")
            print("   # 或使用uv")
            print("   uv pip install tensorboard")
            return False
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False

    def stop(self):
        """停止TensorBoard服务"""
        if not self.is_running():
            print("ℹ️  TensorBoard未运行")
            return True

        pid = self.get_pid()
        if pid is None:
            print("❌ 无法获取PID")
            return False

        try:
            print(f"🛑 停止TensorBoard (PID: {pid})...")
            os.kill(pid, signal.SIGTERM)

            # 等待进程结束
            for _ in range(10):
                try:
                    os.kill(pid, 0)  # 检查进程是否存在
                    time.sleep(0.5)
                except OSError:
                    break

            # 清理PID文件
            if self.pid_file.exists():
                self.pid_file.unlink()

            print("✅ TensorBoard已停止")
            return True

        except ProcessLookupError:
            print("ℹ️  进程已不存在")
            if self.pid_file.exists():
                self.pid_file.unlink()
            return True
        except PermissionError:
            print("❌ 权限不足，无法停止进程")
            return False
        except Exception as e:
            print(f"❌ 停止失败: {e}")
            return False

    def restart(self, **kwargs):
        """重启TensorBoard服务"""
        print("🔄 重启TensorBoard...")
        self.stop()
        time.sleep(1)
        return self.start(**kwargs)

    def status(self):
        """查看TensorBoard状态"""
        if self.is_running():
            pid = self.get_pid()
            print("✅ TensorBoard正在运行")
            print(f"   PID: {pid}")

            # 尝试获取端口信息
            try:
                result = subprocess.run(
                    ["lsof", "-p", str(pid), "-a", "-i", "TCP", "-sTCP:LISTEN"],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        port_info = lines[1].split()
                        if len(port_info) > 8:
                            port = port_info[8].split(':')[-1]
                            print(f"   端口: {port}")
                            print(f"   访问: http://localhost:{port}")
            except Exception:
                pass

            return True
        else:
            print("❌ TensorBoard未运行")
            return False

    def is_running(self):
        """检查TensorBoard是否运行"""
        pid = self.get_pid()
        if pid is None:
            return False

        try:
            os.kill(pid, 0)
            return True
        except OSError:
            # 进程不存在，清理PID文件
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False

    def get_pid(self):
        """获取TensorBoard进程PID"""
        if not self.pid_file.exists():
            return None

        try:
            with open(self.pid_file) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def list_logs(self):
        """列出所有TensorBoard日志"""
        print(f"📊 TensorBoard日志目录: {self.tensorboard_dir}\n")

        if not self.tensorboard_dir.exists():
            print("❌ 日志目录不存在")
            return

        log_dirs = sorted(
            [d for d in self.tensorboard_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not log_dirs:
            print("📁 暂无训练日志")
            print("💡 运行训练脚本将自动生成日志:")
            print("   python scripts/train.py --mode sft --config medium")
            return

        print(f"找到 {len(log_dirs)} 个训练日志:\n")

        for i, log_dir in enumerate(log_dirs, 1):
            name = log_dir.name
            mtime = datetime.fromtimestamp(log_dir.stat().st_mtime)
            size = self._get_dir_size(log_dir)

            # 计算时间差
            age = datetime.now() - mtime
            if age.days > 0:
                age_str = f"{age.days}天前"
            elif age.seconds > 3600:
                age_str = f"{age.seconds // 3600}小时前"
            else:
                age_str = f"{age.seconds // 60}分钟前"

            print(f"{i}. {name}")
            print(f"   修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age_str})")
            print(f"   大小: {size}")
            print()

    def clean_old_logs(self, days=30, dry_run=False):
        """清理旧日志

        参数:
            days: 保留最近N天的日志
            dry_run: 仅显示将要删除的内容，不实际删除
        """
        if not self.tensorboard_dir.exists():
            print("❌ 日志目录不存在")
            return

        cutoff_time = datetime.now() - timedelta(days=days)
        old_dirs = []

        for log_dir in self.tensorboard_dir.iterdir():
            if not log_dir.is_dir():
                continue

            mtime = datetime.fromtimestamp(log_dir.stat().st_mtime)
            if mtime < cutoff_time:
                old_dirs.append((log_dir, mtime))

        if not old_dirs:
            print(f"✅ 没有超过{days}天的旧日志")
            return

        print(f"🗑️  发现 {len(old_dirs)} 个超过{days}天的旧日志:\n")

        total_size = 0
        for log_dir, mtime in old_dirs:
            size = self._get_dir_size_bytes(log_dir)
            total_size += size
            age = (datetime.now() - mtime).days
            print(f"  - {log_dir.name} ({age}天前, {self._format_size(size)})")

        print(f"\n总大小: {self._format_size(total_size)}")

        if dry_run:
            print("\n💡 这是模拟运行，未实际删除")
            print("   移除 --dry-run 参数以实际删除")
            return

        # 确认删除
        response = input("\n⚠️  确定要删除这些日志吗? [y/N]: ")
        if response.lower() != 'y':
            print("❌ 已取消")
            return

        # 执行删除
        deleted = 0
        for log_dir, _ in old_dirs:
            try:
                shutil.rmtree(log_dir)
                deleted += 1
                print(f"✅ 已删除: {log_dir.name}")
            except Exception as e:
                print(f"❌ 删除失败: {log_dir.name} - {e}")

        print(f"\n✅ 清理完成，删除了 {deleted}/{len(old_dirs)} 个日志，释放空间 {self._format_size(total_size)}")

    def _get_dir_size(self, path):
        """获取目录大小（格式化）"""
        size = self._get_dir_size_bytes(path)
        return self._format_size(size)

    def _get_dir_size_bytes(self, path):
        """获取目录大小（字节）"""
        total = 0
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    def _format_size(self, size):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"


def main():
    parser = argparse.ArgumentParser(
        description='TensorBoard管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  启动TensorBoard (默认端口6006):
    python scripts/tensorboard_manager.py start

  指定端口和日志目录:
    python scripts/tensorboard_manager.py start --port 6007 --logdir runs/

  停止TensorBoard:
    python scripts/tensorboard_manager.py stop

  查看状态:
    python scripts/tensorboard_manager.py status

  列出所有日志:
    python scripts/tensorboard_manager.py list

  清理30天前的日志:
    python scripts/tensorboard_manager.py clean --days 30
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='命令')

    # start命令
    start_parser = subparsers.add_parser('start', help='启动TensorBoard')
    start_parser.add_argument('--port', type=int, default=6006, help='端口号 (默认: 6006)')
    start_parser.add_argument('--host', default='0.0.0.0', help='绑定地址 (默认: 0.0.0.0)')
    start_parser.add_argument('--logdir', help='日志目录 (默认: runs/)')
    start_parser.add_argument('--reload-interval', type=int, default=30, help='重载间隔秒数 (默认: 30)')

    # stop命令
    subparsers.add_parser('stop', help='停止TensorBoard')

    # restart命令
    restart_parser = subparsers.add_parser('restart', help='重启TensorBoard')
    restart_parser.add_argument('--port', type=int, default=6006, help='端口号 (默认: 6006)')
    restart_parser.add_argument('--host', default='0.0.0.0', help='绑定地址 (默认: 0.0.0.0)')
    restart_parser.add_argument('--logdir', help='日志目录 (默认: runs/)')
    restart_parser.add_argument('--reload-interval', type=int, default=30, help='重载间隔秒数 (默认: 30)')

    # status命令
    subparsers.add_parser('status', help='查看TensorBoard状态')

    # list命令
    subparsers.add_parser('list', help='列出所有TensorBoard日志')

    # clean命令
    clean_parser = subparsers.add_parser('clean', help='清理旧日志')
    clean_parser.add_argument('--days', type=int, default=30, help='保留最近N天的日志 (默认: 30)')
    clean_parser.add_argument('--dry-run', action='store_true', help='仅显示将要删除的内容，不实际删除')

    args = parser.parse_args()

    # 创建管理器
    manager = TensorBoardManager()

    # 执行命令
    if args.command == 'start':
        kwargs = {
            'port': args.port,
            'host': args.host,
            'reload_interval': args.reload_interval
        }
        if args.logdir:
            kwargs['logdir'] = args.logdir
        manager.start(**kwargs)

    elif args.command == 'stop':
        manager.stop()

    elif args.command == 'restart':
        kwargs = {
            'port': args.port,
            'host': args.host,
            'reload_interval': args.reload_interval
        }
        if args.logdir:
            kwargs['logdir'] = args.logdir
        manager.restart(**kwargs)

    elif args.command == 'status':
        manager.status()

    elif args.command == 'list':
        manager.list_logs()

    elif args.command == 'clean':
        manager.clean_old_logs(days=args.days, dry_run=args.dry_run)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
