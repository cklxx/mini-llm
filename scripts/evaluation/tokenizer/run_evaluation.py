#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 MiniGPT分词器一键评测脚本
==============================

快速启动分词器全面评估的便捷工具
执行风格: ISTJ系统化自动化执行

功能特性:
1. 一键完整评估所有训练好的分词器
2. 自动生成对比分析报告
3. 输出标准化的评估结果
4. 支持自定义评估参数
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from scripts.evaluation.tokenizer.comprehensive_tokenizer_evaluation import TokenizerEvaluator
    from scripts.evaluation.tokenizer.comparison.tokenizer_comparison import TokenizerComparison
    from scripts.evaluation.tokenizer.benchmarks.tokenizer_benchmark import TokenizerBenchmark
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保在MiniGPT项目根目录下运行此脚本")
    sys.exit(1)


class OneClickEvaluator:
    """一键评测管理器"""

    def __init__(self):
        self.project_root = project_root
        self.tokenizer_dir = self.project_root / "tokenizers" / "trained_models"
        self.output_dir = self.project_root / "results" / "tokenizer_evaluation"

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

    def find_tokenizers(self):
        """自动发现可用的分词器"""
        tokenizers = []

        if self.tokenizer_dir.exists():
            for file in self.tokenizer_dir.glob("*.pkl"):
                tokenizers.append(file)

        # 如果训练目录没有分词器，检查checkpoints目录
        if not tokenizers:
            checkpoints_dir = self.project_root / "checkpoints"
            if checkpoints_dir.exists():
                for file in checkpoints_dir.glob("*tokenizer*.pkl"):
                    tokenizers.append(file)

        return sorted(tokenizers)

    def run_complete_evaluation(self, custom_tokenizers=None, iterations=50):
        """运行完整评估流程"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("🚀 MiniGPT分词器一键评测启动")
        print("=" * 50)

        # 1. 发现分词器
        tokenizers = custom_tokenizers if custom_tokenizers else self.find_tokenizers()

        if not tokenizers:
            print("❌ 未发现任何分词器文件！")
            print("请检查以下目录:")
            print(f"   - {self.tokenizer_dir}")
            print(f"   - {self.project_root / 'checkpoints'}")
            return

        print(f"🔍 发现 {len(tokenizers)} 个分词器:")
        for i, tokenizer in enumerate(tokenizers, 1):
            print(f"   {i}. {tokenizer.name}")

        print()

        # 2. 运行综合评估
        print("📊 开始综合评估...")
        evaluator = TokenizerEvaluator()
        evaluation_results = {}

        for tokenizer_path in tokenizers:
            print(f"  ⚡ 评估: {tokenizer_path.name}")
            try:
                result = evaluator.evaluate_tokenizer(str(tokenizer_path))
                if result:
                    evaluation_results[tokenizer_path.name] = result.__dict__
                    print(f"     ✅ 完成")
                else:
                    print(f"     ❌ 失败")
            except Exception as e:
                print(f"     ❌ 错误: {e}")

        if not evaluation_results:
            print("❌ 所有分词器评估都失败了！")
            return

        print(f"\n✅ 综合评估完成，成功评估 {len(evaluation_results)} 个分词器")

        # 3. 生成对比分析
        print("\n📈 生成对比分析...")
        comparison = TokenizerComparison()

        # 创建对比矩阵
        df = comparison.create_comparison_matrix(evaluation_results)

        # 生成图表
        charts_dir = self.output_dir / "charts" / timestamp
        comparison.generate_comparison_charts(df, str(charts_dir))
        print(f"   📊 图表已保存到: {charts_dir}")

        # 生成对比报告
        report_path = self.output_dir / "reports" / f"comparison_report_{timestamp}.json"
        comparison_report = comparison.generate_comparison_report(evaluation_results, str(report_path))
        print(f"   📋 报告已保存到: {report_path}")

        # 4. 显示关键结果
        print("\n🏆 评估结果摘要:")
        print("-" * 40)

        for name, metrics in evaluation_results.items():
            print(f"\n📁 {name}:")
            print(f"   词汇表大小: {metrics.get('vocab_size', 'N/A'):,}")
            print(f"   压缩率: {metrics.get('compression_ratio', 0):.2f}")
            print(f"   编码速度: {metrics.get('encode_speed', 0):.0f} tokens/sec")
            print(f"   中文支持: {metrics.get('chinese_support', 0):.2f}")
            print(f"   英文支持: {metrics.get('english_support', 0):.2f}")

        # 5. 显示最佳推荐
        if 'analysis' in comparison_report and 'best_overall' in comparison_report['analysis']:
            best = comparison_report['analysis']['best_overall']
            print(f"\n🌟 综合最佳: {best['tokenizer']}")
            print(f"   综合得分: {best['score']:.3f}")
            if 'strengths' in best:
                print(f"   优势特点: {', '.join(best['strengths'])}")

        print(f"\n📂 所有结果已保存到: {self.output_dir}")
        print("🎉 一键评测完成！")

        return evaluation_results, comparison_report


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="MiniGPT分词器一键评测工具")
    parser.add_argument("--tokenizers", nargs="+", help="指定要评估的分词器文件路径")
    parser.add_argument("--iterations", type=int, default=50, help="基准测试迭代次数 (默认: 50)")
    parser.add_argument("--output", help="输出目录 (默认: results/tokenizer_evaluation)")

    args = parser.parse_args()

    evaluator = OneClickEvaluator()

    # 如果指定了输出目录
    if args.output:
        evaluator.output_dir = Path(args.output)
        evaluator.output_dir.mkdir(parents=True, exist_ok=True)

    # 转换tokenizer路径
    custom_tokenizers = None
    if args.tokenizers:
        custom_tokenizers = [Path(p) for p in args.tokenizers]

        # 验证文件存在
        missing_files = [p for p in custom_tokenizers if not p.exists()]
        if missing_files:
            print("❌ 以下文件不存在:")
            for f in missing_files:
                print(f"   {f}")
            return 1

    # 运行评估
    try:
        evaluator.run_complete_evaluation(custom_tokenizers, args.iterations)
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断评估")
        return 1
    except Exception as e:
        print(f"❌ 评估过程出现错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())