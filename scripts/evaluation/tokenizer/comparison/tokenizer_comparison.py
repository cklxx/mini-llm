#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 MiniGPT分词器对比分析系统
============================

系统化对比不同分词器的性能特征
执行风格: ISTJ详细记录与分析

对比维度:
1. 词汇表效率对比
2. 语言支持能力对比
3. 性能指标对比
4. 资源使用对比
5. 适用场景分析
"""

import sys
import os
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))


class TokenizerComparison:
    """分词器对比分析器"""

    def __init__(self):
        self.comparison_categories = [
            'compression_ratio',
            'vocab_size',
            'encode_speed',
            'chinese_support',
            'english_support',
            'memory_usage_mb'
        ]

    def create_comparison_matrix(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """创建对比矩阵"""
        data = []

        for tokenizer_name, metrics in evaluation_results.items():
            if isinstance(metrics, dict):
                row = {'tokenizer': tokenizer_name}
                for category in self.comparison_categories:
                    row[category] = metrics.get(category, 0.0)
                data.append(row)

        return pd.DataFrame(data)

    def generate_comparison_charts(self, df: pd.DataFrame, output_dir: str):
        """生成对比图表"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 综合性能雷达图
        self._create_radar_chart(df, os.path.join(output_dir, 'performance_radar.png'))

        # 2. 性能指标条形图
        self._create_performance_bars(df, os.path.join(output_dir, 'performance_bars.png'))

        # 3. 词汇表效率散点图
        self._create_efficiency_scatter(df, os.path.join(output_dir, 'efficiency_scatter.png'))

    def _create_radar_chart(self, df: pd.DataFrame, output_path: str):
        """创建雷达图"""
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        # 选择关键指标
        metrics = ['compression_ratio', 'encode_speed', 'chinese_support', 'english_support']

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 闭合雷达图

        for _, row in df.iterrows():
            values = []
            for metric in metrics:
                # 标准化到0-1范围
                max_val = df[metric].max()
                normalized = row[metric] / max_val if max_val > 0 else 0
                values.append(normalized)

            values = np.concatenate((values, [values[0]]))  # 闭合数据

            ax.plot(angles, values, 'o-', linewidth=2, label=row['tokenizer'])
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title('分词器性能雷达图', size=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_bars(self, df: pd.DataFrame, output_path: str):
        """创建性能条形图"""
        if df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        metrics = ['compression_ratio', 'encode_speed', 'chinese_support', 'english_support']
        titles = ['压缩率', '编码速度', '中文支持', '英文支持']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            bars = ax.bar(df['tokenizer'], df[metric])
            ax.set_title(title, fontsize=12)
            ax.tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_efficiency_scatter(self, df: pd.DataFrame, output_path: str):
        """创建效率散点图"""
        if df.empty or len(df) < 2:
            return

        plt.figure(figsize=(10, 6))

        scatter = plt.scatter(df['vocab_size'], df['compression_ratio'],
                             s=df['encode_speed']*10, alpha=0.6, c=range(len(df)))

        for i, row in df.iterrows():
            plt.annotate(row['tokenizer'],
                        (row['vocab_size'], row['compression_ratio']),
                        xytext=(5, 5), textcoords='offset points')

        plt.xlabel('词汇表大小')
        plt.ylabel('压缩率')
        plt.title('分词器效率分析 (气泡大小表示编码速度)')
        plt.colorbar(scatter, label='分词器编号')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comparison_report(self, evaluation_results: Dict[str, Any], output_path: str):
        """生成详细对比报告"""

        df = self.create_comparison_matrix(evaluation_results)

        report = {
            'summary': {
                'total_tokenizers': len(df),
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'rankings': {},
            'analysis': {},
            'recommendations': []
        }

        # 生成各指标排名
        for metric in self.comparison_categories:
            if metric in df.columns:
                ranking = df.nlargest(len(df), metric)[['tokenizer', metric]].to_dict('records')
                report['rankings'][metric] = ranking

        # 分析最佳选择
        if not df.empty:
            # 综合得分 (加权)
            weights = {
                'compression_ratio': 0.25,
                'encode_speed': 0.20,
                'chinese_support': 0.25,
                'english_support': 0.15,
                'memory_usage_mb': -0.15  # 内存使用越少越好
            }

            df_norm = df.copy()
            for metric, weight in weights.items():
                if metric in df.columns:
                    if weight > 0:
                        df_norm[f'{metric}_score'] = df[metric] / df[metric].max() * weight
                    else:
                        df_norm[f'{metric}_score'] = (1 - df[metric] / df[metric].max()) * abs(weight)

            score_columns = [f'{m}_score' for m in weights.keys() if f'{m}_score' in df_norm.columns]
            df_norm['total_score'] = df_norm[score_columns].sum(axis=1)

            best_overall = df_norm.loc[df_norm['total_score'].idxmax()]

            report['analysis']['best_overall'] = {
                'tokenizer': best_overall['tokenizer'],
                'score': best_overall['total_score'],
                'strengths': self._analyze_strengths(best_overall, df)
            }

        # 保存报告
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    def _analyze_strengths(self, best_tokenizer: pd.Series, df: pd.DataFrame) -> List[str]:
        """分析最佳分词器的优势"""
        strengths = []

        metrics_analysis = {
            'compression_ratio': '压缩效率',
            'encode_speed': '编码速度',
            'chinese_support': '中文处理',
            'english_support': '英文处理'
        }

        for metric, description in metrics_analysis.items():
            if metric in df.columns and metric in best_tokenizer:
                if best_tokenizer[metric] >= df[metric].quantile(0.75):
                    strengths.append(f'{description}表现优异')

        return strengths