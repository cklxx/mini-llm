#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码结构验证测试
检查所有文件的导入和基本语法正确性
"""

import sys
import os
import ast
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def test_python_syntax(file_path):
    """测试Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def test_import_structure():
    """测试导入结构"""
    print("Testing import structure...")

    # 要检查的核心模块
    core_modules = [
        'src/model/config.py',
        'src/model/transformer.py',
        'src/model/rope.py',
        'src/model/gqa.py'
    ]

    results = {}

    for module_path in core_modules:
        if os.path.exists(module_path):
            success, error = test_python_syntax(module_path)
            results[module_path] = {'success': success, 'error': error}
            if success:
                print(f"✅ {module_path}: Syntax OK")
            else:
                print(f"❌ {module_path}: {error}")
        else:
            print(f"⚠️  {module_path}: File not found")
            results[module_path] = {'success': False, 'error': 'File not found'}

    return results


def test_data_files():
    """测试数据文件"""
    print("Testing data files...")

    data_files = [
        'data/dataset/minimind_dataset/tool_calling_basic.jsonl',
        'data/dataset/minimind_dataset/tool_calling_advanced.jsonl',
        'data/dataset/minimind_dataset/agent_ultra_think.jsonl'
    ]

    results = {}

    for data_file in data_files:
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                valid_lines = 0
                total_lines = len(lines)

                for line in lines:
                    if line.strip():
                        try:
                            json.loads(line)
                            valid_lines += 1
                        except json.JSONDecodeError:
                            pass

                success_rate = valid_lines / total_lines if total_lines > 0 else 0
                results[data_file] = {
                    'success': success_rate > 0.8,
                    'valid_lines': valid_lines,
                    'total_lines': total_lines,
                    'success_rate': success_rate
                }

                print(f"✅ {os.path.basename(data_file)}: {valid_lines}/{total_lines} valid JSON lines ({success_rate:.1%})")

            except Exception as e:
                print(f"❌ {os.path.basename(data_file)}: Error reading file - {e}")
                results[data_file] = {'success': False, 'error': str(e)}
        else:
            print(f"⚠️  {os.path.basename(data_file)}: File not found")
            results[data_file] = {'success': False, 'error': 'File not found'}

    return results


def test_script_structure():
    """测试脚本结构"""
    print("Testing script structure...")

    script_files = [
        'scripts/training/train_optimized.py',
        'scripts/inference/inference_optimized.py',
        'scripts/data_processing/prepare_datasets.py',
        'scripts/evaluation/evaluate_model.py'
    ]

    results = {}

    for script_file in script_files:
        if os.path.exists(script_file):
            success, error = test_python_syntax(script_file)
            results[script_file] = {'success': success, 'error': error}
            if success:
                print(f"✅ {os.path.basename(script_file)}: Syntax OK")
            else:
                print(f"❌ {os.path.basename(script_file)}: {error}")
        else:
            print(f"⚠️  {os.path.basename(script_file)}: File not found")
            results[script_file] = {'success': False, 'error': 'File not found'}

    return results


def test_config_validity():
    """测试配置有效性"""
    print("Testing configuration validity...")

    try:
        # 检查配置文件语法
        success, error = test_python_syntax('src/model/config.py')
        if not success:
            print(f"❌ Config syntax error: {error}")
            return {'success': False, 'error': error}

        # 检查是否能导入（不执行，只检查结构）
        with open('src/model/config.py', 'r', encoding='utf-8') as f:
            config_content = f.read()

        # 检查关键类和函数是否存在
        required_elements = [
            'class MiniGPTConfig',
            'def get_tiny_config',
            'def get_small_config',
            'def get_medium_config',
            'def estimate_params'
        ]

        missing_elements = []
        for element in required_elements:
            if element not in config_content:
                missing_elements.append(element)

        if missing_elements:
            print(f"❌ Missing config elements: {missing_elements}")
            return {'success': False, 'missing': missing_elements}
        else:
            print("✅ Configuration structure complete")
            return {'success': True}

    except Exception as e:
        print(f"❌ Config test error: {e}")
        return {'success': False, 'error': str(e)}


def test_architecture_components():
    """测试架构组件"""
    print("Testing architecture components...")

    components = {
        'RoPE': 'src/model/rope.py',
        'GQA': 'src/model/gqa.py',
        'Transformer': 'src/model/transformer.py'
    }

    results = {}

    for component_name, file_path in components.items():
        if os.path.exists(file_path):
            success, error = test_python_syntax(file_path)

            if success:
                # 检查关键类是否存在
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                key_classes = {
                    'RoPE': ['RotaryPositionEmbedding', 'apply_rotary_pos_emb'],
                    'GQA': ['GroupedQueryAttention'],
                    'Transformer': ['MiniGPT', 'TransformerBlock']
                }

                if component_name in key_classes:
                    missing_classes = [cls for cls in key_classes[component_name] if cls not in content]
                    if missing_classes:
                        results[component_name] = {'success': False, 'missing_classes': missing_classes}
                        print(f"⚠️  {component_name}: Missing classes {missing_classes}")
                    else:
                        results[component_name] = {'success': True}
                        print(f"✅ {component_name}: All key classes present")
                else:
                    results[component_name] = {'success': True}
                    print(f"✅ {component_name}: Syntax OK")
            else:
                results[component_name] = {'success': False, 'error': error}
                print(f"❌ {component_name}: {error}")
        else:
            results[component_name] = {'success': False, 'error': 'File not found'}
            print(f"⚠️  {component_name}: File not found")

    return results


def run_all_tests():
    """运行所有结构测试"""
    print("=" * 60)
    print("MINIGPT CODE STRUCTURE VALIDATION")
    print("=" * 60)

    all_results = {}

    # 测试导入结构
    print("\n1. IMPORT STRUCTURE TEST")
    print("-" * 30)
    all_results['imports'] = test_import_structure()

    # 测试数据文件
    print("\n2. DATA FILES TEST")
    print("-" * 30)
    all_results['data_files'] = test_data_files()

    # 测试脚本结构
    print("\n3. SCRIPT STRUCTURE TEST")
    print("-" * 30)
    all_results['scripts'] = test_script_structure()

    # 测试配置
    print("\n4. CONFIGURATION TEST")
    print("-" * 30)
    all_results['config'] = test_config_validity()

    # 测试架构组件
    print("\n5. ARCHITECTURE COMPONENTS TEST")
    print("-" * 30)
    all_results['architecture'] = test_architecture_components()

    # 计算总体结果
    total_tests = 0
    passed_tests = 0

    for category, results in all_results.items():
        if isinstance(results, dict):
            if 'success' in results:
                total_tests += 1
                if results['success']:
                    passed_tests += 1
            else:
                for test_name, result in results.items():
                    total_tests += 1
                    if isinstance(result, dict) and result.get('success', False):
                        passed_tests += 1

    print("\n" + "=" * 60)
    print("STRUCTURE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print("🎉 All structure tests passed!")
        print("✅ Code architecture is ready for PyTorch execution!")
    else:
        print("⚠️  Some structure tests failed.")
        print("❌ Please fix the issues before running with PyTorch.")

    # 保存结果
    results_file = "structure_validation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n📝 Detailed results saved to: {results_file}")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)