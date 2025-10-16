#!/usr/bin/env python3
"""
简单的语法检查测试
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试模块导入"""
    print("Testing module imports...")
    
    try:
        # 添加modules目录到路径
        sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
        
        # 测试box_refinement模块导入
        print("Testing box_refinement module...")
        import modules.box_refinement
        print("✓ box_refinement module imported successfully")
        
        # 测试hqsam_feature_extractor模块导入
        print("Testing hqsam_feature_extractor module...")
        import modules.hqsam_feature_extractor
        print("✓ hqsam_feature_extractor module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\nTesting file structure...")
    
    required_files = [
        'modules/box_refinement.py',
        'modules/hqsam_feature_extractor.py',
        'configs/box_refinement_config.yaml',
        'configs/box_refinement_config_local.yaml',
        'train_box_refiner.py',
        'test_box_refinement.py',
        'infer_yolo_hqsam_with_refinement.py',
        'README_BoxRefinement.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - Missing!")
            all_exist = False
    
    return all_exist

def test_syntax():
    """测试语法"""
    print("\nTesting syntax...")
    
    python_files = [
        'modules/box_refinement.py',
        'modules/hqsam_feature_extractor.py',
        'train_box_refiner.py',
        'test_box_refinement.py',
        'infer_yolo_hqsam_with_refinement.py'
    ]
    
    all_syntax_ok = True
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的语法检查
            compile(content, file_path, 'exec')
            print(f"✓ {file_path} - Syntax OK")
            
        except SyntaxError as e:
            print(f"✗ {file_path} - Syntax Error: {e}")
            all_syntax_ok = False
        except Exception as e:
            print(f"✗ {file_path} - Error: {e}")
            all_syntax_ok = False
    
    return all_syntax_ok

def main():
    """运行所有测试"""
    print("="*60)
    print("Box Refinement Module - Simple Test")
    print("="*60)
    
    all_passed = True
    
    # 测试文件结构
    if not test_file_structure():
        all_passed = False
    
    # 测试语法
    if not test_syntax():
        all_passed = False
    
    # 测试导入
    if not test_imports():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All basic tests passed! Files are ready.")
        print("\nTo run with your environment:")
        print("& C:\\Users\\23215\\micromamba\\Library\\bin\\micromamba.exe run -n fungitastic-seg python test_modules.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    print("="*60)

if __name__ == "__main__":
    main()