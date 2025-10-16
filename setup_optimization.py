#!/usr/bin/env python3
"""
Box Refinement 优化版本安装和设置脚本
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

def check_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    print(f"✅ Python版本: {sys.version}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA支持
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU (性能较低)")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查其他依赖
    required_packages = ['numpy', 'opencv-python', 'matplotlib', 'tqdm', 'pyyaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def setup_directories():
    """创建必要的目录"""
    print("\n📁 创建目录结构...")
    
    directories = [
        'outputs/box_refinement',
        'logs/box_refinement', 
        'checkpoints/box_refinement',
        'visualizations/box_refinement',
        'features/train',
        'features/val'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def backup_original():
    """备份原始训练脚本"""
    print("\n💾 备份原始文件...")
    
    original_file = "train_box_refiner.py"
    backup_file = "train_box_refiner_original.py"
    
    if Path(original_file).exists() and not Path(backup_file).exists():
        shutil.copy2(original_file, backup_file)
        print(f"✅ 原始文件已备份为: {backup_file}")
    else:
        print(f"ℹ️  原始文件已备份或不存在")

def create_requirements():
    """创建requirements.txt"""
    print("\n📝 创建requirements.txt...")
    
    requirements = """# Box Refinement 优化版本依赖
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.5.0
tqdm>=4.64.0
pyyaml>=6.0
pyarrow>=8.0.0  # 用于parquet文件支持
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ requirements.txt 已创建")

def create_launch_script():
    """创建启动脚本"""
    print("\n🚀 创建启动脚本...")
    
    # Windows批处理脚本
    windows_script = """@echo off
echo Box Refinement 优化训练启动器
echo ================================

echo.
echo 选择训练模式:
echo 1. 普通训练 (无优化)
echo 2. 快速模式 (所有优化)
echo 3. 调试模式 (少量数据)
echo 4. 性能测试
echo 5. 退出

set /p choice="请输入选择 (1-5): "

if "%choice%"=="1" (
    python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml
) else if "%choice%"=="2" (
    python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml --fast
) else if "%choice%"=="3" (
    python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml --debug
) else if "%choice%"=="4" (
    python test_optimization_performance.py
) else if "%choice%"=="5" (
    exit
) else (
    echo 无效选择
    pause
)

pause
"""
    
    with open("run_training.bat", "w") as f:
        f.write(windows_script)
    
    # Linux/Mac shell脚本
    linux_script = """#!/bin/bash
echo "Box Refinement 优化训练启动器"
echo "================================"

echo ""
echo "选择训练模式:"
echo "1. 普通训练 (无优化)"
echo "2. 快速模式 (所有优化)"
echo "3. 调试模式 (少量数据)"
echo "4. 性能测试"
echo "5. 退出"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml
        ;;
    2)
        python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml --fast
        ;;
    3)
        python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml --debug
        ;;
    4)
        python test_optimization_performance.py
        ;;
    5)
        exit
        ;;
    *)
        echo "无效选择"
        ;;
esac
"""
    
    with open("run_training.sh", "w") as f:
        f.write(linux_script)
    
    # 设置执行权限
    os.chmod("run_training.sh", 0o755)
    
    print("✅ 启动脚本已创建:")
    print("   - Windows: run_training.bat")
    print("   - Linux/Mac: run_training.sh")

def show_usage_instructions():
    """显示使用说明"""
    print("\n📖 使用说明")
    print("=" * 50)
    
    print("\n1. 基本使用:")
    print("   python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml")
    
    print("\n2. 快速模式 (推荐):")
    print("   python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml --fast")
    
    print("\n3. 调试模式:")
    print("   python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml --debug")
    
    print("\n4. 性能测试:")
    print("   python test_optimization_performance.py")
    
    print("\n5. 使用启动器:")
    print("   Windows: run_training.bat")
    print("   Linux/Mac: ./run_training.sh")
    
    print("\n📊 预期性能提升:")
    print("   - 特征缓存: 30-50x 加速")
    print("   - 数据抽样: 10x 加速")
    print("   - 混合精度: 1.5-2x 加速")
    print("   - 综合效果: 50-100x 加速")
    
    print("\n⚠️  注意事项:")
    print("   - 首次运行会生成特征缓存，需要较长时间")
    print("   - 确保有足够的磁盘空间存储缓存文件")
    print("   - 建议使用GPU以获得最佳性能")

def main():
    """主函数"""
    print("Box Refinement 优化版本安装程序")
    print("=" * 50)
    
    # 检查系统要求
    if not check_requirements():
        print("\n❌ 系统要求检查失败，请安装缺少的依赖")
        return
    
    # 设置目录
    setup_directories()
    
    # 备份原始文件
    backup_original()
    
    # 创建requirements.txt
    create_requirements()
    
    # 创建启动脚本
    create_launch_script()
    
    # 显示使用说明
    show_usage_instructions()
    
    print("\n✅ 安装完成!")
    print("\n🎉 现在可以开始使用优化版本的 Box Refinement 训练了!")
    print("\n💡 建议:")
    print("   1. 先运行性能测试了解优化效果")
    print("   2. 使用 --fast 模式进行快速训练")
    print("   3. 查看 README_OPTIMIZATION.md 了解详细功能")

if __name__ == "__main__":
    main()