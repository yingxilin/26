#!/usr/bin/env python3
"""
修复配置文件键名问题的脚本
"""

import yaml
import os

def fix_config_key():
    """修复配置文件中的键名问题"""
    config_file = 'configs/box_refinement_config.yaml'
    
    print("🔧 修复配置文件键名问题...")
    
    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查并修复hqsam配置
    if 'hqsam' in config:
        if 'checkpoint' in config['hqsam'] and 'checkpoint_path' not in config['hqsam']:
            # 添加checkpoint_path作为checkpoint的别名
            config['hqsam']['checkpoint_path'] = config['hqsam']['checkpoint']
            print("  ✅ 添加了 checkpoint_path 键")
        elif 'checkpoint_path' in config['hqsam'] and 'checkpoint' not in config['hqsam']:
            # 添加checkpoint作为checkpoint_path的别名
            config['hqsam']['checkpoint'] = config['hqsam']['checkpoint_path']
            print("  ✅ 添加了 checkpoint 键")
        else:
            print("  ✅ hqsam配置已存在")
    else:
        # 添加完整的hqsam配置
        config['hqsam'] = {
            'checkpoint': '/path/to/hqsam_weights.pth',
            'checkpoint_path': '/path/to/hqsam_weights.pth',
            'model_type': 'vit_h',
            'device': 'cuda'
        }
        print("  ✅ 添加了完整的hqsam配置")
    
    # 保存修复后的配置
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("  ✅ 配置文件已修复并保存")
    
    return True

def main():
    """主函数"""
    print("Box Refinement 配置文件修复工具")
    print("=" * 50)
    
    try:
        success = fix_config_key()
        if success:
            print("\n🎉 配置文件修复完成！")
            print("\n💡 现在可以运行训练脚本了:")
            print("python train_box_refiner_standalone.py --config configs/box_refinement_config.yaml --fast")
        else:
            print("\n❌ 配置文件修复失败")
    except Exception as e:
        print(f"\n❌ 修复过程中出现错误: {e}")

if __name__ == "__main__":
    main()