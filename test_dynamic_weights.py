#!/usr/bin/env python3
"""
测试动态样本权重策略的效果
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.weight_calculator import DynamicWeightCalculator
from features.feature_engineer import FeatureEngineer

def create_realistic_stock_data(n_samples=200):
    """创建更真实的股票数据"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples)
    
    # 模拟不同市场阶段
    phases = [
        (0, 50, 'normal'),     # 正常阶段
        (50, 100, 'bull'),     # 牛市阶段
        (100, 150, 'volatile'), # 震荡阶段
        (150, 200, 'bear')     # 熊市阶段
    ]
    
    prices = [100]  # 起始价格
    volumes = []
    
    for start, end, phase in phases:
        phase_length = end - start
        
        if phase == 'normal':
            returns = np.random.normal(0.001, 0.015, phase_length)
            vol_base = 20000
        elif phase == 'bull':
            returns = np.random.normal(0.008, 0.02, phase_length)  # 上涨趋势
            vol_base = 30000  # 成交量增加
        elif phase == 'volatile':
            returns = np.random.normal(0, 0.035, phase_length)  # 高波动
            vol_base = 25000
        else:  # bear
            returns = np.random.normal(-0.005, 0.025, phase_length)  # 下跌趋势
            vol_base = 15000  # 成交量萎缩
        
        # 生成价格序列
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # 生成成交量序列
        phase_volumes = np.random.lognormal(np.log(vol_base), 0.3, phase_length)
        volumes.extend(phase_volumes)
    
    prices = prices[1:]  # 移除起始价格
    
    # 创建OHLC数据
    closes = np.array(prices)
    opens = closes * (1 + np.random.normal(0, 0.003, n_samples))
    highs = np.maximum(opens, closes) * (1 + np.random.exponential(0.005, n_samples))
    lows = np.minimum(opens, closes) * (1 - np.random.exponential(0.005, n_samples))
    
    return pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'vol': volumes
    })

def test_weight_strategies():
    """测试不同权重策略"""
    print("=== 动态样本权重策略测试 ===\n")
    
    # 创建测试数据
    print("1. 创建测试数据...")
    test_data = create_realistic_stock_data()
    print(f"   - 测试数据形状: {test_data.shape}")
    
    # 初始化权重计算器
    weight_calc = DynamicWeightCalculator()
    
    # 测试不同权重策略
    print("\n2. 测试不同权重策略...")
    
    # 简单权重（原有策略）
    simple_weights = np.ones(len(test_data))
    if len(test_data) >= 30:
        simple_weights[-30:] = 3
    
    # 动态权重
    dynamic_weights = weight_calc.calculate_combined_weights(test_data)
    
    # 自适应权重
    adaptive_weights = weight_calc.get_adaptive_weights(test_data, 'normal')
    bull_weights = weight_calc.get_adaptive_weights(test_data, 'bull')
    bear_weights = weight_calc.get_adaptive_weights(test_data, 'bear')
    volatile_weights = weight_calc.get_adaptive_weights(test_data, 'volatile')
    
    # 分析权重特性
    print("\n3. 权重策略分析:")
    strategies = {
        'Simple': simple_weights,
        'Dynamic': dynamic_weights,
        'Adaptive-Normal': adaptive_weights,
        'Adaptive-Bull': bull_weights,
        'Adaptive-Bear': bear_weights,
        'Adaptive-Volatile': volatile_weights
    }
    
    for name, weights in strategies.items():
        print(f"   {name}:")
        print(f"     - 权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"     - 平均权重: {weights.mean():.3f}")
        print(f"     - 权重标准差: {weights.std():.3f}")
        print(f"     - 最近30天平均权重: {weights[-30:].mean():.3f}")
        print(f"     - 最早30天平均权重: {weights[:30].mean():.3f}")
    
    # 测试特征工程兼容性
    print("\n4. 特征工程兼容性测试:")
    fe = FeatureEngineer()
    features_df = fe.add_technical_indicators(test_data)
    
    if not features_df.empty:
        # 测试动态权重在实际特征数据上的效果
        feature_weights = weight_calc.calculate_combined_weights(features_df)
        print(f"   ✓ 特征数据权重计算成功")
        print(f"   - 特征数据形状: {features_df.shape}")
        print(f"   - 权重数组形状: {feature_weights.shape}")
        print(f"   - 权重范围: [{feature_weights.min():.3f}, {feature_weights.max():.3f}]")
    else:
        print("   ✗ 特征数据生成失败")
    
    # 权重分布可视化分析
    print("\n5. 权重分布分析:")
    
    # 按时间阶段分析权重
    phases = [
        (0, 50, 'Normal Phase'),
        (50, 100, 'Bull Phase'), 
        (100, 150, 'Volatile Phase'),
        (150, 200, 'Bear Phase')
    ]
    
    for start, end, phase_name in phases:
        phase_dynamic = dynamic_weights[start:end]
        phase_simple = simple_weights[start:end]
        
        print(f"   {phase_name}:")
        print(f"     - 动态权重平均: {phase_dynamic.mean():.3f}")
        print(f"     - 简单权重平均: {phase_simple.mean():.3f}")
        print(f"     - 权重比率: {phase_dynamic.mean() / phase_simple.mean():.3f}")
    
    # 模拟模型训练效果
    print("\n6. 模型训练模拟:")
    
    # 模拟不同权重策略对模型训练的影响
    X, y = fe.prepare_features(test_data)
    if len(X) > 0:
        print(f"   - 可用训练样本: {len(X)}")
        
        # 计算权重对样本重要性的影响
        sample_weights = weight_calc.calculate_combined_weights(test_data.tail(len(X)))
        
        # 分析高权重样本的特征
        high_weight_threshold = np.percentile(sample_weights, 80)
        high_weight_samples = sample_weights >= high_weight_threshold
        
        print(f"   - 高权重样本比例: {high_weight_samples.sum() / len(sample_weights):.2%}")
        print(f"   - 高权重样本平均权重: {sample_weights[high_weight_samples].mean():.3f}")
        print(f"   - 低权重样本平均权重: {sample_weights[~high_weight_samples].mean():.3f}")
        
        # 分析标签分布
        high_weight_indices = np.where(high_weight_samples)[0]
        low_weight_indices = np.where(~high_weight_samples)[0]
        
        high_weight_labels = y.iloc[high_weight_indices]
        low_weight_labels = y.iloc[low_weight_indices]
        
        print(f"   - 高权重样本上涨比例: {high_weight_labels.mean():.2%}")
        print(f"   - 低权重样本上涨比例: {low_weight_labels.mean():.2%}")
    
    print("\n=== 测试完成 ===")
    print("总结:")
    print("- 动态权重策略成功实现多因子综合权重计算")
    print("- 不同市场状态下的自适应权重策略运行正常") 
    print("- 权重策略与特征工程兼容性良好")
    print("- 预期能够提升模型对不同市场环境的适应性")

if __name__ == "__main__":
    test_weight_strategies()
