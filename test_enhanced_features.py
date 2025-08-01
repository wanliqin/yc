#!/usr/bin/env python3
"""
测试增强特征工程的效果
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from features.feature_engineer import FeatureEngineer
from models.stock_predictor import StockPredictor

def create_test_data(n_samples=300):
    """创建模拟股票数据"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples)
    
    # 模拟价格走势
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_samples)  # 日收益率
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 创建OHLC数据
    closes = np.array(prices)
    opens = closes * (1 + np.random.normal(0, 0.005, n_samples))
    highs = np.maximum(opens, closes) * (1 + np.random.exponential(0.01, n_samples))
    lows = np.minimum(opens, closes) * (1 - np.random.exponential(0.01, n_samples))
    volumes = np.random.lognormal(10, 0.5, n_samples).astype(int)
    
    return pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'vol': volumes
    })

def test_feature_comparison():
    """对比原有特征和增强特征的效果"""
    print("=== 股票预测系统特征增强测试 ===\n")
    
    # 创建测试数据
    print("1. 创建测试数据...")
    test_data = create_test_data()
    print(f"   - 测试数据形状: {test_data.shape}")
    print(f"   - 日期范围: {test_data['date'].min()} 到 {test_data['date'].max()}")
    
    # 测试新的特征工程
    print("\n2. 测试增强特征工程...")
    fe = FeatureEngineer()
    
    # 处理特征
    features_df = fe.add_technical_indicators(test_data)
    X, y = fe.prepare_features(test_data)
    
    print(f"   - 原始数据: {test_data.shape[0]} 行")
    print(f"   - 处理后数据: {features_df.shape[0]} 行")
    print(f"   - 特征数量: {X.shape[1]} 个")
    print(f"   - 标签分布: 上涨 {sum(y)} 次, 下跌 {len(y) - sum(y)} 次")
    
    # 显示特征列表
    print("\n3. 特征列表:")
    feature_names = fe.get_feature_names()
    
    # 分类显示特征
    original_features = ["rsi", "k", "d", "j", "bbp", "macd", "macd_signal", "ret"]
    volume_features = ["vol_ma", "vol_ratio", "amount_ratio", "price_volume_trend"]
    trend_features = ["ema", "williams_r", "momentum", "dpo", "trix"]
    volatility_features = ["atr", "bb_width", "keltner_channel", "historical_volatility"]
    
    print(f"   原有特征 ({len(original_features)}个): {original_features}")
    print(f"   量价特征 ({len(volume_features)}个): {volume_features}")
    print(f"   趋势特征 ({len(trend_features)}个): {trend_features}")
    print(f"   波动率特征 ({len(volatility_features)}个): {volatility_features}")
    
    # 检查特征质量
    print("\n4. 特征质量检查:")
    print(f"   - 数据缺失值: {X.isnull().sum().sum()}")
    print(f"   - 无限值: {np.isinf(X.values).sum()}")
    print(f"   - 特征范围:")
    
    for col in X.columns:
        col_data = X[col]
        print(f"     {col}: [{col_data.min():.4f}, {col_data.max():.4f}], std={col_data.std():.4f}")
    
    # 测试模型兼容性
    print("\n5. 模型兼容性测试:")
    try:
        predictor = StockPredictor(feature_names)
        print("   ✓ StockPredictor 创建成功")
        
        # 快速训练测试（小trials）
        if len(X) >= 60:
            print("   - 测试模型训练...")
            model, params, acc = predictor.train_model(X, y, window_size=30, trials=5, feature_columns=feature_names)
            if model is not None:
                print(f"   ✓ 模型训练成功，验证准确率: {acc:.4f}")
                
                # 测试预测 - 需要添加label列用于预测接口
                test_X = X.iloc[-5:].copy()  # 取最后5行数据测试
                test_X['label'] = y.iloc[-5:]  # 添加label列
                pred_result = predictor.predict(test_X, [30], 5, feature_names)
                print(f"   ✓ 预测测试成功，预测结果类型: {type(pred_result)}")
            else:
                print("   ⚠ 模型训练失败（可能数据不足）")
        else:
            print("   ⚠ 数据不足，跳过训练测试")
            
    except Exception as e:
        print(f"   ✗ 模型测试失败: {e}")
    
    print("\n=== 测试完成 ===")
    print(f"总结: 特征数量从 8 个增加到 {len(feature_names)} 个")
    print("新增了量价、趋势、波动率三大类共 13 个特征")
    print("预期能够提升模型的预测精度和稳定性")

if __name__ == "__main__":
    test_feature_comparison()
