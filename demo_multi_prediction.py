#!/usr/bin/env python3
"""
🎯 多维度预测与金融评估体系演示
第四轮优化：从单一涨跌预测扩展到多维度预测 + 完整金融评估
"""

import sys
import os
import time
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

# 禁用警告
warnings.filterwarnings('ignore')

def print_header(title):
    """打印美化的标题"""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)

def print_section(title):
    """打印章节标题"""
    print(f"\n📊 {title}")
    print("-" * 60)

def demo_multi_dimensional_prediction():
    """演示多维度预测"""
    print_section("多维度预测演示")
    
    from models.multi_predictor import MultiDimensionalPredictor
    from data.data_loader import DataLoader
    from features.feature_engineer import FeatureEngineer
    
    # 获取数据
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    
    print("获取股票数据...")
    df = data_loader.get_daily("000001.SZ", start="2024-01-01", end="2024-12-01")
    
    if df.empty:
        print("❌ 无法获取数据，跳过多维度预测演示")
        return None
    
    # 特征工程
    print("生成技术指标特征...")
    features_df = feature_engineer.create_features(df, [5, 10, 20])
    
    if features_df.empty:
        print("❌ 特征工程失败")
        return None
    
    feature_cols = feature_engineer.get_feature_names()
    print(f"✅ 特征工程完成: {len(feature_cols)} 个特征")
    
    # 初始化多维度预测器
    predictor = MultiDimensionalPredictor(feature_cols)
    
    # 准备训练数据
    X = features_df[feature_cols].iloc[-100:]  # 使用最近100条数据
    train_df = df.iloc[-100:]
    
    print("准备多维度标签...")
    labels = predictor.prepare_labels(train_df)
    
    print(f"📊 标签统计:")
    for label_type, label_data in labels.items():
        if label_type == 'direction':
            up_count = (label_data == 1).sum()
            down_count = (label_data == 0).sum()
            print(f"   {label_type}: 上涨 {up_count} 次, 下跌 {down_count} 次")
        else:
            print(f"   {label_type}: 均值 {label_data.mean():.3f}, 标准差 {label_data.std():.3f}")
    
    # 训练多维度模型
    print("\n🔄 训练多维度模型...")
    models = predictor.train_multi_models(X, labels, window_size=20, trials=5)
    
    if not models:
        print("❌ 模型训练失败")
        return None
    
    print(f"✅ 训练完成，成功训练 {len(models)} 个模型:")
    for model_type, model_info in models.items():
        print(f"   📈 {model_type}: {model_info['metric']} = {model_info['score']:.4f}")
    
    # 进行预测
    print("\n🔮 生成多维度预测...")
    test_X = X.iloc[-1:] # 最后一条数据用于预测
    
    try:
        prediction = predictor.get_comprehensive_prediction(test_X, models)
        
        print("🎯 预测结果:")
        print(f"   📈 涨跌方向: {prediction['direction']['signal']}")
        print(f"   🎲 预测概率: {prediction['direction']['probability']:.1%}")
        print(f"   📊 预期涨跌幅: {prediction['magnitude']['prediction']:.2f}%")
        print(f"   📉 预期波动率: {prediction['volatility']['prediction']:.2f}%")
        print(f"   ⚠️ 风险等级: {prediction['risk_assessment']['level']}")
        
        return {'models': models, 'prediction': prediction, 'data': df}
        
    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")
        return None

def demo_financial_evaluation():
    """演示金融评估体系"""
    print_section("金融评估体系演示")
    
    from utils.financial_evaluator import FinancialEvaluator
    
    # 创建模拟预测数据
    print("生成模拟预测数据...")
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
    n_days = len(dates)
    
    # 模拟预测结果
    predictions = pd.DataFrame({
        'direction': np.random.choice([0, 1], n_days, p=[0.4, 0.6]),  # 偏向看涨
        'direction_prob': np.random.uniform(0.5, 0.95, n_days),
        'magnitude': np.random.normal(0.5, 2.0, n_days),  # 平均0.5%涨幅
        'volatility': np.random.uniform(0.5, 3.0, n_days)
    })
    
    # 模拟实际价格数据
    initial_price = 10.0
    returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
    prices = [initial_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    actual_prices = pd.DataFrame({
        'close': prices,
        'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
        'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
        'volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    print(f"✅ 模拟数据生成完成: {len(predictions)} 天的预测数据")
    
    # 金融评估
    evaluator = FinancialEvaluator()
    print("🔄 执行金融评估...")
    
    try:
        metrics = evaluator.evaluate_predictions(predictions, actual_prices)
        
        # 生成评估报告
        report = evaluator.generate_report(metrics)
        print(report)
        
        # 策略比较演示
        print("\n🔄 策略比较演示...")
        
        # 创建几个不同的策略
        conservative_strategy = predictions.copy()
        conservative_strategy['direction_prob'] = conservative_strategy['direction_prob'] * 0.8  # 降低置信度
        
        aggressive_strategy = predictions.copy()
        aggressive_strategy['direction'] = np.random.choice([0, 1], n_days, p=[0.3, 0.7])  # 更激进
        
        strategies = {
            'baseline': predictions,
            'conservative': conservative_strategy,
            'aggressive': aggressive_strategy
        }
        
        comparison = evaluator.compare_strategies(strategies, actual_prices)
        print("📊 策略比较结果:")
        print(comparison.to_string(index=False))
        
        return metrics
        
    except Exception as e:
        print(f"❌ 金融评估失败: {str(e)}")
        return None

def demo_backtest_system():
    """演示回测系统"""
    print_section("完整回测系统演示")
    
    from utils.backtest_engine import BacktestEngine
    
    # 创建回测引擎
    print("🔄 初始化回测引擎...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage=0.0005
    )
    
    try:
        print("🚀 开始回测...")
        
        # 运行回测（使用较短的时间范围进行演示）
        result = engine.run_backtest(
            stock_code="000001.SZ",
            start_date="2024-09-01",
            end_date="2024-12-01",
            train_window=60,
            rebalance_freq=10
        )
        
        if result:
            # 生成回测报告
            report = engine.generate_backtest_report(result)
            print(report)
            
            # 显示预测统计
            predictions = result['predictions']
            if len(predictions) > 0:
                print(f"\n📊 回测期间预测统计:")
                print(f"   总预测次数: {len(predictions)}")
                print(f"   看涨预测: {(predictions['direction'] == 1).sum()} 次")
                print(f"   看跌预测: {(predictions['direction'] == 0).sum()} 次")
                print(f"   平均置信度: {predictions['direction_prob'].mean():.1%}")
                print(f"   平均预测幅度: {predictions['magnitude'].mean():.2f}%")
                print(f"   平均预测波动率: {predictions['volatility'].mean():.2f}%")
            
            return result
        else:
            print("❌ 回测失败")
            return None
            
    except Exception as e:
        print(f"❌ 回测系统演示失败: {str(e)}")
        return None

def demo_web_integration():
    """演示Web界面集成"""
    print_section("Web界面多维度预测集成")
    
    import requests
    import json
    
    base_url = "http://localhost:8001"
    
    print("检查Web服务状态...")
    
    try:
        # 检查主页
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ 主页访问正常")
        else:
            print(f"❌ 主页访问异常: {response.status_code}")
            
        # 检查API文档
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API文档访问正常")
        else:
            print(f"❌ API文档访问异常: {response.status_code}")
            
        print(f"\n🌐 多维度预测Web界面:")
        print(f"   📊 主页预测: {base_url}/")
        print(f"   📖 API文档: {base_url}/docs")
        print(f"   🗂️ 缓存管理: {base_url}/cache/dashboard")
        print(f"   📈 历史记录: {base_url}/history")
        
        print(f"\n🎯 新增功能建议:")
        print(f"   • 多维度预测结果展示")
        print(f"   • 金融评估指标显示")
        print(f"   • 回测结果可视化")
        print(f"   • 风险评级展示")
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Web服务")
        print("请确保Web服务正在运行")
    except Exception as e:
        print(f"❌ Web服务检查失败: {str(e)}")

def main():
    """主函数"""
    print_header("股票预测系统 - 第四轮优化演示")
    
    print("🎯 第四轮优化重点：")
    print("   1️⃣ 多维度预测 - 方向+幅度+波动率")
    print("   2️⃣ 金融评估体系 - 15个专业量化指标")  
    print("   3️⃣ 完整回测系统 - 滚动训练+时序验证")
    print("   4️⃣ 风险评估升级 - VaR、CVaR、最大回撤")
    
    print("\n⏰ 开始第四轮优化演示...")
    total_start = time.time()
    
    try:
        # 1. 多维度预测演示
        prediction_result = demo_multi_dimensional_prediction()
        
        # 2. 金融评估演示
        evaluation_result = demo_financial_evaluation()
        
        # 3. 回测系统演示
        backtest_result = demo_backtest_system()
        
        # 4. Web集成演示
        demo_web_integration()
        
        total_time = time.time() - total_start
        
        print_header("第四轮优化完成总结")
        print(f"⏱️ 总演示时间: {total_time:.2f}秒")
        print(f"✅ 第四轮优化演示完成！")
        
        print(f"\n🎉 第四轮优化成果:")
        print(f"   🎯 预测维度: 1 → 3 (方向+幅度+波动率)")
        print(f"   📊 评估指标: 1 → 15 (完整金融量化指标)")
        print(f"   🔄 回测系统: 0 → 1 (完整历史验证)")
        print(f"   ⚠️ 风险评估: 基础 → 专业 (VaR/CVaR/风险等级)")
        
        print(f"\n📈 系统全面升级:")
        print(f"   • 从简单涨跌预测到多维度量化分析")
        print(f"   • 从准确率单一指标到15个金融指标")
        print(f"   • 从模型验证到完整历史回测")
        print(f"   • 从基础风控到专业风险管理")
        
        print(f"\n🚀 第四轮优化亮点:")
        if prediction_result:
            print(f"   ✅ 多维度预测: 成功预测方向+幅度+波动率")
        if evaluation_result:
            print(f"   ✅ 金融评估: 夏普比率 {evaluation_result.sharpe_ratio:.3f}")
        if backtest_result:
            print(f"   ✅ 回测验证: 胜率 {backtest_result['financial_metrics'].win_rate:.1%}")
        
        print(f"\n🎯 投资应用价值:")
        print(f"   • 预测精度：方向准确率 + 幅度预估")
        print(f"   • 风险控制：波动率预测 + VaR计算")
        print(f"   • 策略评估：15项金融指标全面评估")
        print(f"   • 历史验证：滚动回测确保可靠性")
        
    except KeyboardInterrupt:
        print("\n\n❌ 演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
