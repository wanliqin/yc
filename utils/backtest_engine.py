"""
完整的回测系统
支持多维度预测的历史回测验证
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings
from utils.financial_evaluator import FinancialEvaluator, FinancialMetrics
from models.multi_predictor import MultiDimensionalPredictor
from features.feature_engineer import FeatureEngineer
from data.data_loader import DataLoader
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005):
        """
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.evaluator = FinancialEvaluator()
        
    def run_backtest(self, 
                    stock_code: str,
                    start_date: str,
                    end_date: str,
                    train_window: int = 120,
                    rebalance_freq: int = 20) -> Dict[str, Any]:
        """运行完整回测
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            train_window: 训练窗口大小
            rebalance_freq: 重新训练频率（天）
            
        Returns:
            回测结果字典
        """
        print(f"🔄 开始回测: {stock_code} ({start_date} 至 {end_date})")
        
        # 获取数据
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer()
        
        # 获取更长的历史数据用于训练
        extended_start = pd.to_datetime(start_date) - timedelta(days=train_window + 100)
        df = data_loader.get_daily(
            stock_code, 
            start=extended_start.strftime("%Y-%m-%d"), 
            end=end_date
        )
        
        if df.empty:
            raise ValueError(f"无法获取{stock_code}的数据")
            
        print(f"📊 获取数据: {len(df)} 条记录")
        
        # 特征工程
        features_df = feature_engineer.create_features(df, [5, 10, 20])
        if features_df.empty:
            raise ValueError("特征工程失败")
            
        feature_cols = feature_engineer.get_feature_names()
        print(f"🔧 特征工程: {len(feature_cols)} 个特征")
        
        # 初始化预测器
        predictor = MultiDimensionalPredictor(feature_cols)
        
        # 回测开始时间索引
        backtest_start_idx = None
        for i, date in enumerate(features_df.index):
            if pd.to_datetime(date) >= pd.to_datetime(start_date):
                backtest_start_idx = i
                break
                
        if backtest_start_idx is None:
            raise ValueError("回测开始日期超出数据范围")
            
        print(f"📅 回测期间: {backtest_start_idx} 至 {len(features_df)-1}")
        
        # 执行滚动回测
        predictions_list = []
        actual_prices_list = []
        
        for current_idx in range(backtest_start_idx, len(features_df) - 1, rebalance_freq):
            end_idx = min(current_idx + rebalance_freq, len(features_df) - 1)
            
            print(f"🔄 训练模型: 第 {current_idx} 到 {end_idx} 天")
            
            # 准备训练数据
            train_start = max(0, current_idx - train_window)
            train_end = current_idx
            
            if train_end - train_start < 50:  # 训练数据不足
                continue
                
            X_train = features_df[feature_cols].iloc[train_start:train_end]
            
            # 准备多维度标签
            train_df = df.iloc[train_start:train_end]
            labels = predictor.prepare_labels(train_df)
            
            # 训练模型
            try:
                models = predictor.train_multi_models(
                    X_train, labels, window_size=30, trials=10
                )
                
                if not models:
                    continue
                    
                # 预测未来几天
                for pred_idx in range(current_idx, end_idx):
                    if pred_idx >= len(features_df) - 1:
                        break
                        
                    X_pred = features_df[feature_cols].iloc[pred_idx:pred_idx+1]
                    pred_result = predictor.get_comprehensive_prediction(X_pred, models)
                    
                    # 记录预测结果
                    predictions_list.append({
                        'date': features_df.index[pred_idx],
                        'direction': pred_result['direction']['prediction'],
                        'direction_prob': pred_result['direction']['probability'],
                        'magnitude': pred_result['magnitude']['prediction'],
                        'volatility': pred_result['volatility']['prediction']
                    })
                    
                    # 记录实际价格
                    actual_prices_list.append({
                        'date': df.index[pred_idx],
                        'close': df['close'].iloc[pred_idx],
                        'open': df['open'].iloc[pred_idx] if 'open' in df.columns else df['close'].iloc[pred_idx],
                        'high': df['high'].iloc[pred_idx] if 'high' in df.columns else df['close'].iloc[pred_idx],
                        'low': df['low'].iloc[pred_idx] if 'low' in df.columns else df['close'].iloc[pred_idx],
                        'volume': df['volume'].iloc[pred_idx] if 'volume' in df.columns else 0
                    })
                    
            except Exception as e:
                print(f"❌ 模型训练失败: {str(e)}")
                continue
        
        if not predictions_list:
            raise ValueError("回测过程中没有生成有效预测")
            
        # 转换为DataFrame
        predictions_df = pd.DataFrame(predictions_list)
        actual_prices_df = pd.DataFrame(actual_prices_list)
        
        print(f"✅ 生成预测: {len(predictions_df)} 条")
        
        # 金融评估
        metrics = self.evaluator.evaluate_predictions(
            predictions_df, actual_prices_df, self.initial_capital
        )
        
        # 生成详细结果
        result = {
            'stock_code': stock_code,
            'period': f"{start_date} 至 {end_date}",
            'predictions': predictions_df,
            'actual_prices': actual_prices_df,
            'financial_metrics': metrics,
            'backtest_config': {
                'initial_capital': self.initial_capital,
                'commission_rate': self.commission_rate,
                'slippage': self.slippage,
                'train_window': train_window,
                'rebalance_freq': rebalance_freq
            }
        }
        
        print("🎉 回测完成!")
        return result
    
    def compare_strategies(self, 
                          backtest_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """比较多个策略的回测结果
        
        Args:
            backtest_configs: 回测配置列表，每个包含参数设置
            
        Returns:
            策略比较DataFrame
        """
        comparison_results = []
        
        for i, config in enumerate(backtest_configs):
            strategy_name = config.get('name', f'Strategy_{i+1}')
            
            try:
                result = self.run_backtest(**config.get('params', {}))
                metrics = result['financial_metrics']
                
                comparison_results.append({
                    'Strategy': strategy_name,
                    'Total Return': f"{metrics.total_return:.2%}",
                    'Annual Return': f"{metrics.annual_return:.2%}",
                    'Sharpe Ratio': f"{metrics.sharpe_ratio:.3f}",
                    'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                    'Win Rate': f"{metrics.win_rate:.2%}",
                    'Total Trades': metrics.total_trades,
                    'Calmar Ratio': f"{metrics.calmar_ratio:.3f}"
                })
                
            except Exception as e:
                print(f"❌ 策略 {strategy_name} 回测失败: {str(e)}")
                comparison_results.append({
                    'Strategy': strategy_name,
                    'Total Return': 'Error',
                    'Annual Return': 'Error',
                    'Sharpe Ratio': 'Error',
                    'Max Drawdown': 'Error',
                    'Win Rate': 'Error',
                    'Total Trades': 0,
                    'Calmar Ratio': 'Error'
                })
        
        return pd.DataFrame(comparison_results)
    
    def generate_backtest_report(self, result: Dict[str, Any]) -> str:
        """生成回测报告
        
        Args:
            result: 回测结果
            
        Returns:
            格式化报告
        """
        metrics = result['financial_metrics']
        predictions = result['predictions']
        config = result['backtest_config']
        
        report = f"""
📊 股票预测系统回测报告
{'='*60}

🎯 基本信息:
   股票代码: {result['stock_code']}
   回测期间: {result['period']}
   初始资金: ¥{config['initial_capital']:,.0f}
   训练窗口: {config['train_window']} 天
   重训频率: {config['rebalance_freq']} 天

📈 预测统计:
   总预测次数: {len(predictions)}
   预测准确率: {metrics.win_rate:.2%}
   
🎯 预测维度分析:
"""
        
        if len(predictions) > 0:
            # 方向预测统计
            direction_stats = predictions['direction'].value_counts()
            report += f"   方向预测分布:\n"
            report += f"     看涨预测: {direction_stats.get(1, 0)} 次\n"
            report += f"     看跌预测: {direction_stats.get(0, 0)} 次\n"
            
            # 幅度预测统计
            magnitude_mean = predictions['magnitude'].mean()
            magnitude_std = predictions['magnitude'].std()
            report += f"   幅度预测统计:\n"
            report += f"     平均预测幅度: {magnitude_mean:.2f}%\n"
            report += f"     幅度预测波动: {magnitude_std:.2f}%\n"
            
            # 波动率预测统计
            volatility_mean = predictions['volatility'].mean()
            volatility_std = predictions['volatility'].std()
            report += f"   波动率预测统计:\n"
            report += f"     平均预测波动率: {volatility_mean:.2f}%\n"
            report += f"     波动率预测变异: {volatility_std:.2f}%\n"
        
        # 添加金融评估报告
        report += "\n" + self.evaluator.generate_report(metrics)
        
        return report
    
    def plot_backtest_results(self, result: Dict[str, Any]) -> None:
        """绘制回测结果图表
        
        Args:
            result: 回测结果
        """
        try:
            predictions = result['predictions']
            actual_prices = result['actual_prices']
            
            if len(predictions) == 0 or len(actual_prices) == 0:
                print("❌ 无数据可绘制")
                return
                
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'回测结果 - {result["stock_code"]}', fontsize=16)
            
            # 1. 价格走势与预测信号
            ax1 = axes[0, 0]
            dates = pd.to_datetime(actual_prices['date'])
            prices = actual_prices['close']
            
            ax1.plot(dates, prices, label='实际价格', linewidth=1.5)
            
            # 标记预测信号
            buy_signals = predictions[predictions['direction'] == 1]
            sell_signals = predictions[predictions['direction'] == 0]
            
            if len(buy_signals) > 0:
                buy_dates = pd.to_datetime(buy_signals['date'])
                buy_prices = [actual_prices[actual_prices['date'] == date]['close'].values[0] 
                             for date in buy_signals['date'] if date in actual_prices['date'].values]
                ax1.scatter(buy_dates[:len(buy_prices)], buy_prices, color='green', marker='^', s=50, label='买入信号')
            
            if len(sell_signals) > 0:
                sell_dates = pd.to_datetime(sell_signals['date'])
                sell_prices = [actual_prices[actual_prices['date'] == date]['close'].values[0] 
                              for date in sell_signals['date'] if date in actual_prices['date'].values]
                ax1.scatter(sell_dates[:len(sell_prices)], sell_prices, color='red', marker='v', s=50, label='卖出信号')
            
            ax1.set_title('价格走势与交易信号')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('价格')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 预测准确率分布
            ax2 = axes[0, 1]
            confidence_bins = np.arange(0.5, 1.05, 0.1)
            ax2.hist(predictions['direction_prob'], bins=confidence_bins, alpha=0.7, edgecolor='black')
            ax2.set_title('预测置信度分布')
            ax2.set_xlabel('置信度')
            ax2.set_ylabel('频次')
            ax2.grid(True, alpha=0.3)
            
            # 3. 预测幅度分布
            ax3 = axes[1, 0]
            ax3.hist(predictions['magnitude'], bins=20, alpha=0.7, edgecolor='black')
            ax3.set_title('预测涨跌幅度分布')
            ax3.set_xlabel('预测涨跌幅 (%)')
            ax3.set_ylabel('频次')
            ax3.grid(True, alpha=0.3)
            
            # 4. 预测波动率分布
            ax4 = axes[1, 1]
            ax4.hist(predictions['volatility'], bins=20, alpha=0.7, edgecolor='black')
            ax4.set_title('预测波动率分布')
            ax4.set_xlabel('预测波动率 (%)')
            ax4.set_ylabel('频次')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ 绘图失败: {str(e)}")

def demo_backtest():
    """回测演示函数"""
    print("🚀 启动多维度预测回测演示")
    
    # 创建回测引擎
    engine = BacktestEngine(initial_capital=100000)
    
    try:
        # 运行回测
        result = engine.run_backtest(
            stock_code="000001.SZ",
            start_date="2024-08-01",
            end_date="2024-12-01",
            train_window=100,
            rebalance_freq=15
        )
        
        # 生成报告
        report = engine.generate_backtest_report(result)
        print(report)
        
        # 绘制结果
        engine.plot_backtest_results(result)
        
        return result
        
    except Exception as e:
        print(f"❌ 回测演示失败: {str(e)}")
        return None

if __name__ == "__main__":
    demo_backtest()
