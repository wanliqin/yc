"""
StockPredictor 动态权重策略集成补丁
用于在不影响现有代码的前提下集成新的权重策略
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import sys
import os

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.weight_calculator import DynamicWeightCalculator

class WeightedStockPredictor:
    """带动态权重的股票预测器 - 作为原StockPredictor的增强版本"""
    
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.weight_calculator = DynamicWeightCalculator()
        
    def calculate_advanced_sample_weights(self, 
                                        data: pd.DataFrame, 
                                        method: str = 'dynamic',
                                        **kwargs) -> np.ndarray:
        """计算高级样本权重
        
        Args:
            data: 包含价格和成交量数据的DataFrame
            method: 权重计算方法
                - 'simple': 原有简单权重策略
                - 'dynamic': 多因子动态权重
                - 'adaptive': 自适应权重（根据市场状态）
                - 'time_decay': 时间衰减权重
                - 'volatility': 基于波动率的权重
                - 'volume': 基于成交量的权重
            **kwargs: 额外参数
            
        Returns:
            样本权重数组
        """
        n_samples = len(data)
        
        if n_samples < 5:
            return np.ones(n_samples)
        
        if method == 'simple':
            # 原有的简单权重策略
            weights = np.ones(n_samples)
            if n_samples >= 30:
                weights[-30:] = 3
            return weights
            
        elif method == 'time_decay':
            # 时间衰减权重
            decay_factor = kwargs.get('decay_factor', 0.95)
            return self.weight_calculator.calculate_time_decay_weights(n_samples, decay_factor)
            
        elif method == 'volatility':
            # 基于波动率的权重
            if 'close' in data.columns:
                return self.weight_calculator.calculate_volatility_weights(data['close'])
            else:
                return np.ones(n_samples)
                
        elif method == 'volume':
            # 基于成交量的权重
            if 'vol' in data.columns:
                return self.weight_calculator.calculate_volume_weights(data['vol'])
            else:
                return np.ones(n_samples)
                
        elif method == 'dynamic':
            # 多因子动态权重
            return self.weight_calculator.calculate_combined_weights(data)
            
        elif method == 'adaptive':
            # 自适应权重
            market_state = kwargs.get('market_state', 'normal')
            return self.weight_calculator.get_adaptive_weights(data, market_state)
        
        else:
            return np.ones(n_samples)
    
    def detect_market_state(self, data: pd.DataFrame, window: int = 30) -> str:
        """检测市场状态
        
        Args:
            data: 价格数据
            window: 检测窗口期
            
        Returns:
            市场状态: 'bull', 'bear', 'volatile', 'normal'
        """
        if len(data) < window or 'close' not in data.columns:
            return 'normal'
        
        recent_data = data.tail(window)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return 'normal'
        
        mean_return = returns.mean()
        volatility = returns.std()
        
        # 状态判断阈值
        high_vol_threshold = 0.03
        bull_threshold = 0.008
        bear_threshold = -0.005
        
        if volatility > high_vol_threshold:
            return 'volatile'
        elif mean_return > bull_threshold:
            return 'bull'
        elif mean_return < bear_threshold:
            return 'bear'
        else:
            return 'normal'
    
    def get_weight_strategy_for_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """根据数据特征推荐权重策略
        
        Args:
            data: 输入数据
            
        Returns:
            权重策略配置
        """
        n_samples = len(data)
        market_state = self.detect_market_state(data)
        
        # 根据数据量和市场状态选择策略
        if n_samples < 50:
            # 数据量少，使用简单策略
            return {
                'method': 'time_decay',
                'params': {'decay_factor': 0.9},
                'reason': '数据量较少，使用时间衰减权重'
            }
        elif market_state == 'volatile':
            # 高波动期，使用波动率权重
            return {
                'method': 'adaptive',
                'params': {'market_state': 'volatile'},
                'reason': '市场高波动期，使用自适应权重'
            }
        elif market_state in ['bull', 'bear']:
            # 明确趋势期，使用趋势适配权重
            return {
                'method': 'adaptive', 
                'params': {'market_state': market_state},
                'reason': f'{market_state}市场，使用自适应权重'
            }
        else:
            # 正常市场，使用动态权重
            return {
                'method': 'dynamic',
                'params': {},
                'reason': '正常市场，使用多因子动态权重'
            }
    
    def compare_weight_strategies(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """比较不同权重策略的效果
        
        Args:
            data: 输入数据
            
        Returns:
            各策略的权重数组
        """
        strategies = ['simple', 'time_decay', 'dynamic', 'adaptive']
        results = {}
        
        market_state = self.detect_market_state(data)
        
        for strategy in strategies:
            if strategy == 'adaptive':
                weights = self.calculate_advanced_sample_weights(
                    data, strategy, market_state=market_state
                )
            else:
                weights = self.calculate_advanced_sample_weights(data, strategy)
            
            results[strategy] = weights
        
        return results

# 使用示例函数
def demo_weight_strategies():
    """演示权重策略的使用"""
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    volumes = np.random.lognormal(10, 0.3, 100)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'vol': volumes
    })
    
    # 初始化权重预测器
    predictor = WeightedStockPredictor(['close', 'vol'])
    
    # 获取推荐策略
    recommended = predictor.get_weight_strategy_for_data(data)
    print(f"推荐策略: {recommended['method']}")
    print(f"推荐理由: {recommended['reason']}")
    
    # 比较不同策略
    strategies = predictor.compare_weight_strategies(data)
    
    print("\n策略比较:")
    for name, weights in strategies.items():
        print(f"{name}: 权重范围[{weights.min():.3f}, {weights.max():.3f}], "
              f"平均{weights.mean():.3f}, 最近10天平均{weights[-10:].mean():.3f}")

if __name__ == "__main__":
    demo_weight_strategies()
