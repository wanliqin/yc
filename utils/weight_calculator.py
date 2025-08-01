import numpy as np
import pandas as pd
from typing import Tuple

class DynamicWeightCalculator:
    """动态样本权重计算器"""
    
    def __init__(self):
        self.min_weight = 0.1
        self.max_weight = 5.0
        self.volatility_threshold = 0.02  # 2%波动率阈值
        
    def calculate_time_decay_weights(self, n_samples: int, decay_factor: float = 0.95) -> np.ndarray:
        """计算时间衰减权重
        
        Args:
            n_samples: 样本数量
            decay_factor: 衰减因子，越接近1衰减越慢
            
        Returns:
            权重数组，最新的样本权重最大
        """
        weights = np.array([decay_factor ** (n_samples - 1 - i) for i in range(n_samples)])
        # 归一化到合理范围
        weights = weights / weights.mean()
        return np.clip(weights, self.min_weight, self.max_weight)
    
    def calculate_volatility_weights(self, prices: pd.Series, window: int = 20) -> np.ndarray:
        """基于市场波动率计算权重
        
        Args:
            prices: 价格序列
            window: 计算波动率的窗口期
            
        Returns:
            权重数组，高波动期权重更高
        """
        returns = prices.pct_change().fillna(0)
        volatility = returns.rolling(window=window, min_periods=5).std().fillna(returns.std())
        
        # 波动率越高，权重越大（但有上限）
        normalized_vol = volatility / volatility.mean()
        weights = 1 + (normalized_vol - 1) * 0.5  # 减缓波动率影响
        
        return np.clip(np.array(weights), self.min_weight, self.max_weight)
    
    def calculate_volume_weights(self, volumes: pd.Series, window: int = 20) -> np.ndarray:
        """基于成交量计算权重
        
        Args:
            volumes: 成交量序列
            window: 计算平均成交量的窗口期
            
        Returns:
            权重数组，高成交量期权重更高
        """
        vol_ma = volumes.rolling(window=window, min_periods=5).mean().fillna(volumes.mean())
        vol_ratio = volumes / vol_ma
        
        # 成交量比率转换为权重
        weights = 1 + np.log1p(vol_ratio - 1) * 0.3  # 使用对数函数平滑
        
        return np.clip(np.array(weights), self.min_weight, self.max_weight)
    
    def calculate_trend_weights(self, prices: pd.Series, short_window: int = 5, long_window: int = 20) -> np.ndarray:
        """基于趋势强度计算权重
        
        Args:
            prices: 价格序列
            short_window: 短期均线窗口
            long_window: 长期均线窗口
            
        Returns:
            权重数组，趋势明确期权重更高
        """
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        
        # 计算趋势强度
        trend_strength = np.abs((short_ma - long_ma) / long_ma)
        trend_strength = pd.Series(trend_strength).fillna(0)
        
        # 趋势强度转换为权重
        weights = 1 + trend_strength * 2
        
        return np.clip(np.array(weights), self.min_weight, self.max_weight)
    
    def calculate_market_regime_weights(self, returns: pd.Series, window: int = 30) -> np.ndarray:
        """基于市场状态计算权重
        
        Args:
            returns: 收益率序列
            window: 市场状态检测窗口
            
        Returns:
            权重数组，稳定期权重更高
        """
        # 计算滚动夏普比率作为市场状态指标
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        # 避免除零
        rolling_std = rolling_std.replace(0, rolling_std.mean())
        sharpe_ratio = rolling_mean / rolling_std
        
        # 夏普比率越稳定，权重越高
        sharpe_ratio_filled = pd.Series(sharpe_ratio).fillna(0)
        sharpe_stability = 1 / (1 + np.abs(sharpe_ratio_filled))
        weights = 1 + sharpe_stability * 0.5
        
        return np.clip(np.array(weights), self.min_weight, self.max_weight)
    
    def calculate_combined_weights(self, data: pd.DataFrame, 
                                 time_weight: float = 0.4,
                                 volatility_weight: float = 0.25,
                                 volume_weight: float = 0.15,
                                 trend_weight: float = 0.1,
                                 regime_weight: float = 0.1) -> np.ndarray:
        """计算综合权重
        
        Args:
            data: 包含价格和成交量的数据框，需要包含'close', 'vol'列
            time_weight: 时间衰减权重的权重
            volatility_weight: 波动率权重的权重
            volume_weight: 成交量权重的权重
            trend_weight: 趋势权重的权重
            regime_weight: 市场状态权重的权重
            
        Returns:
            综合权重数组
        """
        n_samples = len(data)
        
        if n_samples < 5:
            return np.ones(n_samples)
        
        # 计算各种权重
        time_weights = self.calculate_time_decay_weights(n_samples)
        
        if 'close' in data.columns:
            volatility_weights = self.calculate_volatility_weights(data['close'])
            trend_weights = self.calculate_trend_weights(data['close'])
            
            returns = data['close'].pct_change().fillna(0)
            regime_weights = self.calculate_market_regime_weights(returns)
        else:
            volatility_weights = np.ones(n_samples)
            trend_weights = np.ones(n_samples)
            regime_weights = np.ones(n_samples)
            
        if 'vol' in data.columns:
            volume_weights = self.calculate_volume_weights(data['vol'])
        else:
            volume_weights = np.ones(n_samples)
        
        # 综合权重计算
        combined_weights = (
            time_weights * time_weight +
            volatility_weights * volatility_weight +
            volume_weights * volume_weight +
            trend_weights * trend_weight +
            regime_weights * regime_weight
        )
        
        # 最终归一化和裁剪
        combined_weights = combined_weights / combined_weights.mean()
        return np.clip(combined_weights, self.min_weight, self.max_weight)
    
    def get_adaptive_weights(self, data: pd.DataFrame, 
                           market_state: str = 'normal') -> np.ndarray:
        """根据市场状态自适应调整权重参数
        
        Args:
            data: 价格数据
            market_state: 市场状态 ('bull', 'bear', 'normal', 'volatile')
            
        Returns:
            自适应权重数组
        """
        if market_state == 'bull':
            # 牛市：更重视趋势和成交量
            return self.calculate_combined_weights(
                data, time_weight=0.3, volatility_weight=0.1, 
                volume_weight=0.3, trend_weight=0.25, regime_weight=0.05
            )
        elif market_state == 'bear':
            # 熊市：更重视波动率和时间衰减
            return self.calculate_combined_weights(
                data, time_weight=0.5, volatility_weight=0.3, 
                volume_weight=0.1, trend_weight=0.05, regime_weight=0.05
            )
        elif market_state == 'volatile':
            # 震荡市：平衡各因子，重视市场状态
            return self.calculate_combined_weights(
                data, time_weight=0.25, volatility_weight=0.25, 
                volume_weight=0.2, trend_weight=0.1, regime_weight=0.2
            )
        else:
            # 正常市场：默认配置
            return self.calculate_combined_weights(data)
