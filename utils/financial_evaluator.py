"""
金融评估指标模块
提供完整的金融量化评估体系
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FinancialMetrics:
    """金融评估指标结果"""
    # 基础指标
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    
    # 风险指标
    max_drawdown: float
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR
    
    # 交易指标
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    
    # 其他指标
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float

class FinancialEvaluator:
    """金融评估器"""
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Args:
            risk_free_rate: 无风险收益率（年化）
        """
        self.risk_free_rate = risk_free_rate
        
    def evaluate_predictions(self, 
                           predictions: pd.DataFrame,
                           actual_prices: pd.DataFrame,
                           initial_capital: float = 100000) -> FinancialMetrics:
        """评估预测结果的金融表现
        
        Args:
            predictions: 预测结果DataFrame，包含direction, magnitude, volatility
            actual_prices: 实际价格数据
            initial_capital: 初始资金
            
        Returns:
            FinancialMetrics: 金融评估结果
        """
        # 生成交易信号
        trades = self._generate_trades(predictions, actual_prices)
        
        # 计算收益序列
        returns = self._calculate_returns(trades, initial_capital)
        
        # 计算各项指标
        metrics = self._calculate_all_metrics(returns, trades)
        
        return metrics
    
    def _generate_trades(self, predictions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """根据预测生成交易记录
        
        Args:
            predictions: 预测结果
            prices: 价格数据
            
        Returns:
            交易记录DataFrame
        """
        trades = []
        
        for i in range(len(predictions)):
            pred = predictions.iloc[i]
            
            # 获取当前和下一日价格
            if i >= len(prices) - 1:
                break
                
            current_price = prices['close'].iloc[i]
            next_price = prices['close'].iloc[i + 1]
            
            # 根据预测方向生成交易信号
            direction = pred.get('direction', 0)
            confidence = pred.get('direction_prob', 0.5)
            predicted_magnitude = pred.get('magnitude', 0)
            predicted_volatility = pred.get('volatility', 1.0)
            
            # 计算实际收益
            actual_return = (next_price - current_price) / current_price
            
            # 交易决策（仅在置信度>0.6时交易）
            if confidence > 0.6:
                if direction == 1:  # 预测上涨，做多
                    position = 1
                    trade_return = actual_return
                elif direction == 0:  # 预测下跌，做空（简化处理）
                    position = -1
                    trade_return = -actual_return
                else:
                    position = 0
                    trade_return = 0
            else:
                position = 0
                trade_return = 0
            
            trades.append({
                'date': prices.index[i] if hasattr(prices.index, 'date') else i,
                'position': position,
                'entry_price': current_price,
                'exit_price': next_price,
                'predicted_direction': direction,
                'confidence': confidence,
                'predicted_magnitude': predicted_magnitude,
                'predicted_volatility': predicted_volatility,
                'actual_return': actual_return,
                'trade_return': trade_return,
                'is_correct': (direction == 1 and actual_return > 0) or (direction == 0 and actual_return < 0)
            })
        
        return pd.DataFrame(trades)
    
    def _calculate_returns(self, trades: pd.DataFrame, initial_capital: float) -> pd.Series:
        """计算累积收益序列
        
        Args:
            trades: 交易记录
            initial_capital: 初始资金
            
        Returns:
            累积收益序列
        """
        if len(trades) == 0:
            return pd.Series([initial_capital])
        
        # 计算每日收益
        daily_returns = trades['trade_return'].fillna(0)
        
        # 计算累积收益
        cumulative_returns = (1 + daily_returns).cumprod()
        portfolio_values = initial_capital * cumulative_returns
        
        return portfolio_values
    
    def _calculate_all_metrics(self, returns: pd.Series, trades: pd.DataFrame) -> FinancialMetrics:
        """计算所有金融指标
        
        Args:
            returns: 收益序列
            trades: 交易记录
            
        Returns:
            FinancialMetrics对象
        """
        if len(returns) <= 1:
            return FinancialMetrics(
                total_return=0, annual_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, var_95=0, cvar_95=0,
                win_rate=0, profit_loss_ratio=0, total_trades=0,
                calmar_ratio=0, sortino_ratio=0, information_ratio=0
            )
        
        # 基础收益指标
        initial_value = returns.iloc[0]
        final_value = returns.iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 年化收益率（假设252个交易日）
        periods = len(returns)
        annual_return = (final_value / initial_value) ** (252 / periods) - 1
        
        # 收益率序列
        daily_returns = returns.pct_change().dropna()
        
        # 波动率（年化）
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 夏普比率
        excess_returns = daily_returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # 最大回撤
        rolling_max = returns.expanding().max()
        drawdowns = (returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # VaR和CVaR (95%置信度)
        var_95 = float(np.percentile(daily_returns, 5))
        cvar_95 = float(daily_returns[daily_returns <= var_95].mean())
        
        # 交易相关指标
        if len(trades) > 0:
            active_trades = trades[trades['position'] != 0]
            
            if len(active_trades) > 0:
                win_rate = active_trades['is_correct'].mean()
                
                winning_trades = active_trades[active_trades['trade_return'] > 0]['trade_return']
                losing_trades = active_trades[active_trades['trade_return'] < 0]['trade_return']
                
                avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
                avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 1
                
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                total_trades = len(active_trades)
            else:
                win_rate = 0
                profit_loss_ratio = 0
                total_trades = 0
        else:
            win_rate = 0
            profit_loss_ratio = 0
            total_trades = 0
        
        # 卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 索提诺比率
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # 信息比率（相对于基准的超额收益）
        # 这里简化处理，使用相对于无风险收益率
        information_ratio = sharpe_ratio  # 简化版本
        
        return FinancialMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            total_trades=total_trades,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            information_ratio=information_ratio
        )
    
    def compare_strategies(self, 
                          strategies: Dict[str, pd.DataFrame],
                          prices: pd.DataFrame) -> pd.DataFrame:
        """比较多个策略的表现
        
        Args:
            strategies: 策略预测结果字典
            prices: 价格数据
            
        Returns:
            策略比较结果DataFrame
        """
        comparison_results = []
        
        for strategy_name, predictions in strategies.items():
            metrics = self.evaluate_predictions(predictions, prices)
            
            result = {
                'Strategy': strategy_name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annual_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.3f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Profit/Loss Ratio': f"{metrics.profit_loss_ratio:.2f}",
                'Total Trades': metrics.total_trades,
                'Calmar Ratio': f"{metrics.calmar_ratio:.3f}",
                'Sortino Ratio': f"{metrics.sortino_ratio:.3f}"
            }
            
            comparison_results.append(result)
        
        return pd.DataFrame(comparison_results)
    
    def generate_report(self, metrics: FinancialMetrics) -> str:
        """生成详细的评估报告
        
        Args:
            metrics: 金融指标
            
        Returns:
            格式化的报告字符串
        """
        report = f"""
📊 金融量化评估报告
{'='*50}

💰 收益指标:
   总收益率: {metrics.total_return:.2%}
   年化收益率: {metrics.annual_return:.2%}
   年化波动率: {metrics.volatility:.2%}

📈 风险调整收益:
   夏普比率: {metrics.sharpe_ratio:.3f}
   索提诺比率: {metrics.sortino_ratio:.3f}
   卡尔玛比率: {metrics.calmar_ratio:.3f}
   信息比率: {metrics.information_ratio:.3f}

⚠️ 风险指标:
   最大回撤: {metrics.max_drawdown:.2%}
   95% VaR: {metrics.var_95:.2%}
   95% CVaR: {metrics.cvar_95:.2%}

🎯 交易表现:
   胜率: {metrics.win_rate:.2%}
   盈亏比: {metrics.profit_loss_ratio:.2f}
   总交易次数: {metrics.total_trades}

📝 评估结论:
"""
        
        # 添加评估结论
        if metrics.sharpe_ratio > 1.5:
            report += "   ✅ 优秀的风险调整收益表现\n"
        elif metrics.sharpe_ratio > 1.0:
            report += "   ✅ 良好的风险调整收益表现\n"
        elif metrics.sharpe_ratio > 0.5:
            report += "   ⚠️ 一般的风险调整收益表现\n"
        else:
            report += "   ❌ 较差的风险调整收益表现\n"
            
        if metrics.max_drawdown > -0.2:
            report += "   ✅ 回撤控制良好\n"
        elif metrics.max_drawdown > -0.3:
            report += "   ⚠️ 回撤控制一般\n"
        else:
            report += "   ❌ 回撤较大，需要改进风控\n"
            
        if metrics.win_rate > 0.6:
            report += "   ✅ 预测准确率较高\n"
        elif metrics.win_rate > 0.5:
            report += "   ⚠️ 预测准确率一般\n"
        else:
            report += "   ❌ 预测准确率偏低\n"
        
        return report
