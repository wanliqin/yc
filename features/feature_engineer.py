import pandas as pd
import ta
import numpy as np

class FeatureEngineer:
    """特征工程类，用于生成技术指标和标签"""
    
    def __init__(self):
        self.feature_columns = [
            "rsi", "k", "d", "j", "bbp", "macd", "macd_signal", "ret"
        ]
    
    def add_technical_indicators(self, df):
        """添加技术指标
        
        Args:
            df: 包含基础价格数据的 DataFrame
            
        Returns:
            DataFrame: 添加了技术指标的 DataFrame
        """
        df = df.copy()
        
        # 确保数据按日期排序
        df = df.sort_values('date')
        
        # 数据清洗：确保数值类型正确
        numeric_cols = ['open', 'high', 'low', 'close', 'vol']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除无效数据
        df = df.dropna(subset=numeric_cols)
        
        # 确保没有无限值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # 如果数据不足，返回空DataFrame
        if len(df) < 20:
            return pd.DataFrame()
        
        # 收益率
        df["ret"] = df["close"].pct_change()
        
        # RSI 相对强弱指标
        try:
            df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        except:
            df["rsi"] = 50  # 默认值
        
        # KDJ 随机指标
        try:
            low9 = df["low"].rolling(9).min()
            high9 = df["high"].rolling(9).max()
            rsv = (df["close"] - low9) / (high9 - low9) * 100
            rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(50)
            df["k"] = rsv.ewm(alpha=1/3).mean()
            df["d"] = df["k"].ewm(alpha=1/3).mean()
            df["j"] = 3 * df["k"] - 2 * df["d"]
        except:
            df["k"] = 50
            df["d"] = 50
            df["j"] = 50
        
        # 布林带百分比
        try:
            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
            df["bbp"] = (df["close"] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
            df["bbp"] = df["bbp"].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        except:
            df["bbp"] = 0.5
        
        # MACD 指标
        try:
            macd = ta.trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
        except:
            df["macd"] = 0
            df["macd_signal"] = 0
        
        # 标签：次日是否上涨
        df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
        
        # 确保所有特征都是数值类型
        feature_cols = ["rsi", "k", "d", "j", "bbp", "macd", "macd_signal", "ret"]
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
    
    def prepare_features(self, df):
        """准备特征数据
        
        Args:
            df: 原始价格数据
            
        Returns:
            tuple: (X, y) 特征和标签
        """
        df_processed = self.add_technical_indicators(df)
        if len(df_processed) == 0:
            return pd.DataFrame(), pd.Series()
            
        # 确保包含所有需要的列
        required_cols = ["date"] + self.feature_columns + ["label"]
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            return pd.DataFrame(), pd.Series()
            
        # 分离特征和标签
        feature_cols = [col for col in df_processed.columns if col not in ["date", "label"]]
        X = df_processed[feature_cols].copy()
        y = df_processed["label"].copy()
        
        # 确保X包含我们定义的特征
        for feat in self.feature_columns:
            if feat not in X.columns:
                X[feat] = 0
        
        return X[self.feature_columns], y
    
    def get_feature_names(self):
        """获取特征名称列表
        
        Returns:
            list: 特征名称列表
        """
        return self.feature_columns.copy()
    
    def create_features(self, df, windows=None):
        """创建特征数据（兼容web接口）
        
        Args:
            df: 原始价格数据DataFrame
            windows: 窗口期列表（可选，用于兼容性）
            
        Returns:
            DataFrame: 包含特征和标签的DataFrame
        """
        if windows is None:
            windows = [120, 250]
            
        # 使用现有的特征工程流程
        df_processed = self.add_technical_indicators(df)
        
        if df_processed.empty:
            return pd.DataFrame()
            
        # 确保包含所有需要的列
        required_cols = ["date"] + self.feature_columns + ["label"]
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            return pd.DataFrame()
            
        # 重命名列以兼容web接口
        df_processed = df_processed.rename(columns={'date': 'trade_date'})
        
        return df_processed