"""
多维度预测器 - 扩展预测能力
支持涨跌方向、涨跌幅度、波动率预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from models.stock_predictor import StockPredictor
import optuna

class MultiDimensionalPredictor(StockPredictor):
    """多维度股票预测器"""
    
    def __init__(self, feature_columns: List[str]):
        super().__init__(feature_columns)
        self.prediction_types = ['direction', 'magnitude', 'volatility']
        self.models = {}
        
    def prepare_labels(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """准备多维度标签
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            Dict包含三种预测标签：
            - direction: 涨跌方向 (0/1)
            - magnitude: 涨跌幅度 (%)
            - volatility: 未来波动率
        """
        labels = {}
        
        # 1. 涨跌方向标签 (原有逻辑)
        labels['direction'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # 2. 涨跌幅度标签 (连续值)
        labels['magnitude'] = ((df['close'].shift(-1) - df['close']) / df['close'] * 100)
        
        # 3. 波动率标签 (未来5日标准差)
        future_returns = df['close'].pct_change().shift(-5).rolling(window=5).std() * 100
        labels['volatility'] = future_returns
        
        # 移除包含NaN的行
        for key in labels:
            labels[key] = labels[key].fillna(labels[key].mean())
            
        return labels
    
    def train_multi_models(self, X: pd.DataFrame, labels: Dict[str, pd.Series],
                          window_size: int = 30, trials: int = 20) -> Dict[str, Any]:
        """训练多维度模型
        
        Args:
            X: 特征数据
            labels: 多维度标签字典
            window_size: 时间窗口大小
            trials: 优化试验次数
            
        Returns:
            训练结果字典
        """
        results = {}
        
        for pred_type, y in labels.items():
            print(f"🔄 训练{pred_type}预测模型...")
            
            # 准备训练数据
            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index]
            
            if len(X_clean) < window_size + 2:
                print(f"❌ {pred_type}数据不足，跳过训练")
                continue
                
            # 根据预测类型选择模型
            if pred_type == 'direction':
                # 分类任务
                model, params, score = self._train_classification_model(
                    X_clean, y_clean, window_size, trials
                )
                metric_name = "准确率"
            else:
                # 回归任务
                model, params, score = self._train_regression_model(
                    X_clean, y_clean, window_size, trials
                )
                metric_name = "R²分数"
            
            results[pred_type] = {
                'model': model,
                'params': params,
                'score': score,
                'metric': metric_name,
                'data_size': len(X_clean)
            }
            
            print(f"✅ {pred_type}模型训练完成，{metric_name}: {score:.4f}")
            
        return results
    
    def _train_classification_model(self, X: pd.DataFrame, y: pd.Series,
                                  window_size: int, trials: int) -> Tuple[Any, Dict, float]:
        """训练分类模型（涨跌方向）"""
        
        def objective(trial):
            model_type = trial.suggest_categorical('model', ['lgb', 'xgb', 'rf'])
            
            if model_type == 'lgb':
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'verbose': -1
                }
                
                # 时序交叉验证
                scores = []
                for i in range(window_size, len(X) - 1, 10):
                    X_train = X.iloc[max(0, i-window_size):i]
                    y_train = y.iloc[max(0, i-window_size):i]
                    X_val = X.iloc[i:i+1]
                    y_val = y.iloc[i:i+1]
                    
                    if len(X_train) < 10 or len(X_val) == 0:
                        continue
                        
                    train_data = lgb.Dataset(X_train, label=y_train)
                    model = lgb.train(params, train_data, num_boost_round=100)
                    pred = (model.predict(X_val) > 0.5).astype(int)
                    scores.append(accuracy_score(y_val, pred))
                
                return np.mean(scores) if scores else 0
                
            elif model_type == 'xgb':
                params = {
                    'objective': 'binary:logistic',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                
                scores = []
                for i in range(window_size, len(X) - 1, 10):
                    X_train = X.iloc[max(0, i-window_size):i]
                    y_train = y.iloc[max(0, i-window_size):i]
                    X_val = X.iloc[i:i+1]
                    y_val = y.iloc[i:i+1]
                    
                    if len(X_train) < 10 or len(X_val) == 0:
                        continue
                        
                    model = xgb.XGBClassifier(**params, random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    scores.append(accuracy_score(y_val, pred))
                
                return np.mean(scores) if scores else 0
                
            else:  # rf
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'random_state': 42
                }
                
                scores = []
                for i in range(window_size, len(X) - 1, 10):
                    X_train = X.iloc[max(0, i-window_size):i]
                    y_train = y.iloc[max(0, i-window_size):i]
                    X_val = X.iloc[i:i+1]
                    y_val = y.iloc[i:i+1]
                    
                    if len(X_train) < 10 or len(X_val) == 0:
                        continue
                        
                    model = RandomForestClassifier(**params)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    scores.append(accuracy_score(y_val, pred))
                
                return np.mean(scores) if scores else 0
        
        # 优化超参数
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        
        best_params = study.best_params
        model_type = best_params.pop('model')
        
        # 训练最终模型
        if model_type == 'lgb':
            best_params.update({
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbose': -1
            })
            train_data = lgb.Dataset(X, label=y)
            final_model = lgb.train(best_params, train_data, num_boost_round=100, verbose_eval=False)
        elif model_type == 'xgb':
            final_model = xgb.XGBClassifier(**best_params, random_state=42)
            final_model.fit(X, y)
        else:
            final_model = RandomForestClassifier(**best_params)
            final_model.fit(X, y)
            
        return final_model, best_params, study.best_value
    
    def _train_regression_model(self, X: pd.DataFrame, y: pd.Series,
                              window_size: int, trials: int) -> Tuple[Any, Dict, float]:
        """训练回归模型（幅度/波动率）"""
        
        def objective(trial):
            model_type = trial.suggest_categorical('model', ['lgb', 'xgb', 'rf'])
            
            if model_type == 'lgb':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'verbose': -1
                }
                
                scores = []
                for i in range(window_size, len(X) - 1, 10):
                    X_train = X.iloc[max(0, i-window_size):i]
                    y_train = y.iloc[max(0, i-window_size):i]
                    X_val = X.iloc[i:i+1]
                    y_val = y.iloc[i:i+1]
                    
                    if len(X_train) < 10 or len(X_val) == 0:
                        continue
                        
                    train_data = lgb.Dataset(X_train, label=y_train)
                    model = lgb.train(params, train_data, num_boost_round=100, verbose_eval=False)
                    pred = model.predict(X_val)
                    
                    # R²分数
                    ss_res = np.sum((y_val - pred) ** 2)
                    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    scores.append(r2)
                
                return np.mean(scores) if scores else 0
                
            elif model_type == 'xgb':
                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                }
                
                scores = []
                for i in range(window_size, len(X) - 1, 10):
                    X_train = X.iloc[max(0, i-window_size):i]
                    y_train = y.iloc[max(0, i-window_size):i]
                    X_val = X.iloc[i:i+1]
                    y_val = y.iloc[i:i+1]
                    
                    if len(X_train) < 10 or len(X_val) == 0:
                        continue
                        
                    model = xgb.XGBRegressor(**params, random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    
                    ss_res = np.sum((y_val - pred) ** 2)
                    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    scores.append(r2)
                
                return np.mean(scores) if scores else 0
                
            else:  # rf
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'random_state': 42
                }
                
                scores = []
                for i in range(window_size, len(X) - 1, 10):
                    X_train = X.iloc[max(0, i-window_size):i]
                    y_train = y.iloc[max(0, i-window_size):i]
                    X_val = X.iloc[i:i+1]
                    y_val = y.iloc[i:i+1]
                    
                    if len(X_train) < 10 or len(X_val) == 0:
                        continue
                        
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    
                    ss_res = np.sum((y_val - pred) ** 2)
                    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    scores.append(r2)
                
                return np.mean(scores) if scores else 0
        
        # 优化超参数
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        
        best_params = study.best_params
        model_type = best_params.pop('model')
        
        # 训练最终模型
        if model_type == 'lgb':
            best_params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1
            })
            train_data = lgb.Dataset(X, label=y)
            final_model = lgb.train(best_params, train_data, num_boost_round=100, verbose_eval=False)
        elif model_type == 'xgb':
            final_model = xgb.XGBRegressor(**best_params, random_state=42)
            final_model.fit(X, y)
        else:
            final_model = RandomForestRegressor(**best_params)
            final_model.fit(X, y)
            
        return final_model, best_params, study.best_value
    
    def predict_multi_dimensions(self, X: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """多维度预测
        
        Args:
            X: 特征数据
            models: 训练好的模型字典
            
        Returns:
            预测结果字典
        """
        predictions = {}
        
        for pred_type, model_info in models.items():
            model = model_info['model']
            
            if pred_type == 'direction':
                # 分类预测
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]
                    predictions[pred_type] = (pred_proba > 0.5).astype(int)
                    predictions[f'{pred_type}_prob'] = pred_proba
                else:  # LightGBM
                    pred_proba = model.predict(X)
                    predictions[pred_type] = (pred_proba > 0.5).astype(int)
                    predictions[f'{pred_type}_prob'] = pred_proba
            else:
                # 回归预测
                predictions[pred_type] = model.predict(X)
        
        return predictions
    
    def get_comprehensive_prediction(self, X: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """获取综合预测结果
        
        Args:
            X: 特征数据（单行）
            models: 训练好的模型字典
            
        Returns:
            综合预测结果
        """
        predictions = self.predict_multi_dimensions(X, models)
        
        # 构建综合结果
        result = {
            'direction': {
                'prediction': int(predictions['direction'][0]),
                'probability': float(predictions['direction_prob'][0]),
                'signal': '上涨' if predictions['direction'][0] == 1 else '下跌'
            },
            'magnitude': {
                'prediction': float(predictions['magnitude'][0]),
                'description': f"预期涨跌幅: {predictions['magnitude'][0]:.2f}%"
            },
            'volatility': {
                'prediction': float(predictions['volatility'][0]),
                'description': f"预期波动率: {predictions['volatility'][0]:.2f}%"
            }
        }
        
        # 风险等级评估
        volatility = predictions['volatility'][0]
        if volatility < 1.0:
            risk_level = "低风险"
        elif volatility < 2.5:
            risk_level = "中等风险"
        else:
            risk_level = "高风险"
            
        result['risk_assessment'] = {
            'level': risk_level,
            'volatility': volatility
        }
        
        return result
