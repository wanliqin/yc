import os
import pickle
import hashlib
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

class ModelCache:
    """模型缓存管理器"""
    
    def __init__(self, cache_dir: str = "model_cache", cache_db: str = "cache.db"):
        self.cache_dir = cache_dir
        self.cache_db = cache_db
        self.max_cache_age_days = 7  # 缓存有效期7天
        self.max_cache_size_mb = 500  # 最大缓存大小500MB
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化缓存数据库
        self._init_cache_db()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_cache_db(self):
        """初始化缓存数据库"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        # 创建模型缓存表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_cache (
                cache_key TEXT PRIMARY KEY,
                stock_code TEXT NOT NULL,
                model_type TEXT NOT NULL,
                feature_columns TEXT NOT NULL,
                window_size INTEGER NOT NULL,
                trials INTEGER NOT NULL,
                data_hash TEXT NOT NULL,
                model_file_path TEXT NOT NULL,
                best_params TEXT NOT NULL,
                validation_accuracy REAL NOT NULL,
                model_size_mb REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        """)
        
        # 创建缓存统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                cache_hits INTEGER DEFAULT 0,
                cache_misses INTEGER DEFAULT 0,
                cache_size_mb REAL DEFAULT 0,
                cleanup_runs INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _generate_cache_key(self, stock_code: str, feature_columns: list, 
                          window_size: int, trials: int, data_hash: str) -> str:
        """生成缓存键"""
        # 将所有参数组合成字符串
        key_data = f"{stock_code}_{sorted(feature_columns)}_{window_size}_{trials}_{data_hash}"
        
        # 生成MD5哈希
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """计算数据哈希，用于检测数据变化"""
        # 使用数据的形状、统计信息和最新几行数据生成哈希
        data_info = {
            'shape': X.shape,
            'columns': list(X.columns),
            'X_last_5_rows': X.tail(5).values.tolist() if len(X) >= 5 else X.values.tolist(),
            'y_last_5_values': y.tail(5).tolist() if len(y) >= 5 else y.tolist(),
            'X_mean': X.mean().to_dict(),
            'X_std': X.std().to_dict()
        }
        
        # 转换为JSON字符串并计算哈希
        data_str = json.dumps(data_info, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_model_file_path(self, cache_key: str) -> str:
        """获取模型文件路径"""
        return os.path.join(self.cache_dir, f"model_{cache_key}.pkl")
    
    def _save_model_to_file(self, model, cache_key: str) -> str:
        """保存模型到文件"""
        file_path = self._get_model_file_path(cache_key)
        
        # 保存模型
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        
        return file_path
    
    def _load_model_from_file(self, file_path: str):
        """从文件加载模型"""
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"加载模型文件失败: {e}")
            return None
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """获取文件大小（MB）"""
        if os.path.exists(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)
        return 0.0
    
    def cache_model(self, stock_code: str, feature_columns: list, window_size: int,
                   trials: int, X: pd.DataFrame, y: pd.Series, model, 
                   best_params: dict, validation_accuracy: float) -> str:
        """缓存模型
        
        Args:
            stock_code: 股票代码
            feature_columns: 特征列
            window_size: 窗口大小
            trials: 调参次数
            X: 特征数据
            y: 标签数据
            model: 训练好的模型
            best_params: 最佳参数
            validation_accuracy: 验证准确率
            
        Returns:
            缓存键
        """
        try:
            # 计算数据哈希
            data_hash = self._calculate_data_hash(X, y)
            
            # 生成缓存键
            cache_key = self._generate_cache_key(stock_code, feature_columns, 
                                               window_size, trials, data_hash)
            
            # 保存模型到文件
            model_file_path = self._save_model_to_file(model, cache_key)
            
            # 获取文件大小
            model_size_mb = self._get_file_size_mb(model_file_path)
            
            # 保存到数据库
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            # 检查是否已存在
            cursor.execute("SELECT cache_key FROM model_cache WHERE cache_key = ?", (cache_key,))
            if cursor.fetchone():
                # 更新现有记录
                cursor.execute("""
                    UPDATE model_cache 
                    SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE cache_key = ?
                """, (cache_key,))
            else:
                # 插入新记录
                cursor.execute("""
                    INSERT INTO model_cache (
                        cache_key, stock_code, model_type, feature_columns, 
                        window_size, trials, data_hash, model_file_path,
                        best_params, validation_accuracy, model_size_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key, stock_code, 'ensemble', json.dumps(feature_columns),
                    window_size, trials, data_hash, model_file_path,
                    json.dumps(best_params), validation_accuracy, model_size_mb
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"模型已缓存: {stock_code}, 缓存键: {cache_key}")
            return cache_key
            
        except Exception as e:
            self.logger.error(f"缓存模型失败: {e}")
            return ""
    
    def get_cached_model(self, stock_code: str, feature_columns: list, 
                        window_size: int, trials: int, X: pd.DataFrame, 
                        y: pd.Series) -> Optional[Tuple[Any, dict, float, str]]:
        """获取缓存的模型
        
        Args:
            stock_code: 股票代码
            feature_columns: 特征列
            window_size: 窗口大小
            trials: 调参次数
            X: 当前特征数据
            y: 当前标签数据
            
        Returns:
            (模型, 最佳参数, 验证准确率, 缓存键) 或 None
        """
        try:
            # 计算当前数据哈希
            current_data_hash = self._calculate_data_hash(X, y)
            
            # 生成查询键
            cache_key = self._generate_cache_key(stock_code, feature_columns,
                                               window_size, trials, current_data_hash)
            
            # 查询数据库
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_file_path, best_params, validation_accuracy, created_at
                FROM model_cache 
                WHERE cache_key = ? AND datetime(created_at) > datetime('now', '-{} days')
            """.format(self.max_cache_age_days), (cache_key,))
            
            result = cursor.fetchone()
            
            if result:
                model_file_path, best_params_str, validation_accuracy, created_at = result
                
                # 加载模型
                model = self._load_model_from_file(model_file_path)
                
                if model is not None:
                    # 更新访问记录
                    cursor.execute("""
                        UPDATE model_cache 
                        SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                        WHERE cache_key = ?
                    """, (cache_key,))
                    
                    conn.commit()
                    conn.close()
                    
                    # 更新缓存统计
                    self._update_cache_stats('hit')
                    
                    self.logger.info(f"缓存命中: {stock_code}, 缓存键: {cache_key}")
                    
                    return (
                        model, 
                        json.loads(best_params_str), 
                        validation_accuracy,
                        cache_key
                    )
            
            conn.close()
            
            # 缓存未命中
            self._update_cache_stats('miss')
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取缓存模型失败: {e}")
            return None
    
    def _update_cache_stats(self, event_type: str):
        """更新缓存统计"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 获取今日统计
        cursor.execute("SELECT * FROM cache_stats WHERE date = ?", (today,))
        result = cursor.fetchone()
        
        if result:
            # 更新现有记录
            if event_type == 'hit':
                cursor.execute("""
                    UPDATE cache_stats SET cache_hits = cache_hits + 1 WHERE date = ?
                """, (today,))
            elif event_type == 'miss':
                cursor.execute("""
                    UPDATE cache_stats SET cache_misses = cache_misses + 1 WHERE date = ?
                """, (today,))
        else:
            # 创建新记录
            hits = 1 if event_type == 'hit' else 0
            misses = 1 if event_type == 'miss' else 0
            cursor.execute("""
                INSERT INTO cache_stats (date, cache_hits, cache_misses)
                VALUES (?, ?, ?)
            """, (today, hits, misses))
        
        conn.commit()
        conn.close()
    
    def cleanup_cache(self):
        """清理过期和无用的缓存"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            # 删除过期的缓存记录
            cutoff_date = (datetime.now() - timedelta(days=self.max_cache_age_days)).strftime('%Y-%m-%d %H:%M:%S')
            
            # 获取要删除的文件路径
            cursor.execute("""
                SELECT model_file_path FROM model_cache 
                WHERE datetime(created_at) <= ?
            """, (cutoff_date,))
            
            expired_files = [row[0] for row in cursor.fetchall()]
            
            # 删除过期记录
            cursor.execute("""
                DELETE FROM model_cache WHERE datetime(created_at) <= ?
            """, (cutoff_date,))
            
            expired_count = cursor.rowcount
            
            # 删除文件
            deleted_files = 0
            for file_path in expired_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files += 1
                    except Exception as e:
                        self.logger.warning(f"删除文件失败: {file_path}, {e}")
            
            # 检查缓存大小限制
            cursor.execute("SELECT SUM(model_size_mb) FROM model_cache")
            total_size = cursor.fetchone()[0] or 0
            
            size_cleanup_count = 0
            if total_size > self.max_cache_size_mb:
                # 删除最少使用的缓存
                cursor.execute("""
                    SELECT cache_key, model_file_path FROM model_cache 
                    ORDER BY access_count ASC, accessed_at ASC
                """)
                
                for cache_key, file_path in cursor.fetchall():
                    if total_size <= self.max_cache_size_mb:
                        break
                    
                    # 删除记录
                    cursor.execute("DELETE FROM model_cache WHERE cache_key = ?", (cache_key,))
                    
                    # 删除文件
                    if os.path.exists(file_path):
                        file_size = self._get_file_size_mb(file_path)
                        try:
                            os.remove(file_path)
                            total_size -= file_size
                            size_cleanup_count += 1
                        except Exception as e:
                            self.logger.warning(f"删除文件失败: {file_path}, {e}")
            
            conn.commit()
            conn.close()
            
            # 更新清理统计
            self._update_cleanup_stats(expired_count + size_cleanup_count)
            
            self.logger.info(f"缓存清理完成: 过期删除{expired_count}个, 大小限制删除{size_cleanup_count}个")
            
        except Exception as e:
            self.logger.error(f"缓存清理失败: {e}")
    
    def _update_cleanup_stats(self, cleanup_count: int):
        """更新清理统计"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute("SELECT * FROM cache_stats WHERE date = ?", (today,))
        result = cursor.fetchone()
        
        if result:
            cursor.execute("""
                UPDATE cache_stats SET cleanup_runs = cleanup_runs + 1 WHERE date = ?
            """, (today,))
        else:
            cursor.execute("""
                INSERT INTO cache_stats (date, cleanup_runs) VALUES (?, 1)
            """, (today,))
        
        conn.commit()
        conn.close()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        # 获取总体统计
        cursor.execute("SELECT COUNT(*), SUM(model_size_mb), AVG(access_count) FROM model_cache")
        total_models, total_size_mb, avg_access = cursor.fetchone()
        
        # 获取最近7天统计
        cursor.execute("""
            SELECT SUM(cache_hits), SUM(cache_misses), SUM(cleanup_runs)
            FROM cache_stats 
            WHERE date >= date('now', '-7 days')
        """)
        recent_hits, recent_misses, recent_cleanups = cursor.fetchone()
        
        # 获取命中率
        total_requests = (recent_hits or 0) + (recent_misses or 0)
        hit_rate = (recent_hits / total_requests * 100) if total_requests > 0 else 0
        
        # 获取最活跃的缓存
        cursor.execute("""
            SELECT stock_code, access_count, datetime(accessed_at)
            FROM model_cache 
            ORDER BY access_count DESC, accessed_at DESC 
            LIMIT 5
        """)
        most_used = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_models': total_models or 0,
            'total_size_mb': round(total_size_mb or 0, 2),
            'avg_access_count': round(avg_access or 0, 2),
            'recent_hits': recent_hits or 0,
            'recent_misses': recent_misses or 0,
            'hit_rate_percent': round(hit_rate, 2),
            'recent_cleanups': recent_cleanups or 0,
            'most_used_models': [
                {
                    'stock_code': row[0],
                    'access_count': row[1],
                    'last_accessed': row[2]
                } for row in most_used
            ]
        }
    
    def invalidate_cache(self, stock_code: Optional[str] = None):
        """失效缓存
        
        Args:
            stock_code: 指定股票代码，None表示清空所有缓存
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        if stock_code:
            # 删除指定股票的缓存
            cursor.execute("SELECT model_file_path FROM model_cache WHERE stock_code = ?", (stock_code,))
            files_to_delete = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("DELETE FROM model_cache WHERE stock_code = ?", (stock_code,))
            deleted_count = cursor.rowcount
        else:
            # 删除所有缓存
            cursor.execute("SELECT model_file_path FROM model_cache")
            files_to_delete = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("DELETE FROM model_cache")
            deleted_count = cursor.rowcount
        
        # 删除文件
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.logger.warning(f"删除文件失败: {file_path}, {e}")
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"缓存失效完成: 删除{deleted_count}个模型缓存")

# 全局缓存实例
_model_cache = None

def get_model_cache() -> ModelCache:
    """获取全局模型缓存实例"""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache
