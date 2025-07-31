import sqlite3
import datetime as dt
import os
import json
from typing import List, Dict, Any, Optional

class PredictionManager:
    """预测结果管理器，负责保存和查询预测记录"""
    
    def __init__(self, db_path: str = None):
        """初始化预测管理器
        
        Args:
            db_path: 数据库路径，默认为 yc 目录下的 cache.db
        """
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache.db")
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_database()
    
    def _init_database(self):
        """初始化预测记录表"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                name TEXT,
                prediction_date TEXT NOT NULL,
                target_date TEXT NOT NULL,
                signal INTEGER NOT NULL,
                accuracy REAL,
                votes TEXT,
                window_sizes TEXT,
                trials INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                actual_result INTEGER,
                actual_accuracy REAL
            )
        """)
        
        # 创建索引以提高查询性能
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_code 
            ON predictions(code)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_date 
            ON predictions(prediction_date)
        """)
        
        self.conn.commit()
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """保存预测结果
        
        Args:
            prediction_data: 预测数据字典，包含：
                - code: 股票代码
                - name: 股票名称
                - prediction_date: 预测日期
                - target_date: 目标日期
                - signal: 预测信号 (1=上涨, 0=下跌)
                - accuracy: 预测准确率
                - votes: 投票结果列表
                - window_sizes: 窗口大小列表
                - trials: 调参次数
                
        Returns:
            int: 插入记录的ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO predictions (
                code, name, prediction_date, target_date, signal,
                accuracy, votes, window_sizes, trials
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_data["code"],
            prediction_data["name"],
            prediction_data["prediction_date"],
            prediction_data["target_date"],
            prediction_data["signal"],
            prediction_data["accuracy"],
            prediction_data["votes"],
            prediction_data["window_sizes"],
            prediction_data["trials"]
        ))
        
        # 如果插入被忽略（已存在），则更新现有记录
        if cursor.rowcount == 0:
            cursor.execute("""
                UPDATE predictions
                SET name = ?, signal = ?, accuracy = ?, votes = ?,
                    window_sizes = ?, trials = ?, created_at = CURRENT_TIMESTAMP
                WHERE code = ? AND prediction_date = ? AND target_date = ?
            """, (
                prediction_data["name"],
                prediction_data["signal"],
                prediction_data["accuracy"],
                prediction_data["votes"],
                prediction_data["window_sizes"],
                prediction_data["trials"],
                prediction_data["code"],
                prediction_data["prediction_date"],
                prediction_data["target_date"]
            ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_predictions_by_code(self, code: str, limit: int = 50) -> List[Dict[str, Any]]:
        """根据股票代码获取预测记录
        
        Args:
            code: 股票代码
            limit: 返回记录数量限制
            
        Returns:
            List[Dict]: 预测记录列表
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE code = ? 
            ORDER BY prediction_date DESC 
            LIMIT ?
        """, (code, limit))
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            # 转换 JSON 字符串
            try:
                record["votes"] = json.loads(record["votes"]) if record["votes"] else []
            except:
                record["votes"] = []
                
            try:
                record["window_sizes"] = json.loads(record["window_sizes"]) if record["window_sizes"] else []
            except:
                record["window_sizes"] = []
                
            results.append(record)
        
        return results
    
    def get_all_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取所有预测记录
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            List[Dict]: 预测记录列表
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY prediction_date DESC, created_at DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            # 转换 JSON 字符串
            try:
                record["votes"] = json.loads(record["votes"]) if record["votes"] else []
            except:
                record["votes"] = []
                
            try:
                record["window_sizes"] = json.loads(record["window_sizes"]) if record["window_sizes"] else []
            except:
                record["window_sizes"] = []
                
            results.append(record)
        
        return results
    
    def get_predictions_by_date_range(self, start_date: str, end_date: str, 
                                    code: str = None) -> List[Dict[str, Any]]:
        """根据日期范围获取预测记录
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            code: 股票代码，可选
            
        Returns:
            List[Dict]: 预测记录列表
        """
        cursor = self.conn.cursor()
        
        if code:
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE code = ? AND prediction_date BETWEEN ? AND ?
                ORDER BY prediction_date DESC
            """, (code, start_date, end_date))
        else:
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE prediction_date BETWEEN ? AND ?
                ORDER BY prediction_date DESC, code
            """, (start_date, end_date))
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            # 转换 JSON 字符串
            try:
                record["votes"] = json.loads(record["votes"]) if record["votes"] else []
            except:
                record["votes"] = []
                
            try:
                record["window_sizes"] = json.loads(record["window_sizes"]) if record["window_sizes"] else []
            except:
                record["window_sizes"] = []
                
            results.append(record)
        
        return results
    
    def update_actual_result(self, prediction_id: int, actual_result: int, 
                           actual_accuracy: float = None):
        """更新实际结果
        
        Args:
            prediction_id: 预测记录ID
            actual_result: 实际结果 (1=上涨, 0=下跌)
            actual_accuracy: 实际准确率
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE predictions 
            SET actual_result = ?, actual_accuracy = ?
            WHERE id = ?
        """, (actual_result, actual_accuracy, prediction_id))
        
        self.conn.commit()
    
    def get_prediction_statistics(self, code: str = None) -> Dict[str, Any]:
        """获取预测统计信息
        
        Args:
            code: 股票代码，可选
            
        Returns:
            Dict: 统计信息
        """
        cursor = self.conn.cursor()
        
        if code:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(accuracy) as avg_accuracy,
                    SUM(CASE WHEN signal = 1 THEN 1 ELSE 0 END) as up_predictions,
                    SUM(CASE WHEN signal = 0 THEN 1 ELSE 0 END) as down_predictions,
                    SUM(CASE WHEN actual_result IS NOT NULL THEN 1 ELSE 0 END) as has_actual,
                    AVG(CASE WHEN actual_result = signal THEN 1 ELSE 0 END) as actual_accuracy
                FROM predictions
                WHERE code = ?
            """, (code,))
        else:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(accuracy) as avg_accuracy,
                    SUM(CASE WHEN signal = 1 THEN 1 ELSE 0 END) as up_predictions,
                    SUM(CASE WHEN signal = 0 THEN 1 ELSE 0 END) as down_predictions,
                    SUM(CASE WHEN actual_result IS NOT NULL THEN 1 ELSE 0 END) as has_actual,
                    AVG(CASE WHEN actual_result = signal THEN 1 ELSE 0 END) as actual_accuracy
                FROM predictions
            """)
        
        result = cursor.fetchone()
        if result:
            return {
                "total_predictions": result[0] or 0,
                "avg_accuracy": result[1] or 0,
                "up_predictions": result[2] or 0,
                "down_predictions": result[3] or 0,
                "predictions_with_actual": result[4] or 0,
                "actual_accuracy": result[5] or 0
            }
        
        return {
            "total_predictions": 0,
            "avg_accuracy": 0,
            "up_predictions": 0,
            "down_predictions": 0,
            "predictions_with_actual": 0,
            "actual_accuracy": 0
        }

    def get_history(self, code: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取历史预测记录
        
        Args:
            code: 股票代码，可选
            limit: 返回记录数量限制
            
        Returns:
            List[Dict]: 历史预测记录列表
        """
        return self.get_all_predictions(limit) if code is None else self.get_predictions_by_code(code, limit)
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()
    
    def __del__(self):
        """析构函数，确保关闭连接"""
        try:
            self.close()
        except:
            pass