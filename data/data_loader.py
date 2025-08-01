import datetime as dt
import sqlite3
import pandas as pd
import tushare as ts
import os
import numpy as np
import sys

# 添加父目录到path以导入config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

class DataLoader:
    def __init__(self, db_path=None, token=None):
        """初始化数据加载器
        
        Args:
            db_path: 数据库路径，默认为 yc 目录下的 cache.db
            token: tushare token，如果为 None 则使用默认 token
        """
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache.db")
        
        if token is None:
            token = config.TUSHARE_TOKEN
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_database()
        
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    def _init_database(self):
        """初始化数据库表"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily(
                code TEXT, 
                date TEXT, 
                open REAL, 
                high REAL, 
                low REAL, 
                close REAL, 
                vol REAL, 
                PRIMARY KEY(code, date)
            )
        """)
        
        # 创建股票信息表，用于存储中文名称
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_info(
                code TEXT PRIMARY KEY,
                name TEXT,
                industry TEXT,
                market TEXT
            )
        """)
    
    def get_daily(self, code, start, end, asset="E"):
        """获取日线数据
        
        Args:
            code: 股票代码
            start: 开始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)
            asset: 资产类型，"E" 为股票，"FD" 为基金
            
        Returns:
            DataFrame: 包含日线数据的 DataFrame
        """
        start_str = start.replace("-", "")
        end_str = end.replace("-", "")
        
        # 先从数据库查询
        sql = "SELECT * FROM daily WHERE code=? AND date BETWEEN ? AND ? ORDER BY date"
        df = pd.read_sql(sql, self.conn, params=(code, start_str, end_str))
        
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            # 数据清洗
            numeric_cols = ['open', 'high', 'low', 'close', 'vol']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=numeric_cols)
            return df
        
        # 数据库中没有，从 API 获取
        # 对于ETF，使用fund_daily接口
        if 'ETF' in str(code).upper() or str(code).startswith(('15', '51', '58', '56')):
            api = self.pro.fund_daily
            asset = "FD"
        else:
            api = self.pro.daily
            
        try:
            df = api(ts_code=code, start_date=start_str, end_date=end_str)
            
            if df.empty:
                # 尝试不带后缀的代码
                if '.' in code:
                    code_no_suffix = code.split('.')[0]
                    df = api(ts_code=code_no_suffix, start_date=start_str, end_date=end_str)
                
                if df.empty:
                    # 尝试添加后缀
                    if not code.endswith('.SZ') and not code.endswith('.SH'):
                        for suffix in ['.SZ', '.SH']:
                            try:
                                df = api(ts_code=code + suffix, start_date=start_str, end_date=end_str)
                                if not df.empty:
                                    code = code + suffix
                                    break
                            except:
                                continue
            
            if df.empty:
                return df
                
            df = df.rename(columns={"trade_date": "date", "ts_code": "code"}).sort_values("date")
            
            # 只保留需要的字段
            cols = ["code", "date", "open", "high", "low", "close", "vol"]
            available_cols = [col for col in cols if col in df.columns]
            df = df[available_cols]
            
            # 数据清洗：确保数值类型正确
            numeric_cols = ['open', 'high', 'low', 'close', 'vol']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 移除无效数据
            df = df.dropna(subset=[col for col in numeric_cols if col in df.columns])
            
            # 确保没有无限值
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if df.empty:
                return df
            
            # 保存到数据库，使用INSERT OR IGNORE避免唯一约束冲突
            try:
                df.to_sql("daily", self.conn, if_exists="append", index=False, method="multi")
            except Exception as e:
                # 如果发生唯一约束冲突，逐条插入
                for _, row in df.iterrows():
                    try:
                        self.conn.execute("""
                            INSERT OR IGNORE INTO daily(code, date, open, high, low, close, vol)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (row['code'], row['date'], row.get('open', 0), row.get('high', 0),
                              row.get('low', 0), row.get('close', 0), row.get('vol', 0)))
                        self.conn.commit()
                    except Exception as insert_error:
                        print(f"插入数据失败: {insert_error}")
            
            df["date"] = pd.to_datetime(df["date"])
            
            return df
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, code):
        """获取股票/ETF信息（包括中文名称）
        
        Args:
            code: 股票/ETF代码
            
        Returns:
            dict: 包含证券信息的字典
        """
        # 先从数据库查询
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, industry, market FROM stock_info WHERE code = ?", (code,))
        result = cursor.fetchone()
        
        if result:
            return {
                "name": result[0],
                "industry": result[1] or "ETF",
                "market": result[2] or "基金"
            }
        
        # 数据库中没有，从 API 获取
        try:
            # 先尝试获取股票信息
            stock_info = self.pro.stock_basic(exchange='', ts_code=code, fields='ts_code,name,industry,market,area,list_date')
            
            if not stock_info.empty:
                info = stock_info.iloc[0]
                
                # 确定市场类型
                if code.endswith('.SH'):
                    market = '上交所'
                elif code.endswith('.SZ'):
                    market = '深交所'
                else:
                    market = info.get('market', '未知市场')
                
                stock_data = {
                    "name": info['name'],
                    "industry": info.get('industry', '未知行业'),
                    "market": market
                }
                
                # 保存到数据库
                self.conn.execute(
                    "INSERT OR REPLACE INTO stock_info(code, name, industry, market) VALUES (?, ?, ?, ?)",
                    (code, stock_data["name"], stock_data["industry"], stock_data["market"])
                )
                self.conn.commit()
                
                return stock_data
            
            # 如果不是股票，尝试获取ETF信息
            etf_info = self.pro.fund_basic(market='E', ts_code=code, fields='ts_code,name,management,found_date')
            
            if not etf_info.empty:
                info = etf_info.iloc[0]
                
                etf_data = {
                    "name": info['name'],
                    "industry": "ETF基金",
                    "market": "基金"
                }
                
                # 保存到数据库
                self.conn.execute(
                    "INSERT OR REPLACE INTO stock_info(code, name, industry, market) VALUES (?, ?, ?, ?)",
                    (code, etf_data["name"], etf_data["industry"], etf_data["market"])
                )
                self.conn.commit()
                
                return etf_data
                
        except Exception as e:
            print(f"获取证券信息失败: {e}")
        
        return {
            "name": code,
            "industry": "ETF基金",
            "market": "基金"
        }
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()