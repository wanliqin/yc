"""
配置文件
"""

import os

# Tushare配置
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', 'bd5193e8e1f07ef9b7bbdc8bf7efdc9ac054932082ae52c9804c01e0')

# 数据库配置
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'cache.db')

# 模型配置
DEFAULT_WINDOW_SIZES = [120, 250]
DEFAULT_TRIALS = 30
MIN_DATA_POINTS = 50

# Web服务配置
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')