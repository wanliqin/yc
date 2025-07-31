import datetime as dt
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from io import StringIO
import pandas as pd
import numpy as np
import os
import sys
import sqlite3
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from models.stock_predictor import StockPredictor
from database.prediction_manager import PredictionManager

app = FastAPI(title="股票智能预测系统", description="基于机器学习的股票涨跌预测")

# 初始化组件
data_loader = DataLoader()
feature_engineer = FeatureEngineer()
prediction_manager = PredictionManager()

# 初始化股票搜索数据库
def init_stock_search_db():
    """初始化股票搜索数据库"""
    conn = sqlite3.connect('cache.db')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_search (
            code TEXT PRIMARY KEY,
            name TEXT,
            pinyin TEXT,
            market_type TEXT
        )
    """)
    
    # 检查是否需要初始化数据
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM stock_search")
    count = cursor.fetchone()[0]
    
    if count == 0:
        try:
            import tushare as ts
            ts.set_token("bd5193e8e1f07ef9b7bbdc8bf7efdc9ac054932082ae52c9804c01e0")
            pro = ts.pro_api()
            
            # 获取所有股票（包括ETF）
            stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
            
            # 获取ETF基金
            etfs = pro.fund_basic(market='E', status='L', fields='ts_code,name')
            
            # 合并股票和ETF
            all_securities = []
            
            # 添加股票
            for _, stock in stocks.iterrows():
                all_securities.append({
                    'code': stock['ts_code'],
                    'name': stock['name'],
                    'type': '股票'
                })
            
            # 添加ETF
            for _, etf in etfs.iterrows():
                all_securities.append({
                    'code': etf['ts_code'],
                    'name': etf['name'],
                    'type': 'ETF'
                })
            
            for security in all_securities:
                # 生成拼音首字母
                pinyin = security['name'][:1] if security['name'] else ''
                
                # 判断资产类型
                market_type = security['type']
                
                conn.execute(
                    "INSERT OR IGNORE INTO stock_search(code, name, pinyin, market_type) VALUES (?, ?, ?, ?)",
                    (security['code'], security['name'], pinyin, market_type)
                )
            
            conn.commit()
            print(f"成功初始化 {len(all_securities)} 条证券数据")
        except Exception as e:
            print(f"初始化股票搜索数据库失败: {e}")
            
            # 添加一些常见的ETF作为备选
            common_etfs = [
                ('159915.SZ', '创业板ETF', 'ETF'),
                ('510300.SH', '沪深300ETF', 'ETF'),
                ('510500.SH', '中证500ETF', 'ETF'),
                ('512000.SH', '券商ETF', 'ETF'),
                ('512880.SH', '证券ETF', 'ETF'),
                ('513050.SH', '中概互联网ETF', 'ETF'),
                ('515030.SH', '新能源车ETF', 'ETF'),
                ('516160.SH', '新能源ETF', 'ETF'),
                ('518880.SH', '黄金ETF', 'ETF'),
                ('159949.SZ', '创业板50ETF', 'ETF'),
                ('159928.SZ', '中证500ETF', 'ETF'),
                ('159939.SZ', '信息技术ETF', 'ETF'),
                ('159952.SZ', '创业板ETF', 'ETF'),
                ('510050.SH', '上证50ETF', 'ETF'),
                ('510180.SH', '上证180ETF', 'ETF'),
                ('510330.SH', '沪深300ETF', 'ETF'),
                ('512010.SH', '医药ETF', 'ETF'),
                ('512170.SH', '医疗ETF', 'ETF'),
                ('512480.SH', '半导体ETF', 'ETF'),
                ('512690.SH', '酒ETF', 'ETF'),
                ('512720.SH', '计算机ETF', 'ETF'),
                ('512760.SH', '芯片ETF', 'ETF'),
                ('512800.SH', '银行ETF', 'ETF'),
                ('512900.SH', '证券ETF', 'ETF'),
                ('513030.SH', '德国ETF', 'ETF'),
                ('513100.SH', '纳指ETF', 'ETF'),
                ('513180.SH', '恒生科技ETF', 'ETF'),
                ('513330.SH', '恒生互联网ETF', 'ETF'),
                ('515000.SH', '科技ETF', 'ETF'),
                ('515700.SH', '新能车ETF', 'ETF'),
                ('516110.SH', '汽车ETF', 'ETF'),
                ('516150.SH', '稀土ETF', 'ETF'),
                ('516510.SH', '云计算ETF', 'ETF'),
                ('516520.SH', '智能驾驶ETF', 'ETF'),
                ('516780.SH', '稀土ETF', 'ETF'),
                ('516830.SH', '光伏ETF', 'ETF'),
                ('516880.SH', '光伏50ETF', 'ETF'),
                ('517090.SH', '沪港深500ETF', 'ETF'),
                ('517110.SH', '沪港深300ETF', 'ETF'),
                ('517990.SH', '沪港深高股息ETF', 'ETF'),
                ('518660.SH', '黄金ETF', 'ETF'),
                ('518800.SH', '黄金ETF', 'ETF'),
                ('518850.SH', '黄金ETF', 'ETF'),
                ('518860.SH', '黄金ETF', 'ETF'),
                ('518890.SH', '黄金ETF', 'ETF'),
                ('519674.SH', '创业板50ETF', 'ETF'),
                ('560010.SH', '碳中和ETF', 'ETF'),
                ('560050.SH', '央企创新ETF', 'ETF'),
                ('560500.SH', '中证500ETF', 'ETF'),
                ('560800.SH', '数字经济ETF', 'ETF'),
                ('561010.SH', '新能源车ETF', 'ETF'),
                ('561120.SH', '芯片ETF', 'ETF'),
                ('561130.SH', '新能源车ETF', 'ETF'),
                ('561190.SH', '机器人ETF', 'ETF'),
                ('561330.SH', '新能源车ETF', 'ETF'),
                ('561350.SH', '芯片ETF', 'ETF'),
                ('561600.SH', '光伏ETF', 'ETF'),
                ('561910.SH', '电池ETF', 'ETF'),
                ('561990.SH', '新能源车ETF', 'ETF'),
                ('562000.SH', '稀有金属ETF', 'ETF'),
                ('562510.SH', '旅游ETF', 'ETF'),
                ('562800.SH', '稀有金属ETF', 'ETF'),
                ('562990.SH', '新能源车ETF', 'ETF'),
                ('563000.SH', '新能源车ETF', 'ETF'),
                ('588000.SH', '科创50ETF', 'ETF'),
                ('588080.SH', '科创50ETF', 'ETF'),
                ('588090.SH', '科创50ETF', 'ETF'),
                ('588180.SH', '科创50ETF', 'ETF'),
                ('588200.SH', '科创50ETF', 'ETF'),
                ('588280.SH', '科创50ETF', 'ETF'),
                ('588290.SH', '科创50ETF', 'ETF'),
                ('588300.SH', '科创50ETF', 'ETF'),
                ('588360.SH', '科创50ETF', 'ETF'),
                ('588380.SH', '科创50ETF', 'ETF'),
                ('588390.SH', '科创50ETF', 'ETF'),
                ('588400.SH', '科创50ETF', 'ETF'),
                ('588460.SH', '科创50ETF', 'ETF'),
                ('588480.SH', '科创50ETF', 'ETF'),
                ('588500.SH', '科创50ETF', 'ETF'),
                ('588520.SH', '科创50ETF', 'ETF'),
                ('588530.SH', '科创50ETF', 'ETF'),
                ('588550.SH', '科创50ETF', 'ETF'),
                ('588560.SH', '科创50ETF', 'ETF'),
                ('588580.SH', '科创50ETF', 'ETF'),
                ('588600.SH', '科创50ETF', 'ETF'),
                ('588680.SH', '科创50ETF', 'ETF'),
                ('588690.SH', '科创50ETF', 'ETF'),
                ('588700.SH', '科创50ETF', 'ETF'),
                ('588720.SH', '科创50ETF', 'ETF'),
                ('588730.SH', '科创50ETF', 'ETF'),
                ('588750.SH', '科创50ETF', 'ETF'),
                ('588760.SH', '科创50ETF', 'ETF'),
                ('588770.SH', '科创50ETF', 'ETF'),
                ('588780.SH', '科创50ETF', 'ETF'),
                ('588790.SH', '科创50ETF', 'ETF'),
                ('588800.SH', '科创50ETF', 'ETF'),
                ('588880.SH', '科创50ETF', 'ETF'),
                ('588900.SH', '科创50ETF', 'ETF'),
                ('588960.SH', '科创50ETF', 'ETF'),
                ('588990.SH', '科创50ETF', 'ETF'),
                ('589000.SH', '科创50ETF', 'ETF'),
                ('589080.SH', '科创50ETF', 'ETF'),
                ('589090.SH', '科创50ETF', 'ETF'),
                ('589100.SH', '科创50ETF', 'ETF'),
                ('589180.SH', '科创50ETF', 'ETF'),
                ('589200.SH', '科创50ETF', 'ETF'),
                ('589280.SH', '科创50ETF', 'ETF'),
                ('589290.SH', '科创50ETF', 'ETF'),
                ('589300.SH', '科创50ETF', 'ETF'),
                ('589380.SH', '科创50ETF', 'ETF'),
                ('589390.SH', '科创50ETF', 'ETF'),
                ('589400.SH', '科创50ETF', 'ETF'),
                ('589480.SH', '科创50ETF', 'ETF'),
                ('589490.SH', '科创50ETF', 'ETF'),
                ('589500.SH', '科创50ETF', 'ETF'),
                ('589580.SH', '科创50ETF', 'ETF'),
                ('589590.SH', '科创50ETF', 'ETF'),
                ('589600.SH', '科创50ETF', 'ETF'),
                ('589680.SH', '科创50ETF', 'ETF'),
                ('589690.SH', '科创50ETF', 'ETF'),
                ('589700.SH', '科创50ETF', 'ETF'),
                ('589780.SH', '科创50ETF', 'ETF'),
                ('589790.SH', '科创50ETF', 'ETF'),
                ('589800.SH', '科创50ETF', 'ETF'),
                ('589880.SH', '科创50ETF', 'ETF'),
                ('589890.SH', '科创50ETF', 'ETF'),
                ('589900.SH', '科创50ETF', 'ETF'),
                ('589980.SH', '科创50ETF', 'ETF'),
                ('589990.SH', '科创50ETF', 'ETF')
            ]
            
            for code, name, mtype in common_etfs:
                conn.execute(
                    "INSERT OR IGNORE INTO stock_search(code, name, pinyin, market_type) VALUES (?, ?, ?, ?)",
                    (code, name, name[:1], mtype)
                )
    
    conn.close()

init_stock_search_db()

@app.get("/", response_class=HTMLResponse)
def root():
    """主页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 股票智能预测系统</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
            }
            
            .header {
                text-align: center;
                color: white;
                margin-bottom: 40px;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                font-size: 1.2rem;
                opacity: 0.9;
            }
            
            .card {
                background: white;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                padding: 40px;
                margin-bottom: 30px;
                transition: transform 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
            }
            
            .card-header {
                display: flex;
                align-items: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #f0f0f0;
            }
            
            .card-header i {
                font-size: 2rem;
                margin-right: 15px;
                color: #667eea;
            }
            
            .card-header h3 {
                font-size: 1.5rem;
                color: #333;
            }
            
            .form-group {
                margin-bottom: 25px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #555;
            }
            
            .form-group input, .form-group select {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            
            .form-group input:focus, .form-group select:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 50px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            .btn:active {
                transform: translateY(0);
            }
            
            .results {
                margin-top: 30px;
            }
            
            .prediction-card {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 30px;
                border-radius: 20px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
            }
            
            .prediction-header {
                text-align: center;
                margin-bottom: 20px;
            }
            
            .prediction-result {
                font-size: 2rem;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
            }
            
            .prediction-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            
            .stat-item {
                background: rgba(255,255,255,0.2);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            
            .stat-value {
                font-size: 1.5rem;
                font-weight: bold;
            }
            
            .stat-label {
                font-size: 0.9rem;
                opacity: 0.9;
            }
            
            .daily-predictions {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                margin-top: 20px;
            }
            
            .daily-predictions h4 {
                color: #333;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }
            
            .daily-predictions h4 i {
                margin-right: 10px;
                color: #667eea;
            }
            
            .prediction-list {
                max-height: 300px;
                overflow-y: auto;
            }
            
            .prediction-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px;
                margin: 5px 0;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .prediction-date {
                font-weight: 600;
                color: #555;
            }
            
            .prediction-label {
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.9rem;
            }
            
            .prediction-up {
                background: #d4edda;
                color: #155724;
            }
            
            .prediction-down {
                background: #f8d7da;
                color: #721c24;
            }
            
            .history-table {
                width: 100%;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .history-table th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
            }
            
            .history-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }
            
            .history-table tr:hover {
                background: #f8f9fa;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                font-size: 1.2rem;
                color: #667eea;
            }
            
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
            }
            
            .search-container {
                position: relative;
            }
            
            .search-results {
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                max-height: 200px;
                overflow-y: auto;
                z-index: 1000;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            .search-item {
                padding: 12px 15px;
                cursor: pointer;
                border-bottom: 1px solid #f0f0f0;
                transition: background-color 0.2s;
            }
            
            .search-item:hover {
                background-color: #f8f9fa;
            }
            
            .search-item:last-child {
                border-bottom: none;
            }
            
            .params-display {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 20px;
                border-radius: 15px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
            }
            
            .training-progress {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px;
                margin: 20px 0;
            }
            
            .progress-bar {
                width: 100%;
                height: 8px;
                background: rgba(255,255,255,0.3);
                border-radius: 4px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: white;
                border-radius: 4px;
                transition: width 0.3s ease;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .card {
                    padding: 20px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .prediction-stats {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-chart-line"></i> 股票智能预测系统</h1>
                <p>基于机器学习的股票涨跌预测</p>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-search"></i>
                    <h3>单个股票预测</h3>
                </div>
                <form action="/predict" method="post" id="predictForm">
                    <div class="form-group">
                        <label>
                            <i class="fas fa-building"></i> 股票代码/名称
                            <span class="tooltip">
                                <i class="fas fa-info-circle"></i>
                                <span class="tooltiptext">支持股票代码、中文名称、拼音首字母模糊搜索</span>
                            </span>
                        </label>
                        <div class="search-container">
                            <input name="code" id="stockInput" placeholder="如：000001.SZ、平安银行、ping" autocomplete="off" required>
                            <div id="searchResults" class="search-results" style="display: none;"></div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label><i class="fas fa-calendar-alt"></i> 窗口期</label>
                        <input name="win" value="120,250" placeholder="逗号分隔，如：120,250">
                        <small style="color: #666;">历史数据天数，多个窗口用逗号分隔</small>
                    </div>
                    
                    <div class="form-group">
                        <label>
                            <i class="fas fa-cogs"></i> 调参次数
                            <span class="tooltip">
                                <i class="fas fa-info-circle"></i>
                                <span class="tooltiptext">模型超参数优化次数，数值越大预测越准确但耗时越长</span>
                            </span>
                        </label>
                        <input name="trials" type="number" value="30" min="5" max="100">
                        <small style="color: #666;">建议10-50次，数值越大越准确但耗时越长</small>
                    </div>
                    
                    <button type="submit" class="btn">
                        <i class="fas fa-magic"></i> 开始预测
                    </button>
                </form>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-history"></i>
                    <h3>历史预测记录</h3>
                </div>
                <form action="/history" method="get" id="historyForm">
                    <div class="form-group">
                        <label><i class="fas fa-filter"></i> 股票代码</label>
                        <input name="code" placeholder="留空查看全部记录">
                    </div>
                    <button type="submit" class="btn">
                        <i class="fas fa-eye"></i> 查看历史记录
                    </button>
                </form>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <div class="prediction-card">
                    <div class="prediction-header">
                        <h2><i class="fas fa-chart-bar"></i> 预测结果</h2>
                    </div>
                    <div id="predictionContent"></div>
                </div>
            </div>
            
            <div id="historyResults" class="results" style="display: none;">
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-list-alt"></i>
                        <h3>历史预测记录</h3>
                    </div>
                    <div id="historyContent"></div>
                </div>
            </div>
        </div>

        <script>
            // 股票搜索功能
            const stockInput = document.getElementById('stockInput');
            const searchResults = document.getElementById('searchResults');
            
            // 检查是否有正在进行的预测
            let isPredicting = false;
            let currentPredictionData = null;
            
            // 页面加载时检查是否有未完成的预测
            window.addEventListener('load', function() {
                const savedState = localStorage.getItem('predictionState');
                if (savedState) {
                    const state = JSON.parse(savedState);
                    if (state.isPredicting) {
                        isPredicting = true;
                        currentPredictionData = state;
                        // 立即显示预测状态，不需要用户操作
                        setTimeout(() => {
                            restorePredictionState();
                        }, 100);
                    }
                }
            });
            
            // 保存预测状态
            function savePredictionState(code, win, trials) {
                const state = {
                    isPredicting: true,
                    code: code,
                    win: win,
                    trials: trials,
                    timestamp: new Date().toISOString()
                };
                localStorage.setItem('predictionState', JSON.stringify(state));
            }
            
            // 清除预测状态
            function clearPredictionState() {
                isPredicting = false;
                currentPredictionData = null;
                localStorage.removeItem('predictionState');
            }
            
            // 恢复预测状态
            function restorePredictionState() {
                if (currentPredictionData) {
                    // 填充表单数据
                    document.getElementById('stockInput').value = currentPredictionData.code;
                    document.querySelector('input[name="win"]').value = currentPredictionData.win;
                    document.querySelector('input[name="trials"]').value = currentPredictionData.trials;
                    
                    // 显示结果区域
                    const resultsDiv = document.getElementById('results');
                    const contentDiv = document.getElementById('predictionContent');
                    
                    resultsDiv.style.display = 'block';
                    document.getElementById('historyResults').style.display = 'none';
                    
                    // 显示正在预测的状态
                    contentDiv.innerHTML = `
                        <div class="training-progress">
                            <h4><i class="fas fa-cog fa-spin"></i> 正在分析数据中，请勿刷新页面...</h4>
                            <div class="progress-steps">
                                <div class="step active"><i class="fas fa-download"></i> 获取股票数据</div>
                                <div class="step"><i class="fas fa-chart-bar"></i> 计算技术指标</div>
                                <div class="step"><i class="fas fa-brain"></i> 训练AI模型</div>
                                <div class="step"><i class="fas fa-magic"></i> 生成预测</div>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: 25%"></div>
                            </div>
                            <p style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                                股票代码: ${currentPredictionData.code}<br>
                                开始时间: ${new Date(currentPredictionData.timestamp).toLocaleString()}
                            </p>
                        </div>
                    `;
                    
                    // 禁用表单
                    const submitBtn = document.getElementById('predictForm').querySelector('button[type="submit"]');
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 预测进行中...';
                    
                    // 禁用输入框
                    document.getElementById('stockInput').disabled = true;
                    document.querySelector('input[name="win"]').disabled = true;
                    document.querySelector('input[name="trials"]').disabled = true;
                    
                    // 滚动到结果区域
                    resultsDiv.scrollIntoView({ behavior: 'smooth' });
                }
            }
            
            stockInput.addEventListener('input', async function() {
                if (isPredicting) return; // 如果正在预测，禁用搜索
                
                const query = this.value.trim();
                if (query.length < 1) {
                    searchResults.style.display = 'none';
                    return;
                }
                
                try {
                    const response = await fetch('/search_stocks?query=' + encodeURIComponent(query));
                    const data = await response.json();
                    
                    if (data.stocks && data.stocks.length > 0) {
                        let html = '';
                        data.stocks.forEach(stock => {
                            html += '<div class="search-item" data-code="' + stock.code + '" data-type="' + stock.type + '">' +
                                stock.code + ' - ' + stock.name +
                                '</div>';
                        });
                        searchResults.innerHTML = html;
                        searchResults.style.display = 'block';
                    } else {
                        searchResults.style.display = 'none';
                    }
                } catch (error) {
                    searchResults.style.display = 'none';
                }
            });
            
            // 点击搜索结果
            searchResults.addEventListener('click', function(e) {
                if (isPredicting) return;
                if (e.target.classList.contains('search-item')) {
                    stockInput.value = e.target.dataset.code;
                    searchResults.style.display = 'none';
                }
            });
            
            // 点击其他地方关闭搜索结果
            document.addEventListener('click', function(e) {
                if (!stockInput.contains(e.target) && !searchResults.contains(e.target)) {
                    searchResults.style.display = 'none';
                }
            });
            
            document.getElementById('predictForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                if (isPredicting) {
                    alert('当前有预测任务正在进行中，请等待完成后再开始新的预测！');
                    return;
                }
                
                const formData = new FormData(this);
                const stockCode = formData.get('code');
                const winValue = formData.get('win');
                const trialsValue = formData.get('trials');
                
                // 验证股票代码不能为空
                if (!stockCode || stockCode.trim() === '') {
                    alert('请输入股票代码或名称！');
                    return;
                }
                
                // 保存预测状态
                savePredictionState(stockCode, winValue, trialsValue);
                
                const resultsDiv = document.getElementById('results');
                const contentDiv = document.getElementById('predictionContent');
                
                resultsDiv.style.display = 'block';
                document.getElementById('historyResults').style.display = 'none';
                
                // 禁用表单
                const submitBtn = this.querySelector('button[type="submit"]');
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 预测进行中...';
                
                // 显示动态进度
                contentDiv.innerHTML = `
                    <div class="training-progress">
                        <h4><i class="fas fa-cog fa-spin"></i> 正在分析数据中，请勿刷新页面...</h4>
                        <div class="progress-steps">
                            <div class="step active"><i class="fas fa-download"></i> 获取股票数据</div>
                            <div class="step"><i class="fas fa-chart-bar"></i> 计算技术指标</div>
                            <div class="step"><i class="fas fa-brain"></i> 训练AI模型</div>
                            <div class="step"><i class="fas fa-magic"></i> 生成预测</div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 25%"></div>
                        </div>
                        <p style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                            股票代码: ${stockCode}<br>
                            开始时间: ${new Date().toLocaleString()}
                        </p>
                    </div>
                `;
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    displayResults(data);
                    
                    // 清除预测状态
                    clearPredictionState();
                    
                    // 恢复表单
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-magic"></i> 开始预测';
                    
                    // 恢复输入框
                    document.getElementById('stockInput').disabled = false;
                    document.querySelector('input[name="win"]').disabled = false;
                    document.querySelector('input[name="trials"]').disabled = false;
                    
                } catch (error) {
                    contentDiv.innerHTML = '<div class="error"><i class="fas fa-exclamation-triangle"></i> 预测失败: ' + error.message + '</div>';
                    
                    // 清除预测状态
                    clearPredictionState();
                    
                    // 恢复表单
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-magic"></i> 开始预测';
                }
            });

            document.getElementById('historyForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const code = formData.get('code');
                const historyDiv = document.getElementById('historyResults');
                const contentDiv = document.getElementById('historyContent');
                
                historyDiv.style.display = 'block';
                document.getElementById('results').style.display = 'none';
                contentDiv.innerHTML = '<p class="loading">正在加载历史记录...</p>';
                
                try {
                    const url = code ? '/history?code=' + encodeURIComponent(code) : '/history';
                    const response = await fetch(url);
                    const data = await response.json();
                    displayHistory(data);
                } catch (error) {
                    contentDiv.innerHTML = '<p class="error">加载失败: ' + error.message + '</p>';
                }
            });

            function displayResults(data) {
                const contentDiv = document.getElementById('predictionContent');
                
                if (data.error) {
                    contentDiv.innerHTML = '<div class="error"><i class="fas fa-exclamation-triangle"></i> ' + data.error + '</div>';
                    return;
                }
                
                const isUp = data.prediction === '上涨';
                const predictionClass = isUp ? 'prediction-up' : 'prediction-down';
                const predictionIcon = isUp ? 'fas fa-arrow-up' : 'fas fa-arrow-down';
                const predictionColor = isUp ? '#28a745' : '#dc3545';
                
                let html = `
                    <div class="prediction-result" style="color: ${predictionColor}">
                        <i class="${predictionIcon}"></i> ${data.prediction}
                    </div>
                    
                    <div class="prediction-stats">
                        <div class="stat-item">
                            <div class="stat-value">${data.code}</div>
                            <div class="stat-label">股票代码</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${data.name}</div>
                            <div class="stat-label">股票名称</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${(data.probability * 100).toFixed(2)}%</div>
                            <div class="stat-label">置信度</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${data.industry || '未知'}</div>
                            <div class="stat-label">行业</div>
                        </div>
                    </div>
                    
                    <div class="params-display">
                        <h4><i class="fas fa-cogs"></i> 模型参数</h4>
                        ${data.model_params}
                    </div>
                    
                    <div class="daily-predictions">
                        <h4><i class="fas fa-chart-bar"></i> 最近30个交易日分析</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                            <div style="background: #ffffff; border: 3px solid #3498db; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                                <h5 style="margin-bottom: 10px; color: #2c3e50; font-weight: bold;">
                                    <i class="fas fa-bullseye" style="color: #3498db;"></i> 验证准确率
                                </h5>
                                <div style="font-size: 1.8rem; font-weight: bold; color: #3498db;">${data.validation_accuracy}%</div>
                                <small style="color: #7f8c8d;">模型训练验证</small>
                            </div>
                            <div style="background: #ffffff; border: 3px solid #e74c3c; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                                <h5 style="margin-bottom: 10px; color: #2c3e50; font-weight: bold;">
                                    <i class="fas fa-chart-line" style="color: #e74c3c;"></i> 回测准确率
                                </h5>
                                <div style="font-size: 1.8rem; font-weight: bold; color: #e74c3c;">${data.prediction_accuracy}%</div>
                                <small style="color: #7f8c8d;">最近30日真实回测</small>
                            </div>
                            <div style="background: #ffffff; border: 3px solid #27ae60; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                                <h5 style="margin-bottom: 10px; color: #2c3e50; font-weight: bold;">
                                    <i class="fas fa-chart-pie" style="color: #27ae60;"></i> 真实涨跌
                                </h5>
                                <div style="font-size: 1.3rem; font-weight: bold;">
                                    <span style="color: #27ae60;">↑ ${data.recent_stats.up_days}</span> /
                                    <span style="color: #c0392b;">↓ ${data.recent_stats.down_days}</span>
                                </div>
                                <small style="color: #7f8c8d;">最近${data.recent_stats.total_days}日</small>
                            </div>
                        </div>
                        
                        <div style="background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin-bottom: 20px; border-radius: 5px;">
                            <h5 style="color: #2c3e50; margin-bottom: 10px;">
                                <i class="fas fa-info-circle" style="color: #3498db;"></i> 准确率说明
                            </h5>
                            <div style="font-size: 0.9rem; line-height: 1.6; color: #555;">
                                <strong>验证准确率：</strong>模型在训练数据上的交叉验证表现，反映模型拟合能力<br>
                                <strong>回测准确率：</strong>模型在历史数据上的实际预测准确率，更接近真实交易表现<br>
                                <strong>差异原因：</strong>验证使用训练期数据，回测使用未来数据，存在时间序列差异
                            </div>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                            <h5 style="color: #667eea; margin-bottom: 10px;">📈 价格表现</h5>
                            <div style="display: flex; justify-content: space-between; text-align: center;">
                                <div>
                                    <div style="font-size: 1.2rem; font-weight: bold; color: ${data.recent_stats.avg_price_change >= 0 ? '#28a745' : '#dc3545'}">
                                        ${data.recent_stats.avg_price_change >= 0 ? '+' : ''}${data.recent_stats.avg_price_change}%
                                    </div>
                                    <small>平均日涨跌幅</small>
                                </div>
                                <div>
                                    <div style="font-size: 1.2rem; font-weight: bold;">${data.recent_stats.total_days}</div>
                                    <small>交易日数</small>
                                </div>
                            </div>
                        </div>
                        
                        <h5 style="color: #333; margin-bottom: 15px;">📅 每日详细数据（预测 vs 实际行情对比）</h5>
                        <div style="background: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr; gap: 10px; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 10px;">
                                <span>日期</span>
                                <span>涨跌幅</span>
                                <span style="color: #3498db;">模型预测</span>
                                <span style="color: #e74c3c;">实际行情</span>
                                <span>置信度</span>
                                <span>结果</span>
                            </div>
                        </div>
                        <div class="prediction-list">
                `;
                
                data.daily_predictions.forEach(item => {
                    const predClass = item.prediction === '上涨' ? 'prediction-up' : 'prediction-down';
                    const actualClass = item.actual === '上涨' ? 'prediction-up' : 'prediction-down';
                    const predIcon = item.prediction === '上涨' ? 'fas fa-arrow-up' : 'fas fa-arrow-down';
                    const actualIcon = item.actual === '上涨' ? 'fas fa-arrow-up' : 'fas fa-arrow-down';
                    const isCorrect = item.is_correct ? '✅' : '❌';
                    
                    html += `
                        <div class="prediction-item" style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr; gap: 10px; align-items: center; background: white; margin: 5px 0; padding: 12px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <span style="font-weight: 600; color: #2c3e50;">${item.date}</span>
                            <span style="font-weight: bold; color: ${item.price_change >= 0 ? '#27ae60' : '#e74c3c'}">
                                ${item.price_change >= 0 ? '+' : ''}${item.price_change}%
                            </span>
                            <span class="prediction-label ${predClass}" style="font-size: 0.9rem;">
                                <i class="${predIcon}"></i> ${item.prediction}
                            </span>
                            <span class="prediction-label ${actualClass}" style="font-size: 0.9rem;">
                                <i class="${actualIcon}"></i> ${item.actual}
                            </span>
                            <span style="font-weight: 600; color: #8e44ad;">${(item.probability * 100).toFixed(1)}%</span>
                            <span style="font-size: 1.2rem; text-align: center;">${isCorrect}</span>
                        </div>
                    `;
                });
                
                html += `
                        </div>
                    </div>
                `;
                
                contentDiv.innerHTML = html;
            }

            function displayHistory(data) {
                const contentDiv = document.getElementById('historyContent');
                
                if (data.error) {
                    contentDiv.innerHTML = '<div class="error"><i class="fas fa-exclamation-triangle"></i> ' + data.error + '</div>';
                    return;
                }
                
                if (!data.history || data.history.length === 0) {
                    contentDiv.innerHTML = '<div style="text-align: center; padding: 40px; color: #666;"><i class="fas fa-inbox" style="font-size: 3rem; margin-bottom: 10px;"></i><p>暂无历史记录</p></div>';
                    return;
                }
                
                let html = '<div class="history-table-container">';
                html += '<table class="history-table">';
                html += '<thead><tr>';
                html += '<th><i class="fas fa-hashtag"></i> 股票代码</th>';
                html += '<th><i class="fas fa-building"></i> 股票名称</th>';
                html += '<th><i class="fas fa-calendar"></i> 预测日期</th>';
                html += '<th><i class="fas fa-chart-line"></i> 预测结果</th>';
                html += '<th><i class="fas fa-percentage"></i> 置信度</th>';
                html += '<th><i class="fas fa-cogs"></i> 模型参数</th>';
                html += '</tr></thead><tbody>';
                
                data.history.forEach(item => {
                    const resultClass = item.prediction === '上涨' ? 'prediction-up' : 'prediction-down';
                    const resultIcon = item.prediction === '上涨' ? 'fas fa-arrow-up' : 'fas fa-arrow-down';
                    
                    html += '<tr>';
                    html += '<td><strong>' + item.code + '</strong></td>';
                    html += '<td>' + item.name + '</td>';
                    html += '<td>' + item.prediction_date + '</td>';
                    html += '<td><span class="prediction-label ' + resultClass + '"><i class="' + resultIcon + '"></i> ' + item.prediction + '</span></td>';
                    html += '<td><strong>' + (item.probability * 100).toFixed(2) + '%</strong></td>';
                    html += '<td><small>' + item.model_params + '</small></td>';
                    html += '</tr>';
                });
                
                html += '</tbody></table></div>';
                contentDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

@app.get("/search_stocks")
def search_stocks(query: str):
    """股票代码和名称模糊搜索"""
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    
    # 支持代码、名称、拼音模糊搜索，支持ETF
    cursor.execute("""
        SELECT code, name, market_type
        FROM stock_search
        WHERE code LIKE ? OR name LIKE ? OR pinyin LIKE ?
        ORDER BY
            CASE
                WHEN name LIKE ? THEN 1
                WHEN code LIKE ? THEN 2
                ELSE 3
            END,
            code
        LIMIT 20
    """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
    
    stocks = []
    for row in cursor.fetchall():
        stocks.append({
            'code': row[0],
            'name': row[1],
            'type': row[2]
        })
    
    conn.close()
    return {"stocks": stocks}

@app.post("/predict")
def predict(code: str = Form(...), win: str = Form("120,250"), trials: int = Form(30)):
    """预测股票涨跌"""
    try:
        # 验证股票代码不能为空
        if not code or code.strip() == '':
            return {"error": "股票代码不能为空，请输入有效的股票代码"}
        
        # 解析窗口期
        try:
            windows = [int(w.strip()) for w in win.split(',')]
        except ValueError:
            return {"error": "窗口期格式错误，请输入逗号分隔的数字，如：120,250"}
        
        # 验证窗口期
        if not windows or any(w <= 0 for w in windows):
            return {"error": "窗口期必须为正整数"}
        
        # 获取股票数据
        df = data_loader.get_daily(code, start="2023-01-01", end=dt.datetime.now().strftime("%Y-%m-%d"))
        if df.empty:
            return {"error": f"无法获取股票 {code} 的数据，请检查股票代码是否正确"}
        
        # 获取股票名称
        stock_info = data_loader.get_stock_info(code)
        name = stock_info["name"]
        industry = stock_info["industry"]
        
        # 判断资产类型 - ETF使用基金接口
        if 'ETF' in str(name).upper() or str(code).startswith(('15', '51', '58', '56')):
            asset_type = 'FD'  # ETF使用基金接口
        else:
            asset_type = 'E' if code.endswith('.SZ') or code.endswith('.SH') else 'FD'
        
        # 数据验证
        if len(df) < 20:
            return {"error": f"数据量不足，需要至少20条数据，当前只有{len(df)}条"}
        
        # 特征工程
        features_df = feature_engineer.create_features(df, windows)
        if features_df.empty:
            return {"error": "特征工程失败，数据质量不符合要求"}
        
        # 准备训练数据
        feature_cols = feature_engineer.get_feature_names()
        
        # 检查数据是否足够
        min_required = max(windows) + 10 if windows else 130
        if len(features_df) < min_required:
            return {"error": f"数据不足，需要至少{min_required}条数据，当前只有{len(features_df)}条"}
        
        # 验证特征数据
        X = features_df[feature_cols]
        y = features_df["label"]
        
        # 检查是否有空值
        if X.isnull().any().any():
            return {"error": "特征数据包含空值，请检查数据质量"}
        
        # 检查是否有无限值
        if np.isinf(X.values).any():
            return {"error": "特征数据包含无限值，请检查数据质量"}
        
        # 检查是否有足够的标签多样性
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return {"error": f"数据标签单一（只有{unique_labels}），无法进行有效预测"}
        
        # 过滤有效的窗口期
        valid_windows = [w for w in windows if len(features_df) >= w + 2]
        if not valid_windows:
            max_suggested = max(20, len(features_df) - 2)
            return {"error": f"窗口期设置过大，可用数据{len(features_df)}条，建议窗口期不超过{max_suggested}"}
        
        # 训练模型
        predictor = StockPredictor(feature_cols)
        model, best_params, val_acc = predictor.train_model(
            X,
            y,
            window_size=min(valid_windows),
            trials=trials,
            feature_columns=feature_cols
        )
        
        # 检查模型是否训练成功
        if model is None:
            return {"error": "模型训练失败，请检查数据质量或减小窗口期"}
        
        # 获取最新数据用于预测
        latest_data = features_df.iloc[-1:]
        latest_features = latest_data[feature_cols]
        
        # 验证最新特征数据
        if latest_features.isnull().any().any():
            return {"error": "最新数据包含空值，无法进行预测"}
        
        # 预测
        try:
            prediction_proba = model.predict_proba(latest_features)[0]
            prediction = "上涨" if prediction_proba[1] > 0.5 else "下跌"
            probability = float(max(prediction_proba))
        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}
        
        # 获取最近30个交易日的预测和真实数据（确保不越界）
        daily_predictions = []
        recent_analysis = []
        correct_predictions = 0
        total_predictions = 0
        
        max_history = min(30, len(features_df))
        for i in range(max_history):
            idx = len(features_df) - i - 1
            if idx >= 0:
                try:
                    # 获取特征和真实标签
                    test_features = features_df.iloc[idx:idx+1][feature_cols]
                    test_proba = model.predict_proba(test_features)[0]
                    test_pred = "上涨" if test_proba[1] > 0.5 else "下跌"
                    actual_label = "上涨" if features_df.iloc[idx]['label'] == 1 else "下跌"
                    
                    # 计算预测是否正确
                    is_correct = test_pred == actual_label
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # 计算涨跌幅（基于收盘价）
                    if idx > 0:
                        prev_close = features_df.iloc[idx-1]['close']
                        curr_close = features_df.iloc[idx]['close']
                        price_change = ((curr_close - prev_close) / prev_close) * 100
                    else:
                        price_change = 0
                    
                    # 确保实际标签与价格变化一致
                    actual_from_price = "上涨" if price_change > 0 else "下跌"
                    
                    daily_predictions.append({
                        'date': features_df.iloc[idx]['trade_date'].strftime('%Y-%m-%d'),
                        'prediction': test_pred,
                        'actual': actual_from_price,  # 使用基于价格的实际涨跌
                        'probability': float(max(test_proba)),
                        'price_change': round(price_change, 2),
                        'is_correct': test_pred == actual_from_price
                    })
                    
                    recent_analysis.append({
                        'date': features_df.iloc[idx]['trade_date'].strftime('%Y-%m-%d'),
                        'close': float(features_df.iloc[idx]['close']),
                        'change': round(price_change, 2),
                        'volume': int(features_df.iloc[idx]['vol']) if 'vol' in features_df.columns else 0
                    })
                    
                except Exception as e:
                    continue
        
        # 计算预测准确率（基于真实价格变化的回测准确率）
        actual_correct = sum(1 for item in daily_predictions if item['is_correct'])
        actual_total = len(daily_predictions)
        prediction_accuracy = (actual_correct / actual_total * 100) if actual_total > 0 else 0
        
        # 验证准确率计算是否正确
        calculated_correct = sum(1 for item in daily_predictions if item['is_correct'])
        calculated_total = len(daily_predictions)
        calculated_accuracy = (calculated_correct / calculated_total * 100) if calculated_total > 0 else 0
        
        # 确保一致性
        prediction_accuracy = calculated_accuracy
        
        # 计算最近30日真实涨跌幅统计（基于实际价格变化）
        up_days = sum(1 for item in daily_predictions if item['price_change'] > 0)
        down_days = sum(1 for item in daily_predictions if item['price_change'] < 0)
        flat_days = sum(1 for item in daily_predictions if item['price_change'] == 0)
        avg_change = sum(item['price_change'] for item in daily_predictions) / len(daily_predictions) if daily_predictions else 0
        
        # 保存预测结果
        prediction_manager.save_prediction({
            "code": code,
            "name": name,
            "prediction_date": dt.datetime.now().strftime("%Y-%m-%d"),
            "target_date": (dt.datetime.now() + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
            "signal": 1 if prediction == "上涨" else 0,
            "accuracy": probability,
            "votes": json.dumps([]),
            "window_sizes": json.dumps(windows),
            "trials": trials
        })
        
        # 格式化模型参数（优化显示 - 高对比度）
        model_params_str = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div style="background: #f8f9fa; border: 2px solid #dee2e6; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h5 style="color: #2c3e50; margin-bottom: 15px; font-size: 1.1rem; font-weight: bold;">
                    <i class="fas fa-chart-line" style="color: #3498db;"></i> 基础参数
                </h5>
                <div style="font-size: 1rem; line-height: 1.8;">
                    <div style="margin-bottom: 8px;">
                        <span style="color: #2c3e50; font-weight: 600;">窗口期:</span>
                        <span style="color: #e74c3c; font-weight: bold; margin-left: 10px;">{windows}</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: #2c3e50; font-weight: 600;">调参次数:</span>
                        <span style="color: #e74c3c; font-weight: bold; margin-left: 10px;">{trials}</span>
                    </div>
                    <div>
                        <span style="color: #2c3e50; font-weight: 600;">训练验证准确率:</span>
                        <span style="color: #27ae60; font-weight: bold; margin-left: 10px;">{(val_acc*100):.2f}%</span>
                    </div>
                </div>
            </div>
            <div style="background: #f8f9fa; border: 2px solid #dee2e6; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h5 style="color: #2c3e50; margin-bottom: 15px; font-size: 1.1rem; font-weight: bold;">
                    <i class="fas fa-robot" style="color: #9b59b6;"></i> 模型参数
                </h5>
                <div style="font-size: 1rem; line-height: 1.8;">
                    <div style="margin-bottom: 8px;">
                        <span style="color: #2c3e50; font-weight: 600;">算法:</span>
                        <span style="color: #8e44ad; font-weight: bold; margin-left: 10px;">{best_params.get('model', 'unknown').upper()}</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: #2c3e50; font-weight: 600;">树深度:</span>
                        <span style="color: #8e44ad; font-weight: bold; margin-left: 10px;">{best_params.get('max_depth', 'N/A')}</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: #2c3e50; font-weight: 600;">树数量:</span>
                        <span style="color: #8e44ad; font-weight: bold; margin-left: 10px;">{best_params.get('n_estimators', 'N/A')}</span>
                    </div>
                    <div>
                        <span style="color: #2c3e50; font-weight: 600;">学习率:</span>
                        <span style="color: #8e44ad; font-weight: bold; margin-left: 10px;">{best_params.get('learning_rate', 'N/A')}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return {
            "code": code,
            "name": name,
            "industry": industry,
            "asset_type": asset_type,
            "prediction": prediction,
            "probability": float(probability),
            "model_params": model_params_str,
            "daily_predictions": daily_predictions[::-1],
            "prediction_accuracy": round(prediction_accuracy, 2),
            "validation_accuracy": round(val_acc * 100, 2),
            "recent_stats": {
                "total_days": len(daily_predictions),
                "up_days": up_days,
                "down_days": down_days,
                "flat_days": flat_days,
                "avg_price_change": round(avg_change, 2),
                "correct_predictions": actual_correct,
                "prediction_accuracy": round(prediction_accuracy, 2)
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_history(code: str = None):
    """获取历史预测记录"""
    try:
        history = prediction_manager.get_history(code)
        
        # 格式化历史记录数据
        formatted_history = []
        for item in history:
            formatted_item = {
                "code": item["code"],
                "name": item["name"],
                "prediction_date": item["prediction_date"],
                "prediction": "上涨" if item["signal"] == 1 else "下跌",
                "probability": float(item["accuracy"]),
                "asset_type": "E"  # 默认股票类型
            }
            
            # 格式化模型参数
            try:
                params = json.loads(item.get("window_sizes", "[]"))
                if params:
                    formatted_item["model_params"] = f"窗口期: {params}"
                else:
                    formatted_item["model_params"] = "默认参数"
            except:
                formatted_item["model_params"] = str(item.get("window_sizes", "默认参数"))
            
            formatted_history.append(formatted_item)
        
        return {"history": formatted_history}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)