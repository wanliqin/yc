from typing import Optional

"""
Web应用的模型缓存管理接口
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from utils.model_cache import get_model_cache
import json

# 创建缓存管理路由
cache_router = APIRouter()

@cache_router.get("/cache/stats")
def get_cache_stats():
    """获取缓存统计信息"""
    try:
        cache = get_model_cache()
        stats = cache.get_cache_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        return {"success": False, "error": str(e)}

@cache_router.post("/cache/cleanup")
def cleanup_cache():
    """清理缓存"""
    try:
        cache = get_model_cache()
        cache.cleanup_cache()
        return {"success": True, "message": "缓存清理完成"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@cache_router.post("/cache/invalidate")
def invalidate_cache(stock_code: Optional[str] = None):
    """失效缓存"""
    try:
        cache = get_model_cache()
        cache.invalidate_cache(stock_code)
        message = f"股票 {stock_code} 的缓存已失效" if stock_code else "所有缓存已清空"
        return {"success": True, "message": message}
    except Exception as e:
        return {"success": False, "error": str(e)}

@cache_router.get("/cache/dashboard", response_class=HTMLResponse)
def cache_dashboard():
    """缓存管理仪表板"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🗂️ 模型缓存管理</title>
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
                max-width: 1200px;
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
            
            .card {
                background: white;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .stat-card {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(240, 147, 251, 0.3);
            }
            
            .stat-value {
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 5px;
            }
            
            .stat-label {
                font-size: 1rem;
                opacity: 0.9;
            }
            
            .action-buttons {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                margin-bottom: 30px;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            .btn.danger {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
            }
            
            .btn.danger:hover {
                box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
            }
            
            .models-table {
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .models-table th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }
            
            .models-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }
            
            .models-table tr:hover {
                background: #f8f9fa;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                font-size: 1.2rem;
                color: #667eea;
            }
            
            .success {
                background: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border: 1px solid #c3e6cb;
            }
            
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border: 1px solid #f5c6cb;
            }
            
            .refresh-time {
                text-align: center;
                color: #666;
                font-size: 0.9rem;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-database"></i> 模型缓存管理</h1>
                <p>监控和管理AI模型缓存系统</p>
            </div>
            
            <div class="card">
                <h3><i class="fas fa-chart-bar"></i> 缓存统计</h3>
                <div id="statsGrid" class="stats-grid">
                    <div class="loading">
                        <i class="fas fa-spinner fa-spin"></i> 加载统计数据...
                    </div>
                </div>
                
                <div class="action-buttons">
                    <button class="btn" onclick="refreshStats()">
                        <i class="fas fa-refresh"></i> 刷新统计
                    </button>
                    <button class="btn" onclick="cleanupCache()">
                        <i class="fas fa-broom"></i> 清理过期缓存
                    </button>
                    <button class="btn danger" onclick="invalidateAllCache()">
                        <i class="fas fa-trash"></i> 清空所有缓存
                    </button>
                </div>
                
                <div id="messageArea"></div>
            </div>
            
            <div class="card">
                <h3><i class="fas fa-list"></i> 活跃模型缓存</h3>
                <div id="modelsTable">
                    <div class="loading">
                        <i class="fas fa-spinner fa-spin"></i> 加载模型列表...
                    </div>
                </div>
            </div>
            
            <div class="refresh-time" id="refreshTime"></div>
        </div>

        <script>
            let refreshInterval;
            
            async function fetchCacheStats() {
                try {
                    const response = await fetch('/cache/stats');
                    const data = await response.json();
                    
                    if (data.success) {
                        displayStats(data.stats);
                    } else {
                        showMessage('error', '获取统计数据失败: ' + data.error);
                    }
                } catch (error) {
                    showMessage('error', '网络错误: ' + error.message);
                }
            }
            
            function displayStats(stats) {
                const statsGrid = document.getElementById('statsGrid');
                
                const hitRate = stats.hit_rate_percent || 0;
                const hitRateColor = hitRate >= 80 ? '#28a745' : hitRate >= 50 ? '#ffc107' : '#dc3545';
                
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${stats.total_models}</div>
                        <div class="stat-label">缓存模型数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.total_size_mb}MB</div>
                        <div class="stat-label">缓存大小</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, ${hitRateColor} 0%, ${hitRateColor}dd 100%);">
                        <div class="stat-value">${hitRate.toFixed(1)}%</div>
                        <div class="stat-label">命中率</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.recent_hits}</div>
                        <div class="stat-label">近期命中</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.recent_misses}</div>
                        <div class="stat-label">近期未命中</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.avg_access_count.toFixed(1)}</div>
                        <div class="stat-label">平均访问次数</div>
                    </div>
                `;
                
                // 显示活跃模型
                displayMostUsedModels(stats.most_used_models);
            }
            
            function displayMostUsedModels(models) {
                const modelsTable = document.getElementById('modelsTable');
                
                if (!models || models.length === 0) {
                    modelsTable.innerHTML = '<div style="text-align: center; padding: 40px; color: #666;">暂无缓存模型</div>';
                    return;
                }
                
                let tableHTML = `
                    <table class="models-table">
                        <thead>
                            <tr>
                                <th><i class="fas fa-building"></i> 股票代码</th>
                                <th><i class="fas fa-mouse-pointer"></i> 访问次数</th>
                                <th><i class="fas fa-clock"></i> 最后访问</th>
                                <th><i class="fas fa-cogs"></i> 操作</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                models.forEach(model => {
                    const lastAccessed = new Date(model.last_accessed).toLocaleString();
                    tableHTML += `
                        <tr>
                            <td><strong>${model.stock_code}</strong></td>
                            <td>${model.access_count}</td>
                            <td>${lastAccessed}</td>
                            <td>
                                <button class="btn danger" style="padding: 5px 15px; font-size: 12px;" 
                                        onclick="invalidateStockCache('${model.stock_code}')">
                                    <i class="fas fa-trash"></i> 删除
                                </button>
                            </td>
                        </tr>
                    `;
                });
                
                tableHTML += '</tbody></table>';
                modelsTable.innerHTML = tableHTML;
            }
            
            async function cleanupCache() {
                try {
                    const response = await fetch('/cache/cleanup', { method: 'POST' });
                    const data = await response.json();
                    
                    if (data.success) {
                        showMessage('success', data.message);
                        await fetchCacheStats();
                    } else {
                        showMessage('error', data.error);
                    }
                } catch (error) {
                    showMessage('error', '清理失败: ' + error.message);
                }
            }
            
            async function invalidateAllCache() {
                if (!confirm('确定要清空所有缓存吗？此操作不可撤销！')) {
                    return;
                }
                
                try {
                    const response = await fetch('/cache/invalidate', { method: 'POST' });
                    const data = await response.json();
                    
                    if (data.success) {
                        showMessage('success', data.message);
                        await fetchCacheStats();
                    } else {
                        showMessage('error', data.error);
                    }
                } catch (error) {
                    showMessage('error', '清空失败: ' + error.message);
                }
            }
            
            async function invalidateStockCache(stockCode) {
                if (!confirm(`确定要删除 ${stockCode} 的缓存吗？`)) {
                    return;
                }
                
                try {
                    const response = await fetch(`/cache/invalidate?stock_code=${stockCode}`, { method: 'POST' });
                    const data = await response.json();
                    
                    if (data.success) {
                        showMessage('success', data.message);
                        await fetchCacheStats();
                    } else {
                        showMessage('error', data.error);
                    }
                } catch (error) {
                    showMessage('error', '删除失败: ' + error.message);
                }
            }
            
            function showMessage(type, message) {
                const messageArea = document.getElementById('messageArea');
                messageArea.innerHTML = `<div class="${type}"><i class="fas fa-${type === 'success' ? 'check' : 'exclamation-triangle'}"></i> ${message}</div>`;
                
                setTimeout(() => {
                    messageArea.innerHTML = '';
                }, 5000);
            }
            
            function refreshStats() {
                fetchCacheStats();
                updateRefreshTime();
            }
            
            function updateRefreshTime() {
                const refreshTime = document.getElementById('refreshTime');
                refreshTime.textContent = `最后更新: ${new Date().toLocaleString()}`;
            }
            
            // 初始化
            window.addEventListener('load', function() {
                fetchCacheStats();
                updateRefreshTime();
                
                // 每30秒自动刷新
                refreshInterval = setInterval(() => {
                    fetchCacheStats();
                    updateRefreshTime();
                }, 30000);
            });
            
            // 页面卸载时清理定时器
            window.addEventListener('beforeunload', function() {
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                }
            });
        </script>
    </body>
    </html>
    """
