# 股票智能预测系统

基于机器学习的股票涨跌预测系统，支持中文股票名称和预测结果持久化存储。

## 功能特性

- ✅ **中文股票名称支持** - 自动获取并显示股票中文名称
- ✅ **结构化代码** - 模块化设计，易于维护和扩展
- ✅ **预测结果保存** - 所有预测结果自动保存到数据库
- ✅ **历史记录查询** - 支持查看历史预测记录
- ✅ **Web界面** - 友好的Web操作界面
- ✅ **智能搜索** - 支持股票代码、中文名称、拼音首字母搜索

## 项目结构

```
yc/
├── data/                    # 数据获取模块
│   └── data_loader.py      # 数据加载器
├── features/               # 特征工程模块
│   └── feature_engineer.py # 特征工程
├── models/                 # 模型模块
│   └── stock_predictor.py  # 股票预测器
├── web/                    # Web接口模块
│   └── app.py             # FastAPI应用
├── database/              # 数据库模块
│   └── prediction_manager.py # 预测管理器
├── config.py              # 配置文件
├── main.py               # 主入口文件
├── requirements.txt      # 依赖列表
└── README.md            # 使用说明
```

## 安装使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 启动Web服务
python main.py

# 或指定端口
python main.py --port 8080 --host 0.0.0.0

# 开发模式（热重载）
python main.py --reload

# 自动选择可用端口
python main.py --auto-port
```

### 3. 访问系统

- Web界面: http://localhost:8000
- API文档: http://localhost:8000/docs

## 使用示例

### 单个股票预测

1. 访问 http://localhost:8000
2. 输入股票代码（如：000001.SZ）或中文名称（如：平安银行）
3. 设置窗口期（如：120,250）
4. 设置调参次数（建议10-50次）
5. 点击"开始预测"

### API调用

```bash
# 单个股票预测
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "code=000001.SZ&win=120,250&trials=30"

# 查看历史记录
curl "http://localhost:8000/history?code=000001.SZ"

# 查看所有历史记录
curl "http://localhost:8000/history"
```

## 数据库结构

系统会自动创建以下数据库表：

- **daily**: 存储股票日线数据
- **stock_info**: 存储股票基本信息（含中文名称）
- **predictions**: 存储预测结果
- **stock_search**: 存储股票搜索信息

## 配置说明

在 `config.py` 中可以修改以下配置：

- `TUSHARE_TOKEN`: Tushare API Token
- `DATABASE_PATH`: 数据库文件路径
- `DEFAULT_WINDOW_SIZES`: 默认窗口大小
- `DEFAULT_TRIALS`: 默认调参次数
- `MIN_DATA_POINTS`: 最小数据点数

## 注意事项

1. 首次运行时会自动下载股票数据，可能需要一些时间
2. 确保网络连接正常以获取最新数据
3. 预测结果仅供参考，投资有风险
4. 系统会自动处理端口占用问题

## 技术支持

如有问题，请检查：
- 网络连接是否正常
- Tushare Token是否有效
- 数据库文件是否有写入权限
- 端口是否被占用（系统会自动处理）

## 更新日志

- v1.0.0: 初始版本，支持基本预测功能
- v1.1.0: 修复FeatureEngineer缺少create_features方法的问题
- v1.1.1: 优化项目结构，删除冗余文件