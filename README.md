# 股票智能预测系统

本项目基于 Python，提供股票数据的智能预测与分析服务，支持 Web API 访问。

---

## 项目简介
本系统集成了数据采集、特征工程、模型训练、预测管理和 Web 服务，旨在为用户提供高效、智能的股票预测解决方案。支持多种数据源扩展，便于二次开发和个性化定制。

## 主要功能
- **股票数据加载与处理**：支持多种数据格式，自动清洗与预处理。
- **特征工程与模型训练**：内置常用特征工程方法，支持自定义特征和多模型训练。
- **预测结果管理与数据库存储**：预测结果自动存储，支持历史查询和分析。
- **Web 服务接口**：基于 FastAPI + Uvicorn，提供 RESTful API，支持在线预测与结果展示。
- **自动端口选择与多进程支持**：便于部署和开发。

## 技术栈
- Python 3.8+
- FastAPI
- Uvicorn
- SQLite（默认缓存数据库）
- 依赖包详见 requirements.txt

## 快速开始
1. 安装依赖：
   ```sh
   pip install -r requirements.txt
   ```
2. 启动服务：
   ```sh
   python main.py
   ```
   可选参数：
   - `--port 端口号` 指定端口
   - `--reload` 开发模式热重载
   - `--auto-port` 自动选择可用端口
   - `--workers` 工作进程数

3. 访问 API 文档：
   http://localhost:8000/docs

## 主要模块说明
- `main.py`：项目主入口，负责参数解析和服务启动。
- `config.py`：系统配置项。
- `data/`：数据加载与预处理相关代码。
- `database/`：预测结果管理与数据库操作。
- `features/`：特征工程相关方法。
- `models/`：模型训练与预测逻辑。
- `utils/`：工具函数集合。
- `web/`：Web 服务接口，核心为 `app.py`。

## 目录结构
```
yc/
    main.py              # 主入口
    config.py            # 配置文件
    requirements.txt     # 依赖
    README.md            # 项目说明
    cache.db             # 缓存数据库
    data/                # 数据加载模块
    database/            # 预测结果管理
    features/            # 特征工程
    models/              # 预测模型
    utils/               # 工具函数
    web/                 # Web 服务
```

## API 示例
- 获取预测结果：
  ```http
  GET /predict?code=xxxx
  ```
- 查看所有支持接口：
  访问 `/docs` 或 `/redoc` 页面。

## 版本管理
- 当前版本：v1.0.0
- 版本标签：v1.0.0

## 常见问题
- **端口被占用**：可使用 `--auto-port` 参数自动选择可用端口。
- **依赖安装失败**：请确认 Python 版本和 pip 已正确安装。
- **API 无法访问**：请检查防火墙设置或端口是否开放。

## 贡献方式
欢迎提交 Issue 或 Pull Request，完善功能和文档。

## 作者
- wanliqin

## License
MIT