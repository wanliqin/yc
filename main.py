#!/usr/bin/env python3
"""
股票智能预测系统 - 主入口文件

使用方法:
    python main.py          # 启动Web服务
    python main.py --help   # 查看帮助信息
    python main.py --port 8001  # 使用不同端口
"""

import argparse
import uvicorn
import os
import sys
import socket

def find_free_port(start_port=8000):
    """查找可用端口"""
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            port += 1
            if port > 9000:
                raise RuntimeError("无法找到可用端口")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股票智能预测系统')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='端口号 (默认: 8000)')
    parser.add_argument('--reload', action='store_true', help='启用热重载 (开发模式)')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数')
    parser.add_argument('--auto-port', action='store_true', help='自动选择可用端口')
    
    args = parser.parse_args()
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 切换到项目目录
    os.chdir(current_dir)
    
    # 处理端口占用问题
    if args.auto_port:
        actual_port = find_free_port(args.port)
        if actual_port != args.port:
            print(f"⚠️  端口 {args.port} 被占用，使用端口 {actual_port}")
    else:
        # 检查端口是否可用
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', args.port))
                actual_port = args.port
        except OSError:
            actual_port = find_free_port(args.port)
            print(f"⚠️  端口 {args.port} 被占用，自动切换到端口 {actual_port}")
    
    print("🚀 启动股票智能预测系统...")
    print(f"📊 访问地址: http://{args.host}:{actual_port}")
    print(f"📖 API文档: http://{args.host}:{actual_port}/docs")
    
    # 启动服务
    import web.app
    uvicorn.run(
        web.app.app,
        host=args.host,
        port=actual_port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )

if __name__ == "__main__":
    main()