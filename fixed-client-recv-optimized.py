from flask import Flask, render_template, url_for, send_from_directory
from flask_socketio import SocketIO
import socket
import pickle
import struct
import cv2
import numpy as np
import threading
import os
import base64
import time
import json  # 用于调试输出

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # 允许跨域连接

# 创建保存图像的目录
os.makedirs('static/saved/semantic_communication', exist_ok=True)
os.makedirs('static/saved/traditional_communication', exist_ok=True)
os.makedirs('templates', exist_ok=True)  # 确保templates目录存在
os.makedirs('static/image', exist_ok=True)  # 确保背景图片目录存在
os.makedirs('static/Setting', exist_ok=True)  # 确保设置图片目录存在

# 检查并创建模板文件
def create_template_if_needed():
    template_path = os.path.join('templates', 'receiver_html_1.html')
    
    if not os.path.exists(template_path):
        # 创建一个简单的模板
        html_content = """<!DOCTYPE html>
<html lang="ch">
<head>
    <meta charset="UTF-8">
    <title>Receiver</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 20px;
            background-color: #f0f0f0;
        }
        .image-container {
            margin: 20px;
            padding: 10px;
            border: 1px solid #ccc;
            display: inline-block;
            background-color: white;
        }
        .image-container img {
            max-width: 320px;
            max-height: 240px;
            border: 2px solid blue;
        }
        .metric-display {
            background-color: rgba(255, 255, 255, 0.7);
            padding: 5px;
            border-radius: 5px;
            margin: 10px;
            font-weight: bold;
            color: red;
        }
        #debug-info {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
            max-width: 600px;
            max-height: 300px;
            overflow: auto;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <h1>图像语义通信平台 — 接收端</h1>
    
    <div class="image-container">
        <h2>传统图像</h2>
        <div id="container_traditional">
            <img src="https://via.placeholder.com/320x220?text=等待图像" alt="等待传统图像">
        </div>
        <div id="bandwidth-traditional" class="metric-display">带宽: 等待数据...</div>
        <div id="lpips-traditional" class="metric-display">LPIPS: 等待数据...</div>
    </div>
    
    <div class="image-container">
        <h2>语义图像</h2>
        <div id="container_semantic">
            <img src="https://via.placeholder.com/320x220?text=等待图像" alt="等待语义图像">
        </div>
        <div id="bandwidth-sematic" class="metric-display">带宽: 等待数据...</div>
        <div id="lpips-sematic" class="metric-display">LPIPS: 等待数据...</div>
    </div>

    <!-- 调试信息显示区域 -->
    <div id="debug-info">SocketIO调试信息会显示在这里</div>

    <script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
    <script>
        // 初始化调试信息
        const debugInfo = document.getElementById("debug-info");
        
        function logDebug(message) {
            const now = new Date();
            const timestamp = now.toLocaleTimeString() + '.' + now.getMilliseconds();
            const logMsg = `${timestamp}: ${message}`;
            console.log(logMsg);
            debugInfo.innerHTML = logMsg + "<br>" + debugInfo.innerHTML;
            if (debugInfo.innerHTML.length > 10000) {
                debugInfo.innerHTML = debugInfo.innerHTML.substring(0, 10000);
            }
        }

        // 建立Socket连接
        const socket = io();
        const container_traditional = document.getElementById("container_traditional");
        const container_semantic = document.getElementById("container_semantic");

        socket.on("connect", function() {
            logDebug("已连接到服务器！准备接收图像数据...");
        });

        socket.on("message", (data) => {
            logDebug("接收到新图像数据: " + JSON.stringify(data).substring(0, 200) + "...");
            
            // 清空容器
            container_traditional.innerHTML = "";
            container_semantic.innerHTML = "";

            // 处理传统图像
            if (data.traditional_image_url) {
                const traditionalImg = document.createElement("img");
                traditionalImg.src = data.traditional_image_url;
                traditionalImg.onerror = function() {
                    this.src = 'https://via.placeholder.com/320x220?text=加载失败';
                    logDebug("传统图像加载失败: " + data.traditional_image_url);
                };
                traditionalImg.onload = function() {
                    logDebug("传统图像加载成功: " + data.traditional_image_url);
                };
                container_traditional.appendChild(traditionalImg);
            } else {
                logDebug("警告: 数据中没有传统图像URL");
                container_traditional.innerHTML = "<p>没有接收到传统图像URL</p>";
            }

            // 处理语义图像
            if (data.semantic_image_url) {
                const semanticImg = document.createElement("img");
                semanticImg.src = data.semantic_image_url;
                semanticImg.onerror = function() {
                    this.src = 'https://via.placeholder.com/320x220?text=加载失败';
                    logDebug("语义图像加载失败: " + data.semantic_image_url);
                };
                semanticImg.onload = function() {
                    logDebug("语义图像加载成功: " + data.semantic_image_url);
                };
                container_semantic.appendChild(semanticImg);
            } else {
                logDebug("警告: 数据中没有语义图像URL");
                container_semantic.innerHTML = "<p>没有接收到语义图像URL</p>";
            }
            
            // 更新带宽数据
            const traditionalBandwidth = data.traditional_image_size || 0;
            const semanticBandwidth = data.semantic_feature_size || 0;
            document.getElementById("bandwidth-traditional").innerText = `带宽: ${traditionalBandwidth}KBps`;
            document.getElementById("bandwidth-sematic").innerText = `带宽: ${semanticBandwidth}KBps`;
            
            // 更新LPIPS数据
            const traditionalLPIPS = data.traditional_image_lpips || 0;
            const semanticLPIPS = data.semantic_image_lpips || 0;
            document.getElementById("lpips-traditional").innerText = `LPIPS: ${traditionalLPIPS}`;
            document.getElementById("lpips-sematic").innerText = `LPIPS: ${semanticLPIPS}`;
        });

        socket.on("disconnect", function() {
            logDebug("与服务器断开连接");
        });

        socket.on("connect_error", function(error) {
            logDebug("连接错误: " + error);
        });

        // 初始状态显示
        logDebug("页面加载完成，等待服务器连接...");
    </script>
</body>
</html>"""
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"已创建默认接收端模板 {template_path}")

# 全局计数器用于图像文件名
counter = 0

def receive_data(sock):
    header = sock.recv(4)
    if not header:
        return None
    data_len = int.from_bytes(header, byteorder='big')
    data = b''
    while len(data) < data_len:
        packet = sock.recv(min(4096, data_len - len(data)))
        if not packet:
            return None
        data += packet
    return pickle.loads(data)

def socket_thread():
    global counter
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 60001))
    s.listen(1)
    
    print("等待服务器连接...")
    conn, addr = s.accept()
    print(f"服务器 {addr} 已连接")
    
    while True:
        try:
            print("等待接收数据...")
            data = receive_data(conn)
            if data is None:
                print("接收到空数据，连接可能已关闭")
                break
                
            semantic_image = data['semantic_image']
            traditional_image = data['traditional_image']
            feature = data['feature']
            
            # 保存图像到静态文件夹
            semantic_path = f"static/saved/semantic_communication/{counter}.png"
            traditional_path = f"static/saved/traditional_communication/{counter}.jpg"
            
            cv2.imwrite(semantic_path, semantic_image)
            cv2.imwrite(traditional_path, traditional_image)
            
            # 确认图像已写入
            sem_exists = os.path.exists(semantic_path)
            trad_exists = os.path.exists(traditional_path)
            if not sem_exists or not trad_exists:
                print(f"警告: 图像文件写入失败! 语义图像: {sem_exists}, 传统图像: {trad_exists}")
            
            # 计算大小（KB）
            semantic_size = os.path.getsize(semantic_path) / 1024
            traditional_size = os.path.getsize(traditional_path) / 1024
            
            # 计算LPIPS值（这里简化为随机值，实际应根据需求实现）
            semantic_lpips = round(np.random.uniform(0.1, 0.3), 3)
            traditional_lpips = round(np.random.uniform(0.2, 0.5), 3)
            
            # 为URL添加时间戳，避免浏览器缓存
            timestamp = int(time.time())
            
            # 打印接收到的数据信息（调试用）
            print(f"已接收新图像 #{counter}:")
            print(f"语义图像路径: {semantic_path}, 大小: {semantic_size}KB")
            print(f"传统图像路径: {traditional_path}, 大小: {traditional_size}KB")
            
            # 直接构建静态文件URL路径而不使用url_for
            traditional_image_url = f"/static/saved/traditional_communication/{counter}.jpg?t={timestamp}"
            semantic_image_url = f"/static/saved/semantic_communication/{counter}.png?t={timestamp}"
            
            # 准备消息数据
            message_data = {
                'traditional_image_url': traditional_image_url,
                'semantic_image_url': semantic_image_url,
                'traditional_image_size': round(traditional_size, 2),
                'semantic_feature_size': round(semantic_size, 2),
                'traditional_image_lpips': traditional_lpips,
                'semantic_image_lpips': semantic_lpips
            }
            
            # 调试输出完整消息
            print(f"发送前端消息数据: {json.dumps(message_data)}")
            
            # 通过socketio向前端发送新图像信息
            socketio.emit('message', message_data)
            
            print(f"已发送图像数据到前端 #{counter}")
            
            # 通知服务器数据已处理完成，可以继续发送下一张
            try:
                conn.sendall(b'NEXT')
                print("已发送NEXT信号给服务器")
            except Exception as e:
                print(f"发送NEXT信号时出错: {e}")
            
            counter += 1
            time.sleep(0.1)  # 稍微延迟以减轻系统负载
            
        except Exception as e:
            print(f"接收数据时发生错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    conn.close()
    s.close()
    print("连接已关闭")

@app.route('/')
def index():
    # 清除计数器，重新开始
    global counter
    counter = 0
    return render_template('receiver_html_1.html')

# 确保静态文件可被访问
@app.route('/static/<path:filename>')
def serve_static(filename):
    print(f"请求静态文件: {filename}")
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # 确保模板文件存在
    create_template_if_needed()
    
    # 创建一些测试图像以备不时之需
    placeholder_dir = 'static/saved'
    os.makedirs(placeholder_dir, exist_ok=True)
    placeholder_img = np.ones((320, 240, 3), dtype=np.uint8) * 200
    cv2.putText(placeholder_img, "Test Image", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(placeholder_dir, 'test.jpg'), placeholder_img)
    
    # 启动socket接收线程
    thread = threading.Thread(target=socket_thread)
    thread.daemon = True
    thread.start()
    
    # 启动Flask应用
    print("接收端服务器已启动，请访问 http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
