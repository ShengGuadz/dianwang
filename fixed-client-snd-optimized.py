from flask import Flask, render_template, url_for, send_from_directory
from flask_socketio import SocketIO
import socket
import cv2
import glob
import time
import struct
import os
import threading
import numpy as np
import pickle
import json  # 用于调试输出

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # 允许跨域连接

# 创建保存图像的目录
os.makedirs('static/saved/sent_image', exist_ok=True)
os.makedirs('templates', exist_ok=True)  # 确保templates目录存在
os.makedirs('static/image', exist_ok=True)  # 确保背景图片目录存在
os.makedirs('static/Setting', exist_ok=True)  # 确保设置图片目录存在

# 检查并创建模板文件
def create_template_if_needed():
    # 先检查模板文件是否存在
    template_path = os.path.join('templates', 'sender_html_1.html')
    
    if not os.path.exists(template_path):
        # 创建一个简单的模板
        html_content = """<!DOCTYPE html>
<html lang="ch">
<head>
    <meta charset="UTF-8">
    <title>Transmitter</title>
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
    <h1>图像语义通信平台 — 发送端</h1>
    
    <div class="image-container">
        <h2>传统图像</h2>
        <div id="container_traditional">
            <img src="https://via.placeholder.com/320x220?text=等待图像" alt="等待传统图像">
        </div>
        <div id="bandwidth-traditional" class="metric-display">带宽: 等待数据...</div>
    </div>
    
    <div class="image-container">
        <h2>语义图像</h2>
        <div id="container_semantic">
            <img src="https://via.placeholder.com/320x220?text=等待图像" alt="等待语义图像">
        </div>
        <div id="bandwidth-sematic" class="metric-display">带宽: 等待数据...</div>
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
        const bandwidthTraditional = document.getElementById("bandwidth-traditional");
        const bandwidthSematic = document.getElementById("bandwidth-sematic");

        socket.on("connect", function() {
            logDebug("已连接到服务器！准备发送图像...");
        });

        socket.on("message", (data) => {
            logDebug("接收到新图像信息: " + JSON.stringify(data).substring(0, 200) + "...");
            
            // 清空容器
            container_traditional.innerHTML = "";
            container_semantic.innerHTML = "";

            // 处理传统图像
            if (data.original_image_url) {
                const originalImg = document.createElement("img");
                originalImg.src = data.original_image_url;
                originalImg.onerror = function() {
                    this.src = 'https://via.placeholder.com/320x220?text=加载失败';
                    logDebug("传统图像加载失败: " + data.original_image_url);
                };
                originalImg.onload = function() {
                    logDebug("传统图像加载成功: " + data.original_image_url);
                };
                container_traditional.appendChild(originalImg);
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
            const traditionalBandwidth = data.original_image_size || 0;
            const semanticBandwidth = data.semantic_feature_size || 0;
            bandwidthTraditional.innerText = `带宽: ${traditionalBandwidth}KBps`;
            bandwidthSematic.innerText = `带宽: ${semanticBandwidth}KBps`;
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
        print(f"已创建默认模板 {template_path}")

# 全局计数器和当前发送的图像信息
counter = 0
current_image_info = None
# 控制显示图像的锁
display_lock = threading.Lock()
display_queue = []

def send_image(sock, image_path):
    global counter, current_image_info, display_queue
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
            
        # 保存发送的图像到静态文件夹
        saved_path = f"static/saved/sent_image/{counter}.png"
        cv2.imwrite(saved_path, image)
        
        # 计算原图大小（KB）
        original_size = os.path.getsize(image_path) / 1024
        
        # 为URL添加时间戳，避免浏览器缓存
        timestamp = int(time.time())
        
        # 直接构建静态文件URL路径而不使用url_for
        original_image_url = f"/static/saved/sent_image/{counter}.png?t={timestamp}"
        semantic_image_url = f"/static/saved/sent_image/{counter}.png?t={timestamp}"
        
        # 更新当前图像信息
        current_image_info = {
            'original_image_url': original_image_url,
            'semantic_image_url': semantic_image_url,
            'original_image_size': round(original_size, 2),
            'semantic_feature_size': round(original_size * 0.3, 2)
        }
        
        # 不立即显示，而是等待服务器处理完成后再显示
        print(f"图像准备好待发送: #{counter}")
        
        # 图像压缩后发送
        image_bytes = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        length = len(image_bytes)
        
        # 添加等待服务器就绪信号
        response = sock.recv(10)
        if response == b'READY' or response == b'SHOW_IMAGE':
            print(f"服务器已准备好接收图像 #{counter}")
        else:
            print(f"收到未知服务器信号: {response}")
        
        # 发送图像数据
        sock.sendall(struct.pack('!I', length))
        sock.sendall(image_bytes)
        print(f"已发送图像数据到服务器 #{counter}")
        
        # 等待服务器确认收到图像
        ack = sock.recv(10)
        if ack == b'GOT_IMAGE':
            print(f"服务器已确认接收图像 #{counter}")
        
        # 等待服务器通知可以显示图像
        show_signal = sock.recv(10)
        if show_signal == b'SHOW_IMAGE':
            print(f"服务器已处理完成，可以显示图像 #{counter}")
            with display_lock:
                display_queue.append(current_image_info)
                socketio.emit('message', current_image_info)
                print(f"已发送图像数据到前端 #{counter}")
        
        counter += 1
        
    except Exception as e:
        print(f"发送图像时出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误堆栈信息

def socket_thread():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("正在连接到服务器...")
        
        # 尝试连接到服务器，如果失败则重试
        server_ip = '127.0.0.1'  # 使用localhost进行本地测试
        retries = 0
        max_retries = 5
        
        while retries < max_retries:
            try:
                s.connect((server_ip, 60000))
                print(f"已连接到服务器 {server_ip}")
                break
            except socket.error as e:
                retries += 1
                print(f"连接服务器失败 (尝试 {retries}/{max_retries}): {e}")
                if retries == max_retries:
                    print("达到最大尝试次数，退出")
                    return
                time.sleep(5)  # 等待5秒后重试
        
        # 检查图像目录是否存在
        image_dir = 'data/CLIC21'
        if not os.path.exists(image_dir):
            print(f"警告: 图像目录 '{image_dir}' 不存在，创建目录并放置测试图像")
            os.makedirs(image_dir, exist_ok=True)
            # 创建一个测试图像
            test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
            cv2.imwrite(f'{image_dir}/0.png', test_image)
        
        # 获取图像路径列表
        image_paths = sorted(glob.glob(f'{image_dir}/*.png'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        if not image_paths:
            print(f"警告: 目录 '{image_dir}' 中没有PNG图像")
            # 创建一个测试图像
            test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
            test_path = f'{image_dir}/0.png'
            cv2.imwrite(test_path, test_image)
            image_paths = [test_path]
        
        print(f"找到 {len(image_paths)} 张图像")
        
        for image_path in image_paths:
            send_image(s, image_path)
            time.sleep(1)  # 每秒发送一张图片
        
        s.close()
        print("发送完成，连接已关闭")
    except Exception as e:
        print(f"连接服务器时发生错误: {e}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    # 清除计数器，重新开始
    global counter
    counter = 0
    return render_template('sender_html_1.html')

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
    
    # 启动socket发送线程
    thread = threading.Thread(target=socket_thread)
    thread.daemon = True
    thread.start()
    
    # 启动Flask应用
    print("发送端服务器已启动，请访问 http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
