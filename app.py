import socket
import torch
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import threading
from receiver import ConfigReceiver, Receiver
from sender import ConfigSender, Sender
import cv2
import webbrowser
import eventlet
eventlet.monkey_patch()
import numpy as np
import os
import time
import pickle
import json
import pyldpc


#将应用模式设置为"发送端"
#使用其配置初始化发送器对象

type = 'sender'
if type == 'receiver':
    receiver = Receiver(ConfigReceiver())
if type == 'sender':
    sender = Sender(ConfigSender())

#初始化Flask应用和SocketIO，用于实时通信
app = Flask(__name__)
socketio = SocketIO(app)
#定义发送端和接收端界面的Web路由
@app.route('/sender')
def show_sender():
    return render_template('sender.html')

@app.route('/receiver')
def show_receiver():
    return render_template('receiver.html')
#计算峰值信噪比（PSNR）来衡量原始图像和处理后图像之间的质量差异
def calculate_psnr(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    psnr_value = cv2.PSNR(img1, img2)
    return psnr_value
# 将传输数据保存到JSON文件的函数，追加到现有数据
def save_to_json(data, json_file='image_data.json'):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            # 读取已有的 JSON 数据
            existing_data = json.load(f)
    else:
        existing_data = []

    # 将新数据添加到现有的数据列表中
    existing_data.append(data)

    # 保存更新后的数据
    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=4)
#处理接收图像、保存图像和计算大小及质量指标
# 通过SocketIO向前端发送数据
def receiver_server():
    for semantic_image, original_image, traditional_image, feature, counter in receiver.receive():
        # semantic_image_path = f"static/saved/semantic_communication/{counter}.png"
        # traditional_image_path = f"static/saved/traditional_communication/{counter}.jpg"
        semantic_image_path = f"static/saved/semantic_communicationdw/{counter}.png"
        traditional_image_path = f"static/saved/traditional_communicationdw/{counter}.jpg"

        cv2.imwrite(semantic_image_path, semantic_image)
        cv2.imwrite(traditional_image_path, traditional_image)


        semantic_feature_size = int(len(feature.cpu().numpy().tobytes()) / 1024)
        traditional_image_size = int(len(cv2.imencode('.jpg', traditional_image, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()) / 1024)

        traditional_image_psnr = calculate_psnr(original_image, traditional_image)
        traditional_image_psnr = round(traditional_image_psnr, 2)#保留两位小数
        semantic_image_psnr = calculate_psnr(original_image, semantic_image)
        semantic_image_psnr = round(semantic_image_psnr, 2)
#通过socketio.emit发送图像路径、大小及PSNR值至前端
        socketio.emit('new_image', {
            'traditional_image_url': traditional_image_path,
            'semantic_image_url': semantic_image_path,
            'traditional_image_size': traditional_image_size,
            'semantic_feature_size': semantic_feature_size,
            'traditional_image_psnr': traditional_image_psnr,
            'semantic_image_psnr': semantic_image_psnr,
        })
        print(f"图片已保存并推送：{semantic_image_path}, {traditional_image_path}")

def sender_server():#选择5G模式
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect(('10.243.75.36', 60000))
    s.connect(('10.243.230.31', 60000))
    # s.connect(('10.129.82.31', 60000))
    # s.connect(('10.129.78.30', 60000))
    counter = 1
    while True:
        with torch.no_grad():#禁用梯度计算，减少内存消耗；
            for batch_idx, input in enumerate(sender.test_loader):#遍历测试数据加载器中的每一批图像数据，进行推理（无需反向传播）
                #该段代码实现图像的编码与量化处理：
                print(f"开始编码第{counter}张图片",time.time())
                input = input.cuda()#将输入数据移至GPU进行加速计算；
                B, _, H_new, W_new = input.shape#获取当前图像的高宽，若与前一次不同，则更新编码器分辨率；
                if H_new != sender.H or W_new != sender.W:
                    sender.net.encoder.update_resolution(H_new, W_new)
                    sender.H, sender.W = H_new, W_new
                feature = sender.net.encoder(x=input, snr=12, model=sender.config.model)#使用神经网络对图像进行编码（含信噪比SNR=12）；

                # 对提取的特征进行量化以压缩数据
                feature_quantized, min_val, max_val = sender._quantize(feature)
                #读取原始图像并将其编码为JPEG格式字节流用于后续传输。
                # original_image_path = os.path.join(sender.config.test_data_dir[0], f'{counter}.png')
                # original_image = cv2.imread(f"data/CLIC21/{counter}.png")
                # original_image = cv2.imread(f"data/kodak/{counter}.png")
                original_image = cv2.imread(f"data/epower/{counter}.png")
                original_image_data = cv2.imencode('.jpg', original_image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

                # noisy_image = sender._add_gaussian_noise(original_image, sender.config.snr)
                # noisy_image_data = cv2.imencode('.jpg', noisy_image, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()

                # feature = None
                time.sleep(2)
                #构建数据字典：将图片序号、尺寸、量化特征和原始图像数据封装到 data 字典；
                data = {
                    'counter': counter,
                    'size': (sender.H, sender.W),
                    'feature': [feature_quantized, min_val, max_val],
                    'image': original_image_data
                }
                bytes_data = pickle.dumps(data)# 序列化数据：使用 pickle.dumps(data) 将字典序列化为字节流；
                data_length = len(bytes_data)# 计算数据长度：获取序列化后数据的字节长度，用于后续网络传输。

                # 前端推送
                print(f"开始推送第{counter}张图片", time.time())#打印当前推送图片序号及时间戳；

                original_image_path = f"static/saved/sent_image/{counter}.png"#构建图片保存路径并使用OpenCV保存图像；

                cv2.imwrite(original_image_path, original_image)

                original_image_size = int(len(original_image_data) / 1024)#计算原始图像和语义特征数据的大小（单位KB）；
                semantic_feature_size = int(len(feature_quantized.tobytes()) / 1024)
                # semantic_feature_size = 0

                s.sendall(data_length.to_bytes(4, byteorder='big'))#通过socket发送数据长度和图像数据；
                s.sendall(bytes_data)

                # time.sleep(2)
                #使用Socket.IO向前端推送图像路径及大小信息；
                socketio.emit('message', {
                    'original_image_url': original_image_path,
                    'semantic_image_url': original_image_path,
                    'original_image_size': original_image_size,
                    'semantic_feature_size': semantic_feature_size,
                })

                counter += 1

                print(f"图片已保存并推送：{original_image_path}", time.time())

                ack = s.recv(4)
                # print(int.from_bytes(ack, byteorder='big'))


# 测试版sender
def sender_server2():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect(('127.0.0.1', 60000))
    s.connect(('10.129.82.31', 60000))
    counter = 1
    while True:
        #依次读取指定路径的图片文件；
        # original_image_path = f"data/CLIC21/{counter}.png"
        # original_image_path = f"data/kodak/{counter}.png"
        original_image_path = f"data/kodak/{counter}.png"
        original_image = cv2.imread(original_image_path)
        #将原始图片保存到另一个路径以模拟处理过程；
        traditional_image_path = f"static/saved/sent_image/{counter}.png"
        cv2.imwrite(traditional_image_path, original_image)

        #模拟图片处理
        print(f"开始编码第{counter}张图片",time.time())
        time.sleep(1)
        #将图片转换为JPEG格式字节流并封装成数据包；
        original_image_data = cv2.imencode('.jpg', original_image, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()
        #构造包含计数器和原始图像数据的字典 data
        data = {
                'counter': counter,
                'image': original_image_data
        }
        counter += 1

        bytes_data = pickle.dumps(data)#使用 pickle.dumps 将数据序列化为字节流；
        #获取字节长度，并将长度信息以 4 字节大端格式发送
        data_length = len(bytes_data)
        s.sendall(data_length.to_bytes(4, byteorder='big'))
        s.sendall(bytes_data)#发送实际的数据字节；
        ack = s.recv(4)#接收服务端确认响应（4 字节）；
        # print(ack.decode('utf-8'))
        #通过 socketio 向前端发送图像路径，用于更新页面显示。
        socketio.emit('message', {
            'original_image_url': traditional_image_path,
        })


#根据类型启动适当的服务器线程
# 在正确的界面上打开Web浏览器
# 在端口5001上运行带有SocketIO的Flask应用
if __name__ == '__main__':
    if type == 'receiver':
        threading.Thread(target=receiver_server, daemon=True).start()
    if type == 'sender':
        threading.Thread(target=sender_server, daemon=True).start()
    webbrowser.open("http://127.0.0.1:5002/{}".format(type))

    socketio.run(app, port=5002, debug=False)

