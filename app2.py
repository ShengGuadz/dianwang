import socket
from flask import Flask, render_template
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
import lpips
lpips_fn = lpips.LPIPS(net='alex')

type = 'receiver'
if type == 'receiver':
    receiver = Receiver(ConfigReceiver())
if type == 'sender':
    sender = Sender(ConfigSender())

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/receiver')
def show_receiver():
    return render_template('receiver.html')

@app.route('/sender')
def show_sender():
    return render_template('sender.html')

def calculate_psnr(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    psnr_value = cv2.PSNR(img1, img2)
    return psnr_value

def calculate_lpips(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.uint8)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.uint8)
    lpips_value = lpips_fn.forward(lpips.im2tensor(img1), lpips.im2tensor(img2)).item()


    # lpips_value = lpips_fn(img1, img2, normalize=True).item()
    return lpips_value

def receiver_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind(('127.0.0.1', 60000))
    s.bind(('0.0.0.0', 60000))
    s.listen(5)
    conn, addr = s.accept()
    print('Connected by ', addr)

    while True:
        time1 = time.time()
        data_length_bytes = conn.recv(4)
        if not data_length_bytes:
            continue
        data_length = int.from_bytes(data_length_bytes, byteorder='big')
        received = bytearray()
        while len(received) < data_length:
            chunk = conn.recv(4096)
            if not chunk:
                break
            received.extend(chunk)
        time2 = time.time()
        print('接收时间：',time2-time1)

        data = pickle.loads(received)
        counter = data['counter']

        print(f"开始解码第{counter}张图片", time.time())

        # img_data = data['image']
        # original_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        semantic_image, original_image, feature_quantized,traditional_image_size = receiver._decode(data)
        traditional_image = receiver._add_gaussian_noise(original_image, ConfigReceiver.snr)



        print(f"开始推送第{counter}张图片", time.time())

        semantic_image_path = f"static/saved/semantic_communication/{counter}.png"
        traditional_image_path = f"static/saved/traditional_communication/{counter}.png"

        cv2.imwrite(semantic_image_path, semantic_image)
        cv2.imwrite(traditional_image_path, traditional_image)

        semantic_feature_size = int(len(feature_quantized.tobytes()) / 1024)
        # semantic_feature_size = 0
        # traditional_image_size = int(len(cv2.imencode('.jpg', original_image, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()) / 1024)

        traditional_image_psnr = calculate_psnr(original_image, traditional_image)
        traditional_image_lpips = calculate_lpips(original_image, traditional_image)
        # traditional_image_psnr = round(traditional_image_psnr, 2)
        traditional_image_lpips = round(traditional_image_lpips, 2)
        # semantic_image_psnr = calculate_psnr(original_image, semantic_image)
        semantic_image_lpips = calculate_lpips(original_image, semantic_image)
        # semantic_image_psnr = round(semantic_image_psnr, 2)
        semantic_image_lpips = round(semantic_image_lpips, 2)
        original_image_size = int(
            len(cv2.imencode('.jpg', original_image, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()) / 1024)
        traditional_compression_ratio = round((original_image_size-traditional_image_size)*100 / original_image_size, 2)
        semantic_compression_ratio = round((original_image_size-semantic_feature_size)*100 / original_image_size, 2)
        traditional_throughout=round( 102400/ traditional_image_size, 2)
        semantic_throughout=round( 102400/ semantic_feature_size, 2)

        socketio.emit('message', {
            'traditional_image_url': traditional_image_path,
            'semantic_image_url': semantic_image_path,
            'traditional_image_size': traditional_image_size,
            'semantic_feature_size': semantic_feature_size,
            # 'traditional_image_psnr': traditional_image_psnr,
            # 'semantic_image_psnr': semantic_image_psnr,k
            'traditional_image_lpips': traditional_image_lpips,
            'semantic_image_lpips': semantic_image_lpips,
            'traditional_compression_ratio': traditional_compression_ratio,
            'semantic_compression_ratio': semantic_compression_ratio,
            'traditional_throughout':traditional_throughout,
            'semantic_throughout':semantic_throughout,
        })

        print(f"图片已保存并推送： {traditional_image_path}", time.time())
        ack = (1).to_bytes(4, byteorder='big')
        conn.sendall(ack)






if __name__ == '__main__':
    if type == 'receiver':
        threading.Thread(target=receiver_server, daemon=True).start()
    webbrowser.open("http://127.0.0.1:5002/{}".format(type))

    socketio.run(app, port=5002, debug=False)

