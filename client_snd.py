import socket
import cv2
import glob
import time
import struct
import os

def send_image(sock, image_path):
    image = cv2.imread(image_path)
    image_bytes = cv2.imencode('jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90])[1].tobytes()
    length = len(image_bytes)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(image_bytes)

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('192.168.1.101', 60000)) # 服务器IP地址

    # image_paths = glob.glob('test_images/*.png')  # 替换自己的路径
    image_paths = sorted(glob.glob('data/kodak/*.png'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    for image_path in image_paths:
        send_image(s, image_path)
        time.sleep(1)

    s.close()
