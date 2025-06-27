import socket
import pickle
import struct
import cv2
import numpy as np

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

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 60001))
    s.listen(1)

    conn, addr = s.accept()
    print(f"服务器 {addr} 已连接")

    while True:
        data = receive_data(conn)
        if data is None:
            break

        semantic_image = data['semantic_image']
        traditional_image = data['traditional_image']
        feature = data['feature']

        cv2.imshow("Semantic Reconstructed", semantic_image)
        cv2.imshow("Traditional Compressed", traditional_image)
        cv2.waitKey(100)


        print(f"语义特征大小: {feature.shape}")

    conn.close()
    s.close()