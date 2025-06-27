import os
import time

import torch
import numpy as np
import argparse
import torch.nn as nn
from utils import seed_torch
from net.network import WITT
import socket
import pickle
import cv2
import glob

parser = argparse.ArgumentParser(description='WITT Inference')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet, default is WITT')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name, default is DIV2K')
parser.add_argument('--testset', type=str, default='CLIC21',
                    choices=['kodak', 'CLIC21'],
                    help='test dataset name, default is CLIC21')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics, default is MSE')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh', 'rician'],
                    help='wireless channel model, default is awgn')
parser.add_argument('--C', type=int, default=32,
                    help='bottleneck dimension, default is 32')
parser.add_argument('--multiple-snr', type=str, default='12',
                    help='random or fixed snr, default is 12')
args = parser.parse_args()

class ConfigReceiver():
    seed = 1024
    pass_channel = False
    CUDA = True
    device = torch.device("cuda:0")
    logger = None
    image_dims = (3, 256, 256)
    batch_size = 1
    downsample = 4
    snr = 12
    model = 'WITT'
    test_data_dir = "./data/CLIC21/"
    model_path = "./WITT_model/WITT_AWGN_DIV2K_fixed_snr10_psnr_C32.model"
    sent_dir = './static/saved/sent_image/'
    save_dir1 = './static/saved/semantic_communication/'
    save_dir2 = './static/saved/traditional_communication/'

    host = '0.0.0.0'
    port = 60000

    encoder_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
        C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    decoder_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
        C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )

class Receiver():
    def __init__(self, config: ConfigReceiver):
        self.config = config
        self.net = self._load_model()
        self.H, self.W = 0, 0
        self._prepare_save_dirs()
        # self.conn = self._setup_socket()

        seed_torch()
        torch.manual_seed(seed=config.seed)

    def _load_model(self):
        net = WITT(args, self.config)
        net.load_state_dict(torch.load(self.config.model_path), strict=True)
        net = net.cuda() if self.config.CUDA else net.cpu()
        net.eval()
        return net

    def _prepare_save_dirs(self):
        for save_dir in [self.config.save_dir1, self.config.save_dir2]:
            os.makedirs(save_dir, exist_ok=True)
            for file in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, file))

    # def _setup_socket(self):
    #     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     server_socket.bind((self.config.host, self.config.port))
    #     server_socket.listen(5)
    #     print(f"Listening on {self.config.host}:{self.config.port}")
    #     conn, addr = server_socket.accept()
    #     print(f"Connected by {addr}")
    #     return conn

    def _decode(self, data):
        feature_quantized, min_val, max_val = data['feature']
        feature = self._dequantize(feature_quantized, min_val, max_val)
        # feature = data['feature']
        H_new, W_new = data['size']
        img_data = data['image']
        img_data_size=int(len(img_data)/1024)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)


        with torch.no_grad():
            if H_new != self.H or W_new != self.W:
                self.net.decoder.update_resolution(
                    H_new // (2 ** self.config.downsample),
                    W_new // (2 ** self.config.downsample)
                )
                self.H, self.W = H_new, W_new

            recon_image = self.net.decoder(x=feature, snr=12, model=self.config.model)
            recovery_image = recon_image.clamp(0., 1.).cpu().numpy()
            recovery_image = (recovery_image[0] * 255).astype(np.uint8)
            recovery_image = np.transpose(recovery_image, (1, 2, 0))
            recovery_image = cv2.cvtColor(recovery_image, cv2.COLOR_RGB2BGR)

        return recovery_image, img, feature_quantized,img_data_size

    def _dequantize(self, tensor_quantized, min_val, max_val):
        """
        将量化后的 8 位整数反量化为浮动数据。
        :param tensor_quantized: 量化后的 8 位整数张量
        :param min_val: 原始数据的最小值，用于恢复
        :param max_val: 原始数据的最大值，用于恢复
        :return: 反量化后的浮动数据张量
        """
        # 将量化的整数数据转换为浮动数值，并映射回 [0, 1]
        tensor_normalized = tensor_quantized.astype(np.float32) / 255.0

        # 使用反量化公式恢复到原始范围
        tensor_dequantized = tensor_normalized * (max_val - min_val) + min_val

        # 返回反量化后的浮动数据（可以转换为 tensor）
        return torch.tensor(tensor_dequantized).cuda()
    def _add_gaussian_noise(self, image, snr_db):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        signal_pwr = np.mean(image ** 2)
        noise_pwr = signal_pwr / (10 ** (snr_db / 10))

        noise = np.random.normal(0, np.sqrt(noise_pwr), image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
        return image_bgr




if __name__ == '__main__':
    seed_torch()
    torch.manual_seed(seed=ConfigReceiver.seed)

    net = WITT(args, ConfigReceiver)
    net.load_state_dict(torch.load(ConfigReceiver.model_path), strict=True)
    net = net.cuda()
    net.eval()
    H, W = 0, 0
    if not os.path.exists(ConfigReceiver.save_dir1):
        os.makedirs(ConfigReceiver.save_dir1)
    else:
        image_list = glob.glob(os.path.join(ConfigReceiver.save_dir1, '*'))
        for image in image_list:
            os.remove(image)
    if not os.path.exists(ConfigReceiver.save_dir2):
        os.makedirs(ConfigReceiver.save_dir2)
    else:
        image_list = glob.glob(os.path.join(ConfigReceiver.save_dir2, '*'))
        for image in image_list:
            os.remove(image)

    local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_socket.bind(('0.0.0.0', 60000))
    local_socket.listen(5)
    conn, addr = local_socket.accept()
    print('Connected by', addr)

    counter = 1
    while True:
        data_length_bytes = conn.recv(4)
        if not data_length_bytes:
            continue
        data_length = int.from_bytes(data_length_bytes, byteorder='big')
        # print("data_length:", data_length)

        received = bytearray()
        while len(received) < data_length:
            chunk = conn.recv(4096)
            if not chunk:
                break
            received.extend(chunk)
        # print("Received semantic text")

        data = pickle.loads(received)
        # feature = data['feature']
        # H_new, W_new = data['size']
        img_data = data['image']
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        with torch.no_grad():
            # if H_new != H or W_new != W:
            #     net.decoder.update_resolution(H_new // (2 ** net.downsample), W_new // (2 ** net.downsample))
            #     H = H_new
            #     W = W_new

            # recon_image = net.decoder(x=feature, snr=12, model='WITT')

            # recovery_image = recon_image.clamp(0., 1.).cpu().numpy()
            # recovery_image = np.squeeze(recovery_image, axis=0)
            # recovery_image = (recovery_image * 255).astype(np.uint8)
            # recovery_image = np.transpose(recovery_image, (1, 2, 0))
            # recovery_image = cv2.cvtColor(recovery_image, cv2.COLOR_RGB2BGR)

            recovery_image = img

            conn.sendall('ok'.encode('utf-8'))

            # cv2.imshow('Reconstructed Image', recovery_image)
            # cv2.imshow('Original Image', img)

            cv2.imwrite(os.path.join(ConfigReceiver.save_dir1, '{}.png'.format(counter)), recovery_image)
            cv2.imwrite(os.path.join(ConfigReceiver.save_dir2, '{}.jpg'.format(counter)), img)

            counter += 1
            cv2.waitKey(20)





