import os
import time
from data.datasets import get_loader
from loss.distortion import *
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
import struct

parser = argparse.ArgumentParser(description='WITT Inference')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet, default is WITT')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name, default is DIV2K')
parser.add_argument('--testset', type=str, default='kodak',
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


class Config():
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    logger = None
    image_dims = (3, 256, 256)
    batch_size = 1
    downsample = 4
    snr = 12
    model = 'WITT'
    test_data_dir = ["./data/CLIC21/"]
    model_path = "./WITT_model/WITT_AWGN_DIV2K_fixed_snr10_psnr_C32.model"
    sent_dir = './static/saved/sent_image/'
    save_dir1 = './static/saved/semantic_communication/'
    save_dir2 = './static/saved/traditional_communication/'


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

class Inference():
    def __init__(self, config: Config):
        self.config = config
        self.net = self._load_model()
        self.H, self.W = 0, 0
        self._prepare_save_dirs()
        self.test_loader = get_loader(args, self.config)

        seed_torch()
        torch.manual_seed(seed=config.seed)

    def _load_model(self):
        net = WITT(args, self.config)
        net.load_state_dict(torch.load(self.config.model_path), strict=True)
        net = net.cuda() if self.config.CUDA else net.cpu()
        net.eval()
        return net

    def _prepare_save_dirs(self):
        os.makedirs(self.config.sent_dir, exist_ok=True)
        for file in os.listdir(self.config.sent_dir):
            os.remove(os.path.join(self.config.sent_dir, file))
        for save_dir in [self.config.save_dir1, self.config.save_dir2]:
            os.makedirs(save_dir, exist_ok=True)
            for file in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, file))

    def _add_gaussian_noise(self, image, snr_db):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        signal_pwr = np.mean(image ** 2)
        noise_pwr = signal_pwr / (10 ** (snr_db / 10))

        noise = np.random.normal(0, np.sqrt(noise_pwr), image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
        return image_bgr

    def _quantize(self, tensor, num_bits=8):
        """
        将一个张量进行量化，将浮点数转为 8 位整数（0-255）并存储。
        :param tensor: 输入的张量（3D Tensor）
        :param num_bits: 量化的比特数，默认为8
        :return: 量化后的 8 位整数
        """
        # 获取张量的形状
        shape = tensor.shape
        # 将 tensor 移到 CPU 并转换为 numpy 数组
        tensor = tensor.cpu().detach().numpy()

        # 归一化到 [0, 1] 范围
        min_val = tensor.min()
        max_val = tensor.max()
        tensor_normalized = (tensor - min_val) / (max_val - min_val)

        # 量化到 [0, 255] 范围
        tensor_quantized = np.round(tensor_normalized * 255).astype(np.uint8)

        # 返回量化后的数据和用于反量化的参数
        return tensor_quantized, min_val, max_val

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

    def infer(self, input_image):
        with torch.no_grad():
            input_image = input_image.cuda() if self.config.CUDA else input_image.cpu()
            B, _, H_new, W_new = input_image.shape
            if H_new != self.H or W_new != self.W:
                self.net.encoder.update_resolution(H_new, W_new)
                self.net.decoder.update_resolution(H_new // (2 ** self.config.downsample), W_new // (2 ** self.config.downsample))
                self.H, self.W = H_new, W_new

            feature = self.net.encoder(x=input_image, snr=12, model=self.config.model)
            feature_quantized, min_val, max_val = self._quantize(feature)

            feature_new = self._dequantize(feature_quantized, min_val, max_val)

            recon_image = self.net.decoder(x=feature_new, snr=12, model=self.config.model)
            recovery_image = recon_image.clamp(0., 1.).cpu().numpy()
            recovery_image = (recovery_image[0] * 255).astype(np.uint8)
            recovery_image = np.transpose(recovery_image, (1, 2, 0))
            recovery_image = cv2.cvtColor(recovery_image, cv2.COLOR_RGB2BGR)

        return feature_quantized, recovery_image

    def load_image_as_tensor(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W, _ = image.shape
        H_new = H - (H % 128)
        W_new = W - (W % 128)
        image = image_rgb[:H_new, :W_new, :]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image).unsqueeze(0).cuda()
        return tensor

    def receive_image(self, conn):
        length_data = conn.recv(4)
        if not length_data:
            return None

        length = struct.unpack('!I', length_data)[0]
        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

if __name__ == '__main__':
    config = Config()
    demo = Inference(config)

    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_socket.bind(('0.0.0.0', 60000))
    recv_socket.listen(1)

    snd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    snd_socket.connect(('192.168.1.102', 60001))  # 替换接收端的IP地址


    conn, addr = recv_socket.accept()
    print(f"来自 {addr} 的连接已建立")

    while True:
        data = demo.receive_image(conn)
        if data is None:
            break

        np_arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input = demo.load_image_as_tensor(image_rgb)
        feature, semantic_image = demo.infer(input)

        image_data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        image_jpg = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        traditional_image = demo._add_gaussian_noise(image_jpg, config.snr)

        sent_data = {
            'semantic_image': semantic_image,
            'traditional_image': traditional_image,
            'feature': feature,
        }
        sent_data_bytes = pickle.dumps(sent_data)
        sent_data_length = len(sent_data_bytes)

        snd_socket.sendall(sent_data_length.to_bytes(4, byteorder='big'))
        snd_socket.sendall(sent_data_bytes)



    conn.close()
    recv_socket.close()
    snd_socket.close()

