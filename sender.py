import os
import time

import cv2
from net.network import WITT
from data.datasets import get_loader
from loss.distortion import *
import torch.nn as nn
import numpy as np
import argparse
from utils import seed_torch
import socket
import pickle
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


class ConfigSender():
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
    test_data_dir = ["./data/epower/"]
    model_path = "./WITT_model/WITT_AWGN_DIV2K_fixed_snr10_psnr_C32.model"
    sent_dir = './static/saved/sent_image/'

    # save_dir1 = './static/saved/semantic_communication/'
    # save_dir2 = './static/saved/traditional_communication/'
    save_dir1 = './static/saved/semantic_communication/'
    save_dir2 = './static/saved/traditional_communication/'

    # host = '10.129.78.30'
    # host = '192.168.43.163'
    host = '127.0.0.1'
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

class Sender():
    def __init__(self, config: ConfigSender):
        self.config = config
        self.net = self._load_model()
        self.H, self.W = 0, 0
        self.test_loader = get_loader(args, config)
        self._prepare_save_dirs()
        # self.conn = self._setup_socket()
        # self.counter = 1

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

    # def _setup_socket(self):
    #     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     server_socket.connect((self.config.host, self.config.port))
    #     print(f"Connecting to {self.config.host}:{self.config.port}")
    #     return server_socket

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




if __name__ == '__main__':
    def add_gaussian_noise(image, snr_db):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        signal_pwr = np.mean(image ** 2)
        noise_pwr = signal_pwr / (10 ** (snr_db / 10))

        noise = np.random.normal(0, np.sqrt(noise_pwr), image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
        return image_bgr
    seed_torch()
    torch.manual_seed(seed=ConfigSender.seed)
    net = WITT(args, ConfigSender)

    if not os.path.exists(ConfigSender.sent_dir):
        os.makedirs(ConfigSender.sent_dir)
    else:
        image_list = glob.glob(os.path.join(ConfigSender.sent_dir, '*'))
        for image in image_list:
            os.remove(image)

    net.load_state_dict(torch.load(ConfigSender.model_path), strict=True)
    net = net.cuda()
    net.eval()

    local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # local_socket.connect(('10.129.78.30', 60000))
    local_socket.connect(('127.0.0.1', 60000))

    test_loader = get_loader(args, ConfigSender)
    H, W = 0, 0
    counter = 1
    with torch.no_grad():
        for batch_idx, input in enumerate(test_loader):
            print(f"Batch: {batch_idx}")
            input = input.cuda()
            B, _, H_new, W_new = input.shape
            if H_new != H or W_new != W:
                net.encoder.update_resolution(H_new, W_new)
                H = H_new
                W = W_new
            # feature = net.encoder(x=input, snr=12, model='WITT')
            # print("feature shape:", feature.shape)
            img_origin = cv2.imread(os.path.join(ConfigSender.test_data_dir[0], '{}.png'.format(counter)))

            img = add_gaussian_noise(img_origin, ConfigSender.snr)

            img_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            # data = {'size':(H_new, W_new), 'feature':feature, 'image':img_data}
            data = {'size': (H, W), 'feature': None, 'image': img_data}
            bytes_data = pickle.dumps(data)

            data_length = len(bytes_data)
            print("data_length:", data_length)
            local_socket.sendall(data_length.to_bytes(4, byteorder='big'))

            local_socket.sendall(bytes_data)

            cv2.imwrite(os.path.join(ConfigSender.sent_dir, '{}.png'.format(counter)), img_origin)

            counter += 1

            ack = local_socket.recv(4096)
            print(ack.decode('utf-8'))
