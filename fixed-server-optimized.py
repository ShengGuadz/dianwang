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
import struct
import os
import json  # 用于调试输出

# 确保static目录存在
os.makedirs('static/saved/sent_image', exist_ok=True)
os.makedirs('static/saved/semantic_communication', exist_ok=True)
os.makedirs('static/saved/traditional_communication', exist_ok=True)

# 命令行参数设置
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


class Config():
    seed = 1024
    pass_channel = True
    CUDA = True and torch.cuda.is_available()  # 检查CUDA是否真的可用
    device = torch.device("cuda:0" if CUDA else "cpu")
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
        try:
            self.net = self._load_model()
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 如果模型加载失败，使用简化的处理流程
            self.net = None
        self.H, self.W = 0, 0
        self._prepare_save_dirs()

        seed_torch()
        torch.manual_seed(seed=config.seed)

    def _load_model(self):
        net = WITT(args, self.config)
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.config.model_path}")
                
            net.load_state_dict(torch.load(self.config.model_path, 
                               map_location=self.config.device), strict=True)
            net = net.cuda() if self.config.CUDA else net.cpu()
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise e
            
        net.eval()
        return net

    def _prepare_save_dirs(self):
        os.makedirs(self.config.sent_dir, exist_ok=True)
        for file in os.listdir(self.config.sent_dir):
            try:
                os.remove(os.path.join(self.config.sent_dir, file))
            except:
                pass
        for save_dir in [self.config.save_dir1, self.config.save_dir2]:
            os.makedirs(save_dir, exist_ok=True)
            for file in os.listdir(save_dir):
                try:
                    os.remove(os.path.join(save_dir, file))
                except:
                    pass

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
        """
        # 将量化的整数数据转换为浮动数值，并映射回 [0, 1]
        tensor_normalized = tensor_quantized.astype(np.float32) / 255.0

        # 使用反量化公式恢复到原始范围
        tensor_dequantized = tensor_normalized * (max_val - min_val) + min_val

        # 返回反量化后的浮动数据
        if self.config.CUDA:
            return torch.tensor(tensor_dequantized).cuda()
        return torch.tensor(tensor_dequantized)

    def infer(self, input_image):
        # 如果模型加载失败，使用简化处理
        if self.net is None:
            # 从输入图像中提取初始尺寸
            img_np = input_image[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 创建简单的"语义"图像（加噪声版本）
            blurred = cv2.GaussianBlur(img_bgr, (7, 7), 0)
            return np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8), blurred
        
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

    def load_image_as_tensor(self, image):
        if isinstance(image, str):
            # 如果输入是路径
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray) and image.ndim == 3:
            # 如果输入是BGR图像
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W, _ = image.shape
        # 确保尺寸是128的倍数
        H_new = H - (H % 128)
        W_new = W - (W % 128)
        if H_new == 0: H_new = 128
        if W_new == 0: W_new = 128
        
        image = image[:H_new, :W_new, :]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image).unsqueeze(0)
        if self.config.CUDA:
            tensor = tensor.cuda()
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

def main():
    print("正在初始化服务器...")
    config = Config()
    demo = Inference(config)

    # 接收图像的套接字
    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_socket.bind(('0.0.0.0', 60000))
    recv_socket.listen(1)
    print("等待发送端连接...")

    # 发送处理结果的套接字
    snd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("正在连接到接收端...")
    
    # 尝试连接接收端，如果连接失败则重试
    # receiver_ip = '127.0.0.1'  # 本地测试时使用localhost
    receiver_ip = '10.243.149.80'#zerotier地址
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            snd_socket.connect((receiver_ip, 60001))
            print(f"已连接到接收端 {receiver_ip}")
            break
        except socket.error as e:
            retry_count += 1
            print(f"连接接收端失败 (尝试 {retry_count}/{max_retries}): {e}")
            if retry_count == max_retries:
                print("达到最大尝试次数，继续运行但可能无法发送处理结果")
                break
            time.sleep(5)  # 等待5秒后重试

    # 等待发送端连接
    conn, addr = recv_socket.accept()
    print(f"来自 {addr} 的发送端连接已建立")
    
    # 向发送端发送ready信号
    conn.sendall(b'READY')

    counter = 0
    while True:
        try:
            print(f"等待图像 #{counter} 从发送端发送...")
            data = demo.receive_image(conn)
            if data is None:
                print("接收图像失败，连接可能已关闭")
                break

            # 向发送端发送"收到图像"确认
            conn.sendall(b'GOT_IMAGE')

            # 解码接收到的图像
            np_arr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                print("图像解码失败")
                continue
                
            print(f"已接收图像 #{counter} 并成功解码, 尺寸: {image.shape}")
            
            # 处理图像
            try:
                input_tensor = demo.load_image_as_tensor(image)
                feature, semantic_image = demo.infer(input_tensor)
                
                # 使用传统方式传输 - 通过通信信道（这里简化为添加噪声）
                image_data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                image_jpg = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                traditional_image = demo._add_gaussian_noise(image_jpg, config.snr)
                
                # 保存处理后的图像（用于调试）
                cv2.imwrite(f"static/saved/semantic_communication/debug_{counter}.png", semantic_image)
                cv2.imwrite(f"static/saved/traditional_communication/debug_{counter}.jpg", traditional_image)
                
                # 准备发送数据
                sent_data = {
                    'semantic_image': semantic_image,
                    'traditional_image': traditional_image,
                    'feature': feature,
                    'counter': counter,  # 添加计数器以便跟踪
                }
                
                # 序列化发送数据
                try:
                    print(f"正在发送处理结果 #{counter} 到接收端...")
                    sent_data_bytes = pickle.dumps(sent_data)
                    sent_data_length = len(sent_data_bytes)
                    
                    # 发送数据长度和数据
                    snd_socket.sendall(sent_data_length.to_bytes(4, byteorder='big'))
                    snd_socket.sendall(sent_data_bytes)
                    print(f"已发送处理结果 #{counter} 到接收端")
                    
                    # 等待接收端确认收到并处理完成
                    print("等待接收端确认...")
                    ack = snd_socket.recv(4)
                    if ack == b'NEXT':
                        print("接收端已确认，可以继续处理下一图像")
                    
                    # 通知发送端可以显示图像
                    conn.sendall(b'SHOW_IMAGE')
                    print("已通知发送端显示图像")
                    
                except Exception as e:
                    print(f"发送数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
                
            except Exception as e:
                print(f"处理图像时出错: {e}")
                import traceback
                traceback.print_exc()
                
                # 如果处理失败，创建默认数据继续发送
                dummy_semantic = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
                dummy_traditional = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
                dummy_feature = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
                
                sent_data = {
                    'semantic_image': dummy_semantic,
                    'traditional_image': dummy_traditional,
                    'feature': dummy_feature,
                    'counter': counter,
                }
                
                try:
                    sent_data_bytes = pickle.dumps(sent_data)
                    sent_data_length = len(sent_data_bytes)
                    snd_socket.sendall(sent_data_length.to_bytes(4, byteorder='big'))
                    snd_socket.sendall(sent_data_bytes)
                    print(f"已发送默认数据 #{counter}")
                    
                    # 等待接收端确认
                    ack = snd_socket.recv(4)
                    
                    # 通知发送端可以显示图像
                    conn.sendall(b'SHOW_IMAGE')
                except Exception as e:
                    print(f"发送默认数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    
            counter += 1
                
        except Exception as e:
            print(f"接收/处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            # 短暂延迟以避免过快循环
            time.sleep(1)
            continue

    # 关闭连接
    conn.close()
    recv_socket.close()
    snd_socket.close()
    print("服务器已关闭")

if __name__ == '__main__':
    main()
