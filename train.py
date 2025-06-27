import torch.optim as optim
from net.network import WITT
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 修改后的参数解析
parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true', default=False,
                    help='training or testing, default is testing')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name, default is DIV2K')
parser.add_argument('--testset', type=str, default='CLIC21',
                    choices=['kodak', 'CLIC21'],
                    help='test dataset name, default is CLIC21')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics, default is MSE')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet, default is WITT')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh', 'rician'],
                    help='wireless channel model, default is awgn')
parser.add_argument('--C', type=int, default=32,
                    help='bottleneck dimension, default is 32')
parser.add_argument('--multiple-snr', type=str, default='12',
                    help='random or fixed snr, default is 12')
args = parser.parse_args()

class config():
    seed = 1024
    ###在这里设置是否经过软件模拟的信道
    pass_channel = False
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "/media/Dataset/CIFAR10/"
        test_data_dir = "/media/Dataset/CIFAR10/"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        # train_data_dir = ["/home/zhuxiangben/WITT/data/media/Dataset/clic2024_test_image/"]
        train_data_dir = ["/root/autodl-tmp/users/zxb/WITT/data/media/Dataset/clic2024_test_image/"]
        if args.testset == 'kodak':
            test_data_dir = ["./Dataset/kodak_test/"]

        ########在这里修要传输的图像，png格式
        elif args.testset == 'CLIC21':
            # test_data_dir = ["/home/zhuxiangben/WITT/data/media/Dataset/clic2024_test_image/"]
            test_data_dir = ["CLIC21/"]
        batch_size = 16
        downsample = 4
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


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained


# def train_one_epoch(args):
#     net.train()
#     elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
#     metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
#     global global_step
#     if args.trainset == 'CIFAR10':
#         for batch_idx, (input, label) in enumerate(train_loader):
#             start_time = time.time()
#             global_step += 1
#             input = input.cuda()
#             recon_image, CBR, SNR, mse, loss_G = net(input)
#             loss = loss_G
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             elapsed.update(time.time() - start_time)
#             losses.update(loss.item())
#             cbrs.update(CBR)
#             snrs.update(SNR)
#             if mse.item() > 0:
#                 psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
#                 psnrs.update(psnr.item())
#                 msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
#                 msssims.update(msssim)
#             else:
#                 psnrs.update(100)
#                 msssims.update(100)
#
#             if (global_step % config.print_step) == 0:
#                 process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
#                 log = (' | '.join([
#                     f'Epoch {epoch}',
#                     f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
#                     f'Time {elapsed.val:.3f}',
#                     f'Loss {losses.val:.3f} ({losses.avg:.3f})',
#                     f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
#                     f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
#                     f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
#                     f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
#                     f'Lr {cur_lr}',
#                 ]))
#                 logger.info(log)
#                 for i in metrics:
#                     i.clear()
#     else:
#         for batch_idx, input in enumerate(train_loader):
#             start_time = time.time()
#             global_step += 1
#             input = input.cuda()
#             recon_image, CBR, SNR, mse, loss_G = net(input)
#             loss = loss_G
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             elapsed.update(time.time() - start_time)
#             losses.update(loss.item())
#             cbrs.update(CBR)
#             snrs.update(SNR)
#             if mse.item() > 0:
#                 psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
#                 psnrs.update(psnr.item())
#                 msssim = 1 - loss_G
#                 msssims.update(msssim)
#
#             else:
#                 psnrs.update(100)
#                 msssims.update(100)
#
#             if (global_step % config.print_step) == 0:
#                 process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
#                 log = (' | '.join([
#                     f'Epoch {epoch}',
#                     f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
#                     f'Time {elapsed.val:.3f}',
#                     f'Loss {losses.val:.3f} ({losses.avg:.3f})',
#                     f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
#                     f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
#                     f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
#                     f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
#                     f'Lr {cur_lr}',
#                 ]))
#                 logger.info(log)
#                 for i in metrics:
#                     i.clear()
#     for i in metrics:
#         i.clear()

def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    total_time = 0.
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
            else:
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G, noisy_feature= net(input)

                    recon_image = net.decoder(noisy_feature, 12, net.model)

                    # 将recon_image存储
                    recovery_image = recon_image.clamp(0., 1.).cpu()
                    recovery_image = recovery_image.numpy()
                    recovery_image = np.squeeze(recovery_image, axis=0)
                    recovery_image = recovery_image * 255
                    recovery_image = recovery_image.astype(np.uint8)
                    recovery_image = np.transpose(recovery_image, (1, 2, 0))
                    image = Image.fromarray(recovery_image, 'RGB')
                    # recover_images_dir = "/home/zhuxiangben/WITT/datasets/recovered_clic_C16_12dB/"
                    recover_images_dir = "/root/autodl-tmp/users/zxb/WITT/datasets/recovered_clic_C16_12dB/"
                    if not os.path.exists(recover_images_dir):
                        os.makedirs(recover_images_dir)
                    image.save(recover_images_dir + str(batch_idx) + 'clic.png')
                    # 将recon_image存储

                    elapsed.update(time.time() - start_time)
                    this_time = time.time() - start_time
                    total_time += this_time
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()
    Average_time = total_time/len(test_loader)

    # print("Average_time:", Average_time)
    # print("SNR: {}" .format(results_snr.tolist()))
    # print("CBR: {}".format(results_cbr.tolist()))
    # print("PSNR: {}" .format(results_psnr.tolist()))
    # print("MS-SSIM: {}".format(results_msssim.tolist()))

    print("Average_time: {:.3f}".format(Average_time))
    print("SNR: {}".format([f"{x:.3f}" for x in results_snr.tolist()]))
    print("CBR: {}".format([f"{x:.3f}" for x in results_cbr.tolist()]))
    print("PSNR: {}".format([f"{x:.3f}" for x in results_psnr.tolist()]))
    print("MS-SSIM: {}".format([f"{x:.3f}" for x in results_msssim.tolist()]))



    print("Finish Test!")

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = WITT(args, config)
    model_path = "./WITT_model/WITT_AWGN_DIV2K_fixed_snr10_psnr_C32.model"
    load_weights(model_path)
    net = net.cuda()
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    # train_loader, test_loader = get_loader(args, config)
    test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    # optimizer = optim.Adam(model_params, lr=cur_lr)
    # global_step = 0
    # steps_epoch = global_step // train_loader.__len__()
    # if args.training:
    #     for epoch in range(steps_epoch, config.tot_epoch):
    #         train_one_epoch(args)
    #         if (epoch + 1) % config.save_model_freq == 0:
    #             save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
    #             test()
    # else:
    test()

