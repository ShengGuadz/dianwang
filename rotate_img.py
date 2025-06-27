from PIL import Image
import os

def transpose_jpeg_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                # 对图像进行90度旋转来对调长和宽
                transposed_img = img.transpose(Image.Transpose.ROTATE_90)

                # 保存修改后的图像，可以选择覆盖原图像或保存为新文件
                transposed_img.save(img_path)  # 覆盖原图像
                # transposed_img.save(img_path.replace('.jpeg', '_transposed.jpeg'))  # 保存为新文件

# 指定包含JPEG图像的文件夹路径
folder_path = '/home/ubuntu/users/zhuxiangben/WITT/data/media/Dataset/test_jpeg_out/11dB/origin_clic2024_numbered'  # 替换为你的文件夹路径
transpose_jpeg_images_in_folder(folder_path)




# from PIL import Image
# import os

# def flip_images_in_folder(folder_path, flip_horizontal=True):
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.png'):
#             img_path = os.path.join(folder_path, filename)
#             with Image.open(img_path) as img:
#                 # 根据需要选择水平翻转或垂直翻转
#                 if flip_horizontal:
#                     flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
#                 else:
#                     flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
#
#                 # 保存翻转后的图像，可以选择覆盖原图像或保存为新文件
#                 flipped_img.save(img_path)  # 覆盖原图像
#                 # flipped_img.save(img_path.replace('.png', '_flipped.png'))  # 保存为新文件
#
# # 指定包含PNG图像的文件夹路径
# folder_path = 'path_to_your_folder'  # 替换为你的文件夹路径
# flip_images_in_folder(folder_path)
