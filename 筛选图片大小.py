import os

# 设置要遍历的文件夹路径（你可以修改为你自己的路径）
folder_path = r"D:\pythonproject\收发版本WITT\input\test\ywzt_yfyc"  # 如：r"D:\images" 或 "./my_images"

# 设置大小阈值：200KB = 200 * 1024 字节
size_threshold = 300 * 1024

# 设置图片扩展名
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

print("超过 300KB 的图片有：\n")

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(root, filename)
            size = os.path.getsize(filepath)
            if size > size_threshold:
                print(f"{filename}  ({size/1024:.1f} KB)")
