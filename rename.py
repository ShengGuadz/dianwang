import os

# 指定图片文件夹路径
image_folder = "./CLIC21/"

# 检查文件夹是否存在
if not os.path.exists(image_folder):
    print(f"Folder {image_folder} does not exist.")
    exit()

# 初始化计数器
counter = 1

# 遍历文件夹中的所有图片
for filename in os.listdir(image_folder):
    # 构造图片的完整路径
    old_path = os.path.join(image_folder, filename)

    # 检查是否为图片文件
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff','extension')):
        # 获取文件扩展名
        extension = os.path.splitext(filename)[1]

        # 新的文件名（数字序号）
        new_filename = "{}.png".format(counter)  # 例如 001.png
        new_path = os.path.join(image_folder, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

        # 更新计数器
        counter += 1