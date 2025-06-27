
import os
import shutil

folder_path = r"D:\pythonproject\收发版本WITT\data\epower"
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff')

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

print(f"\n开始重命名，总共 {len(image_files)} 张图片...\n")

for index, filename in enumerate(image_files, start=1):
    old_path = os.path.join(folder_path, filename)
    ext = os.path.splitext(filename)[1]  # 保留原扩展名
    new_filename = f"{index}{ext}"
    new_path = os.path.join(folder_path, new_filename)

    # 防止同名冲突
    if old_path != new_path:
        os.rename(old_path, new_path)
        print(f"{filename} → {new_filename}")

print("\n✅ 重命名完成，文件大小不会改变。")



