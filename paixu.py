import os

def rename_images(folder_path):
    # 获取文件夹内所有文件
    files = os.listdir(folder_path)
    # 筛选出图片文件（可以根据需要扩展支持更多图片格式）
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # 对图片文件进行排序（按照文件名排序）
    image_files.sort()
    # 重命名图片
    for i, file_name in enumerate(image_files, start=1):
        # 构造新的文件名
        new_file_name = f"{i:04d}.jpg"  # 格式化为四位数字，如0001.jpg
        # 构造完整的文件路径
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {file_name} to {new_file_name}")

# 使用示例
folder_path = r"D:\3Dceshi\Datasets\IGBT"  # 替换为你的图片文件夹路径
rename_images(folder_path)