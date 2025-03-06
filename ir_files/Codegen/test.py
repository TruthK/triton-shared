import os

def rename_files(directory):
    # 检查目标目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在")
        return False

    # 遍历目录中的所有条目
    for entry in os.scandir(directory):
        # 仅处理文件
        if entry.is_file():
            filename = entry.name
            # 检查文件名是否以IREE开头
            if filename.startswith("IREE"):
                # 构建新文件名
                new_filename = "TTS" + filename[4:]
                new_path = os.path.join(directory, new_filename)
                
                try:
                    # 执行重命名操作
                    os.rename(entry.path, new_path)
                    print(f"成功重命名: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"重命名失败: {filename} | 错误: {str(e)}")
                    continue
    return True

if __name__ == "__main__":
    target_dir = "//workspace/triton-shared/include/triton-shared/Codegen/Dialect/GPU/TransformExtensions/"
    rename_files(target_dir)
