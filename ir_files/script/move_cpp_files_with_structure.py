import os
import shutil

def move_cpp_files_with_structure(src_root, dest_root):
    """
    将源目录中的所有.cpp文件移动到目标目录，保持目录结构不变，目标存在时跳过
    
    :param src_root: 源根目录（包含需要移动的.cpp文件）
    :param dest_root: 目标根目录（需要创建相同的目录结构）
    """
    for root, dirs, files in os.walk(src_root):
        for filename in files:
            if filename.endswith(".cpp"):
                src_path = os.path.join(root, filename)
                
                # 计算相对于源根目录的相对路径
                rel_path = os.path.relpath(src_path, src_root)
                
                # 构建目标路径
                dest_path = os.path.join(dest_root, rel_path)
                
                # 检查目标文件是否存在
                if os.path.exists(dest_path):
                    print(f"Skipped: {dest_path} already exists")
                    continue  # 跳过已存在的文件
                
                # 创建目标目录（如果不存在）
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                # 移动文件
                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} -> {dest_path}")

if __name__ == "__main__":
    # 配置路径（根据实际需要修改）
    source_directory = "/workspace/triton-shared/include/triton-shared/Codegen"
    destination_directory = "/workspace/triton-shared/lib/Codegen"
    
    # 执行移动操作
    move_cpp_files_with_structure(source_directory, destination_directory)
    print("\n所有.cpp文件移动完成，目录结构已保持！")
