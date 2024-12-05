import os
from pathlib import Path

def delete_process_mark_files(root_path: Path):
    """删除文件夹及子文件夹中的process.mark文件"""
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == "process.mark":
                file_path = Path(dirpath) / filename
                try:
                    file_path.unlink()
                    print(f"已删除 {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {str(e)}")

if __name__ == "__main__":
    root_path = Path(r"F:\Dataset_selected_MAPO_Run_1126")
    delete_process_mark_files(root_path)
