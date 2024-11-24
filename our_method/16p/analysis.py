import os
import pandas as pd


def process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("result-generated") and file.endswith(".csv"):
                    total_files += 1
                    file_path = os.path.join(folder_path, file)
                    # 读取CSV文件，指定header=None
                    try:
                        df = pd.read_csv(file_path, header=None)
                        # 检查数据是否为空
                        if df.empty:
                            print(f"警告: {file_path} 是空文件.")
                            continue
                        # 获取第一行第一列的前四个字符
                        first_column_value = str(df.iloc[0, 0])[:4]
                        # 解析文件名中的相关字符
                        parts = file.split('-')
                        if len(parts) >= 6:
                            matched_value = ''.join(parts[2:6])
                            # 检查是否一致
                            if matched_value == first_column_value:
                                matched_files += 1
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    return total_files, matched_files, success_rate


if __name__ == "__main__":
    base_directory = "/home/hmsun/wordnet/our_method/16p/Llama3.2-3b-instruct/our_method"  # 请根据实际情况修改路径
    total, matched, rate = process_directory(base_directory)
    print(f"总文件数: {total}")
    print(f"匹配的文件数: {matched}")
    print(f"成功率: {rate:.2f}%")