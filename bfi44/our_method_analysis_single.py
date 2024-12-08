import os
import pandas as pd

def N1_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[3] == 'High':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[2:4]).lower()  # 转换为小写

                                # 提取指定列的内容
                                est_score = df.loc[0, 'EST_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"N-{est_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"N+ 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def N2_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[3] == 'Low':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[2:4]).lower()  # 转换为小写

                                # 提取指定列的内容

                                est_score = df.loc[0, 'EST_Score_Comparison']


                                # 拼接字符串
                                constructed_string = (
                                    f"N-{est_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"N- 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def O1_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[5] == 'High':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[4:6]).lower()  # 转换为小写

                                # 提取指定列的内容
                                ext_score = df.loc[0, 'EXT_Score_Comparison']
                                agr_score = df.loc[0, 'AGR_Score_Comparison']
                                csn_score = df.loc[0, 'CSN_Score_Comparison']
                                est_score = df.loc[0, 'EST_Score_Comparison']
                                opn_score = df.loc[0, 'OPN_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"O-{opn_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"O+成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def O2_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[5] == 'Low':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[4:6]).lower()  # 转换为小写

                                # 提取指定列的内容
                                ext_score = df.loc[0, 'EXT_Score_Comparison']
                                agr_score = df.loc[0, 'AGR_Score_Comparison']
                                csn_score = df.loc[0, 'CSN_Score_Comparison']
                                est_score = df.loc[0, 'EST_Score_Comparison']
                                opn_score = df.loc[0, 'OPN_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"O-{opn_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"O- 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def C1_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[7] == 'High':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[6:8]).lower()  # 转换为小写

                                # 提取指定列的内容
                                ext_score = df.loc[0, 'EXT_Score_Comparison']
                                agr_score = df.loc[0, 'AGR_Score_Comparison']
                                csn_score = df.loc[0, 'CSN_Score_Comparison']
                                est_score = df.loc[0, 'EST_Score_Comparison']
                                opn_score = df.loc[0, 'OPN_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"C-{csn_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"C+ 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def C2_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[7] == 'Low':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[6:8]).lower()  # 转换为小写

                                # 提取指定列的内容
                                ext_score = df.loc[0, 'EXT_Score_Comparison']
                                agr_score = df.loc[0, 'AGR_Score_Comparison']
                                csn_score = df.loc[0, 'CSN_Score_Comparison']
                                est_score = df.loc[0, 'EST_Score_Comparison']
                                opn_score = df.loc[0, 'OPN_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"C-{csn_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"C- 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def E1_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[9] == 'High':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[8:10]).lower()  # 转换为小写

                                # 提取指定列的内容
                                ext_score = df.loc[0, 'EXT_Score_Comparison']
                                agr_score = df.loc[0, 'AGR_Score_Comparison']
                                csn_score = df.loc[0, 'CSN_Score_Comparison']
                                est_score = df.loc[0, 'EST_Score_Comparison']
                                opn_score = df.loc[0, 'OPN_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"E-{ext_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"E+ 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def E2_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[9] == 'Low':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[8:10]).lower()  # 转换为小写

                                # 提取指定列的内容
                                ext_score = df.loc[0, 'EXT_Score_Comparison']
                                agr_score = df.loc[0, 'AGR_Score_Comparison']
                                csn_score = df.loc[0, 'CSN_Score_Comparison']
                                est_score = df.loc[0, 'EST_Score_Comparison']
                                opn_score = df.loc[0, 'OPN_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"E-{ext_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"E- 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def A1_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[11] == 'High':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[10:12]).lower()  # 转换为小写

                                # 提取指定列的内容
                                ext_score = df.loc[0, 'EXT_Score_Comparison']
                                agr_score = df.loc[0, 'AGR_Score_Comparison']
                                csn_score = df.loc[0, 'CSN_Score_Comparison']
                                est_score = df.loc[0, 'EST_Score_Comparison']
                                opn_score = df.loc[0, 'OPN_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"A-{agr_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"A+ 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate

def A2_process_directory(base_dir):
    total_files = 0
    matched_files = 0

    # 遍历our_method目录下的所有文件夹
    for folder in os.listdir(base_dir):
        if folder.startswith("result_iteration"):
            folder_path = os.path.join(base_dir, folder)
            # 遍历文件夹中的所有文件
            for file in os.listdir(folder_path):
                if file.startswith("updated_result") and file.endswith(".csv"):
                    parts = file.split('-')
                    if len(parts) >= 5 and parts[11] == 'Low':
                        total_files += 1
                        file_path = os.path.join(folder_path, file)
                        # 读取CSV文件
                        try:
                            df = pd.read_csv(file_path)
                            # 检查数据是否为空
                            if df.empty:
                                print(f"警告: {file_path} 是空文件.")
                                continue

                            # 提取文件名中的相关字符
                            parts = file.split('-')
                            if len(parts) >= 12:
                                extracted_name = '-'.join(parts[10:12]).lower()  # 转换为小写

                                # 提取指定列的内容
                                ext_score = df.loc[0, 'EXT_Score_Comparison']
                                agr_score = df.loc[0, 'AGR_Score_Comparison']
                                csn_score = df.loc[0, 'CSN_Score_Comparison']
                                est_score = df.loc[0, 'EST_Score_Comparison']
                                opn_score = df.loc[0, 'OPN_Score_Comparison']

                                # 拼接字符串
                                constructed_string = (
                                    f"A-{agr_score}"
                                ).lower()  # 转换为小写

                                # 检查构建的字符串是否与提取的名称一致（不区分大小写）
                                if constructed_string == extracted_name:
                                    matched_files += 1
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")

    # 计算成功率
    success_rate = (matched_files / total_files) * 100 if total_files > 0 else 0
    print(f"总文件数: {total_files}")
    print(f"匹配的文件数: {matched_files}")
    print(f"A- 成功率: {success_rate:.2f}%")
    return total_files, matched_files, success_rate



if __name__ == "__main__":
    base_directory = ("/Users/haoming/Desktop/gpt_16p/our_method_server/qwen2.5-72b-instruct/bfi44/our_method"
                      )  # 请根据实际情况修改路径
    A1_process_directory(base_directory)
    A2_process_directory(base_directory)
    C1_process_directory(base_directory)
    C2_process_directory(base_directory)
    E1_process_directory(base_directory)
    E2_process_directory(base_directory)
    N1_process_directory(base_directory)
    N2_process_directory(base_directory)
    O1_process_directory(base_directory)
    O2_process_directory(base_directory)

