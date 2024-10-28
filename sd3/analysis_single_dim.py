directory_path = '/home/hmsun/LLM-Questionaries-Personality/Llama/sd3/Llama3.2-3b-instruct/combine'  # 替换为你的目录路径
import os
import pandas as pd
'''
EXT_Score_Comparison,AGR_Score_Comparison,CSN_Score_Comparison,EST_Score_Comparison,OPN_Score_Comparison
'''
#'MAC_Score_Comparison','NAR_Score_Comparison','PSY_Score_Comparison'

def MAC_Score_Comparison(directory_path, column_name):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith('update') and filename.endswith('.csv'):
            # 提取文件名中第二个字符
            if len(filename.split('-')) > 1:
                second_character = filename.split('-')[2].lower()
                # 检查second_character是否为'high'
                if second_character != 'high':
                    print(f"Skipping file: {filename}")  # 调试输出
                    continue  # 不是'High'则跳过
            else:
                print(f"Skipping file due to insufficient parts in filename: {filename}")  # 调试输出
                continue

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保指定的列存在
            if column_name in df.columns:
                total_files += 1

                # 比对指定列中的值与文件名第二个字符
                comparison_results = df[column_name].str.lower() == second_character

                # 计算相同个数和不同个数
                same_count = comparison_results.sum()
                diff_count = len(comparison_results) - same_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(comparison_results) * 100
                }

                # 更新总计
                total_same += same_count
                total_diff += diff_count
            else:
                print(f"Warning: {filename} does not contain '{column_name}' column.")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    print(f"总成功率 = {overall_success_rate:.2f}%")

def MAC_Score_Comparison_Low(directory_path, column_name):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith('update') and filename.endswith('.csv'):
            # 提取文件名中第二个字符
            if len(filename.split('-')) > 1:
                second_character = filename.split('-')[2].lower()
                # 检查second_character是否为'high'
                if second_character != 'low':
                    print(f"Skipping file : {filename}")  # 调试输出
                    continue  # 不是'High'则跳过
            else:
                print(f"Skipping file due to insufficient parts in filename: {filename}")  # 调试输出
                continue

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保指定的列存在
            if column_name in df.columns:
                total_files += 1

                # 比对指定列中的值与文件名第二个字符
                comparison_results = df[column_name].str.lower() == second_character

                # 计算相同个数和不同个数
                same_count = comparison_results.sum()
                diff_count = len(comparison_results) - same_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(comparison_results) * 100
                }

                # 更新总计
                total_same += same_count
                total_diff += diff_count
            else:
                print(f"Warning: {filename} does not contain '{column_name}' column.")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    print(f"总成功率 = {overall_success_rate:.2f}%")

def NAR_Score_Comparison(directory_path, column_name):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith('update') and filename.endswith('.csv'):
            # 提取文件名中第4个字符
            if len(filename.split('-')) > 1:
                second_character = filename.split('-')[4].lower()
                # 检查second_character是否为'high'
                if second_character != 'high':
                    print(f"Skipping file : {filename}")  # 调试输出
                    continue  # 不是'High'则跳过
            else:
                print(f"Skipping file due to insufficient parts in filename: {filename}")  # 调试输出
                continue

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保指定的列存在
            if column_name in df.columns:
                total_files += 1

                # 比对指定列中的值与文件名第二个字符
                comparison_results = df[column_name].str.lower() == second_character

                # 计算相同个数和不同个数
                same_count = comparison_results.sum()
                diff_count = len(comparison_results) - same_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(comparison_results) * 100
                }

                # 更新总计
                total_same += same_count
                total_diff += diff_count
            else:
                print(f"Warning: {filename} does not contain '{column_name}' column.")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    print(f"总成功率 = {overall_success_rate:.2f}%")

def NAR_Score_Comparison_Low(directory_path, column_name):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith('update') and filename.endswith('.csv'):
            # 提取文件名中第4个字符

            if len(filename.split('-')) > 1:
                second_character = filename.split('-')[4].lower()
                # 检查second_character是否为'high'
                if second_character != 'low':
                    print(f"Skipping file : {filename}")  # 调试输出
                    continue  # 不是'High'则跳过
            else:
                print(f"Skipping file due to insufficient parts in filename: {filename}")  # 调试输出
                continue

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保指定的列存在
            if column_name in df.columns:
                total_files += 1

                # 比对指定列中的值与文件名第二个字符
                comparison_results = df[column_name].str.lower() == second_character

                # 计算相同个数和不同个数
                same_count = comparison_results.sum()
                diff_count = len(comparison_results) - same_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(comparison_results) * 100
                }

                # 更新总计
                total_same += same_count
                total_diff += diff_count
            else:
                print(f"Warning: {filename} does not contain '{column_name}' column.")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    print(f"总成功率 = {overall_success_rate:.2f}%")

def PSY_Score_Comparison(directory_path, column_name):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith('update') and filename.endswith('.csv'):
            # 提取文件名中第6个字符
            if len(filename.split('-')) > 1:
                second_character = filename.split('-')[6].lower()
                # 检查second_character是否为'high'
                if second_character != 'high':
                    print(f"Skipping file : {filename}")  # 调试输出
                    continue  # 不是'High'则跳过
            else:
                print(f"Skipping file due to insufficient parts in filename: {filename}")  # 调试输出
                continue

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保指定的列存在
            if column_name in df.columns:
                total_files += 1

                # 比对指定列中的值与文件名第二个字符
                comparison_results = df[column_name].str.lower() == second_character

                # 计算相同个数和不同个数
                same_count = comparison_results.sum()
                diff_count = len(comparison_results) - same_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(comparison_results) * 100
                }

                # 更新总计
                total_same += same_count
                total_diff += diff_count
            else:
                print(f"Warning: {filename} does not contain '{column_name}' column.")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    print(f"总成功率 = {overall_success_rate:.2f}%")

def PSY_Score_Comparison_Low(directory_path, column_name):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith('update') and filename.endswith('.csv'):
            # 提取文件名中第6个字符

            if len(filename.split('-')) > 1:
                second_character = filename.split('-')[6].lower()
                # 检查second_character是否为'high'
                if second_character != 'low':
                    print(f"Skipping file : {filename}")  # 调试输出
                    continue  # 不是'High'则跳过
            else:
                print(f"Skipping file due to insufficient parts in filename: {filename}")  # 调试输出
                continue

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保指定的列存在
            if column_name in df.columns:
                total_files += 1

                # 比对指定列中的值与文件名第二个字符
                comparison_results = df[column_name].str.lower() == second_character

                # 计算相同个数和不同个数
                same_count = comparison_results.sum()
                diff_count = len(comparison_results) - same_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(comparison_results) * 100
                }

                # 更新总计
                total_same += same_count
                total_diff += diff_count
            else:
                print(f"Warning: {filename} does not contain '{column_name}' column.")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    print(f"总成功率 = {overall_success_rate:.2f}%")

#3.1-8b 100 22.5 100 50 100 2.5 vanilla#combine 97.5 65 100 5 100 12.5
# 调用示例
if __name__ == '__main__':
    MAC_Score_Comparison(directory_path, 'MAC_Score_Comparison')
    MAC_Score_Comparison_Low(directory_path, 'MAC_Score_Comparison')
    NAR_Score_Comparison(directory_path, 'NAR_Score_Comparison')
    NAR_Score_Comparison_Low(directory_path, 'NAR_Score_Comparison')
    PSY_Score_Comparison(directory_path, 'PSY_Score_Comparison')
    PSY_Score_Comparison_Low(directory_path, 'PSY_Score_Comparison')
    # 替换为你的目录路径和列名