directory_path = '/home/hmsun/LLM-Questionaries-Personality/Llama/16p/Qwen2.5-3b-instruct/llm-gen'  # 替换为你的目录路径
import os
import pandas as pd


def compare_E(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # 调试输出
            if filename[0].upper() != 'E':
                print(f"Skipping file due to first character: {filename}")  # 调试输出
                continue  # 不是以E开头则跳过

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的第一个字符
            file_first_character = filename[0].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的第一个字符
                code_first_characters = df['Code'].str[0].str.upper()  # 提取所有行的第一个字符
                same_count = (code_first_characters == file_first_character).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")

def compare_I(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # 调试输出
            if filename[0].upper() != 'I':
                print(f"Skipping file due to first character: {filename}")  # 调试输出
                continue  # 不是以E开头则跳过

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的第一个字符
            file_first_character = filename[0].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的第一个字符
                code_first_characters = df['Code'].str[0].str.upper()  # 提取所有行的第一个字符
                same_count = (code_first_characters == file_first_character).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")

def compare_N(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")
            # 调试输出
            if filename[1].upper() != 'N':
                print(f"Skipping file due to first character: {filename}")  # 调试输出
                continue  # 不是以E开头则跳过

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的第2个字符
            file_first_character = filename[1].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的第2个字符
                code_first_characters = df['Code'].str[1].str.upper()  # 提取所有行的第2个字符
                same_count = (code_first_characters == file_first_character).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")

def compare_S(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")
            # 调试输出
            if filename[1].upper() != 'S':
                print(f"Skipping file due to first character: {filename}")  # 调试输出
                continue  # 不是以E开头则跳过

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的第2个字符
            file_first_character = filename[1].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的第2个字符
                code_first_characters = df['Code'].str[1].str.upper()  # 提取所有行的第2个字符
                same_count = (code_first_characters == file_first_character).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")

def compare_F(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # 调试输出
            if filename[2].upper() != 'F':
                print(f"Skipping file due to first character: {filename}")  # 调试输出
                continue  # 不是以E开头则跳过
            '''
                        if filename[0].upper() != 'E':
                            print(f"Skipping file due to first character: {filename}")  # 调试输出
                            continue  # 不是以E开头则跳过
                        '''

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的第3个字符
            file_first_character = filename[2].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的第3个字符
                code_first_characters = df['Code'].str[2].str.upper()  # 提取所有行的第一个字符
                same_count = (code_first_characters == file_first_character).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")

def compare_T(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # 调试输出
            if filename[2].upper() != 'T':
                print(f"Skipping file due to first character: {filename}")  # 调试输出
                continue  # 不是以E开头则跳过
            '''
                        if filename[0].upper() != 'E':
                            print(f"Skipping file due to first character: {filename}")  # 调试输出
                            continue  # 不是以E开头则跳过
                        '''

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的第3个字符
            file_first_character = filename[2].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的第3个字符
                code_first_characters = df['Code'].str[2].str.upper()  # 提取所有行的第一个字符
                same_count = (code_first_characters == file_first_character).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")

def compare_J(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # 调试输出
            if filename[3].upper() != 'J':
                print(f"Skipping file due to first character: {filename}")  # 调试输出
                continue  # 不是以E开头则跳过
            '''
                        if filename[0].upper() != 'E':
                            print(f"Skipping file due to first character: {filename}")  # 调试输出
                            continue  # 不是以E开头则跳过
            '''

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的第4个字符
            file_first_character = filename[3].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的第一个字符
                code_first_characters = df['Code'].str[3].str.upper()  # 提取所有行的第4个字符
                same_count = (code_first_characters == file_first_character).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")

def compare_P(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # 调试输出
            if filename[3].upper() != 'P':
                print(f"Skipping file due to first character: {filename}")  # 调试输出
                continue  # 不是以E开头则跳过
            '''
                        if filename[0].upper() != 'E':
                            print(f"Skipping file due to first character: {filename}")  # 调试输出
                            continue  # 不是以E开头则跳过
            '''

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的第4个字符
            file_first_character = filename[3].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的第一个字符
                code_first_characters = df['Code'].str[3].str.upper()  # 提取所有行的第4个字符
                same_count = (code_first_characters == file_first_character).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")



def compare_ALL(directory_path):
    results = {}
    total_same = 0
    total_diff = 0
    total_files = 0

    # 遍历目录下所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # 调试输出

            # 提取文件名中以-分割的第一个词
            first_word = filename.split('-')[0]

            # 检查第一个词的字符长度
            if len(first_word) > 4:
                print(f"Skipping file due to first word length: {filename}")  # 调试输出
                continue  # 超过4个字符则跳过

            # 提取文件名的前四个字符并转换为大写
            file_first_characters = filename[:4].upper()

            # 读取CSV文件
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 确保Code列存在
            if 'Code' in df.columns:
                total_files += 1

                # 提取Code列的前四个字符并转换为大写
                code_first_characters = df['Code'].str[:4].str.upper()  # 提取所有行的前四个字符
                same_count = (code_first_characters == file_first_characters).sum()  # 统计相同个数
                diff_count = len(code_first_characters) - same_count  # 统计不同个数

                # 更新总计
                total_same += same_count
                total_diff += diff_count

                # 记录比对结果
                results[filename] = {
                    'same_count': same_count,
                    'diff_count': diff_count,
                    'success_rate': same_count / len(code_first_characters) * 100 if len(
                        code_first_characters) > 0 else 0
                }
            else:
                print(f"Warning: {filename} does not contain 'Code' column.")

    # 输出比对结果
    for file, result in results.items():
        print(
            f"{file}: 相同个数 = {result['same_count']}, 不同个数 = {result['diff_count']}, 成功率 = {result['success_rate']:.2f}%")

    # 计算总成功率
    if total_files > 0:
        overall_success_rate = total_same / (total_same + total_diff) * 100
    else:
        overall_success_rate = 0

    print(f"总成功率 = {overall_success_rate:.2f}%")


# 你可以调用该函数并传入目录路径
# compare_E('你的目录路径')


# 调用示例
if __name__ == '__main__':
    compare_E(directory_path)
    compare_I(directory_path)
    compare_N(directory_path)
    compare_S(directory_path)
    compare_F(directory_path)
    compare_T(directory_path)
    compare_J(directory_path)
    compare_P(directory_path)
    compare_ALL(directory_path)