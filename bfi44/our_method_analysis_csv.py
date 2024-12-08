import pandas as pd
import os

# 设置要分析的文件夹路径
#folder_path = '/home/hmsun/LLM-Questionaries-Personality/bfi44/Llama3.1-8b-instruct/our_method/result_iteration_5'  # 替换为你的文件夹路径

base_folder_path = '/home/hmsun/wordnet/our_method/bfi44/Llama3.2-3b-instruct/our_method_advanced'

# 遍历每个 iteration 文件夹
for i in range(1, 11):
    folder_path = os.path.join(base_folder_path, f'result_iteration_{i}')
    # 在这里对每个文件夹进行操作
    print(f'Processing folder: {folder_path}')
    # 例如：读取文件、分析数据等


    # 定义阈值
    thresholds = {
        'EXT_Score': 3.39,
        #'EXT_std':  0.84,
        'AGR_Score': 3.78,
        #'AGR_std':  0.67,
        'CSN_Score': 3.59,
        #'CSN_std':  0.71,
        'EST_Score': 2.90,
        #'EST_std':  0.82,
        'OPN_Score': 3.67,
        #'OPN_std':  0.66

    }

    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # 确保数据框中包含所有需要的列
            for column in thresholds.keys():
                if column in df.columns:
                    df[column + '_Comparison'] = df[column].apply(lambda x: 'high' if x > thresholds[column] else 'low')
                else:
                    print(f'Warning: {column} not found in {filename}')

            # 将结果写回到新的CSV文件
            output_file_path = os.path.join(folder_path, f'updated_{filename}')
            df.to_csv(output_file_path, index=False)
            print(f'Processed file: {output_file_path}')
