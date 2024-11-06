import pandas as pd
import os

# 设置要分析的文件夹路径
folder_path = '/home/hmsun/LLM-Questionaries-Personality/Llama/bfi44/Qwen2.5-3b-instruct/llm-gen'  # 替换为你的文件夹路径

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
        output_file_path = os.path.join(folder_path, f'new-{filename}')
        df.to_csv(output_file_path, index=False)
        print(f'Processed file: {output_file_path}')