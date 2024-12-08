import pandas as pd
import os


# 设置要分析的文件夹路径
#folder_path = '/home/hmsun/LLM-Questionaries-Personality/bfi44/Llama3.1-8b-instruct/our_method/result_iteration_5'  # 替换为你的文件夹路径

base_folder_path = '/home/hmsun/our_method/sd3/Qwen2.5-3b-instruct/our_method'

# 遍历每个 iteration 文件夹
for i in range(1, 11):
    folder_path = os.path.join(base_folder_path, f'result_iteration_{i}')
    # 在这里对每个文件夹进行操作
    print(f'Processing folder: {folder_path}')
# 设置要分析的文件夹路径
#folder_path = '/home/hmsun/LLM-Personality-Questionnaires/sd3/Llama3.1-8b-instruct/llm-gen2'  # 替换为你的文件夹路径

    # 定义阈值
    thresholds = {
        'MAC_Score': 2.96,
        #'MAC_Std': 0.65,
        'NAR_Score': 2.97,
        #'NAR_Std': 0.61,
        'PSY_Score': 2.09,
        #'PSY_Std': 0.63
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
