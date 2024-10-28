import os
import pandas as pd


def process_csv_files(folder_path):
    results = []
    total_same_count = 0
    total_rows_count = 0

    for filename in os.listdir(folder_path):
        if filename.startswith('updated') and filename.endswith('.csv'):
            # 提取第一个-与第十一个-之间的名称
            parts = filename.split('-')
            if len(parts) >= 7:  # 确保有至少11个部分
                extracted_name = '-'.join(parts[1:7]).upper()

                # 读取 CSV 文件
                csv_path = os.path.join(folder_path, filename)
                df = pd.read_csv(csv_path)

                # 提取指定列
                comparison_cols = ['MAC_Score_Comparison','NAR_Score_Comparison','PSY_Score_Comparison']

                if all(col in df.columns for col in comparison_cols):
                    # 将所需列转为大写
                    df[comparison_cols] = df[comparison_cols].apply(
                        lambda x: x.str.upper() if x.dtype == 'object' else x)

                    # 构建比较内容，确保格式一致
                    df['Comparison'] = (
                            'M-' + df['MAC_Score_Comparison'].astype(str).str.strip()  + '-' +
                            'N-' + df['NAR_Score_Comparison'].astype(str).str.strip()  + '-' +
                            'P-' + df['PSY_Score_Comparison'].astype(str).str.strip()
                    )
                    print(df['Comparison'])

                    # 检查与提取的文件名是否完全相同
                    same_count = (df['Comparison'] == extracted_name).sum()
                    different_count = len(df) - same_count  # 不同的个数
                    success_rate = same_count / len(df) if len(df) > 0 else 0

                    # 更新总计数
                    total_same_count += same_count
                    total_rows_count += len(df)

                    # 保存结果
                    results.append({
                        'File Name': extracted_name,
                        'Same Count': same_count,
                        'Different Count': different_count,
                        'Success Rate': success_rate
                    })

    # 计算总成功率
    overall_success_rate = total_same_count / total_rows_count if total_rows_count > 0 else 0

    return results, overall_success_rate


# 使用示例
folder_path = '/home/hmsun/LLM-Questionaries-Personality/Llama/sd3/Llama3.1-8b-instruct/combine'  # 替换为你的文件夹路径
results, overall_success_rate = process_csv_files(folder_path)

# 输出结果
for result in results:
    print(result)

# 输出总成功率
print(f"Overall Success Rate: {overall_success_rate:.2%}")