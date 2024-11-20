import time

import os
import re
import ast
import pandas as pd
import torch
import transformers
from nltk.corpus import wordnet as wn
import nltk
from openai import OpenAI

# 用于将字符串转换为列表
words_num = 50
single_prompt_template = {
    "mbti_prompt": [
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are strategic, pragmatic, and often willing to manipulate others to achieve their goals. They may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection.",
            "label": "M-High"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who are exceptionally honest, transparent, and guided by strong moral principles. They respect others' autonomy and have no interest in manipulation. They value sincerity and trust, making them someone others can rely on for straightforward and ethical interactions.",
            "label":"M-Low"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who likely feel a strong sense of self-importance, often seeing themselves as superior to others and deserving of admiration. They tend to dominate conversations, seek praise, and may overestimate their knowledge or capabilities.",
            "label":"N-High"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who are genuinely humble and uninterested in self-promotion. They don't seek attention or admiration from others, often downplaying their own achievements. They’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting their own importance.",
            "label":"N-Low"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who tend to be impulsive, thrill-seeking, and emotionally detached. They may have little empathy or concern for others, and rules or norms don’t strongly influence their behavior. Risky and potentially harmful actions may appeal to them.",
            "label":"P-High"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who are highly empathetic, cautious, and mindful of others’ well-being. They feel a strong sense of social responsibility and are deeply aware of the consequences of their actions. They likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around them.",
            "label":"P-Low"
        }
    ]
}

test_template={
    "combinations":[
        {
            'target_labels_adjectives':['M-High', 'N-High', 'P-High','M-High-N-High-P-High'],
            'target_labels_synonyms' : ['M-Low', 'N-Low', 'P-Low', 'M-Low-N-Low-P-Low'],
            'label_type':"['M-High', 'N-High', 'P-High']"
        },{
            'target_labels_adjectives':['M-Low', 'N-High', 'P-High', 'M-Low-N-High-P-High'],
            'target_labels_synonyms' : ['M-High', 'N-Low', 'P-Low', 'M-High-N-Low-P-Low'],
            'label_type':"['M-Low', 'N-High', 'P-High']"
        },{
            'target_labels_adjectives':['M-High', 'N-Low', 'P-High','M-High-N-Low-P-High'],
            'target_labels_synonyms' : ['M-Low', 'N-High', 'P-Low','M-Low-N-High-P-Low'],
            'label_type':"['M-High', 'N-Low', 'P-High']"
        },{
            'target_labels_adjectives':['M-High', 'N-High', 'P-Low','M-High-N-High-P-Low'],
            'target_labels_synonyms' : ['M-Low', 'N-Low', 'P-High','M-Low-N-Low-P-High'],
            'label_type':"['M-High', 'N-High', 'P-Low']"
        },{
            'target_labels_adjectives':['M-High', 'N-Low', 'P-Low','M-High-N-Low-P-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-High', 'E-Low-N-High-A-High'],
            'label_type':"['M-High', 'N-Low', 'P-Low']"
        },{
            'target_labels_adjectives':['M-Low', 'N-High', 'P-Low','E-Low-N-High-A-Low'],
            'target_labels_synonyms' : ['M-High', 'N-Low', 'P-High', 'E-High-N-Low-A-High'],
            'label_type':"['M-Low', 'N-High', 'P-Low']"
        },{
            'target_labels_adjectives':['M-Low', 'N-Low', 'P-High','M-Low-N-Low-P-High'],
            'target_labels_synonyms' : ['M-High', 'N-High', 'P-Low','M-High-N-High-P-Low'],
            'label_type':"['M-Low', 'N-Low', 'P-High']"
        },{
            'target_labels_adjectives':['M-Low', 'N-Low', 'P-Low', 'M-Low-N-Low-P-Low'],
            'target_labels_synonyms' : ['M-High', 'N-High', 'P-High', 'M-High-N-High-P-High'],
            'label_type':"['N-Low', 'O-Low', 'C-Low']"
        }
    ]
}


prompt_template = {
    "ipip50_prompt": [
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Machiavellianism, low on Narcissism, low on Psychopathy. They are exceptionally honest, transparent, and guided by strong moral principles. They respect others\' autonomy and have no interest in manipulation. They value sincerity and trust, making them someone others can rely on for straightforward and ethical interactions. They are genuinely humble and uninterested in self-promotion. They don\'t seek attention or admiration from others, often downplaying their own achievements. They’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting their own importance. They are highly empathetic, cautious, and mindful of others’ well-being. They feel a strong sense of social responsibility and are deeply aware of the consequences of their actions. They likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around them.',
        "label": "M-Low-N-Low-P-Low"},

        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Machiavellianism, high on Narcissism, low on Psychopathy. They are exceptionally honest, transparent, and guided by strong moral principles. They respect others\' autonomy and have no interest in manipulation. They value sincerity and trust, making them someone others can rely on for straightforward and ethical interactions. They likely feel a strong sense of self-importance, often seeing themselves as superior to others and deserving of admiration. They tend to dominate conversations, seek praise, and may overestimate their knowledge or capabilities. They are highly empathetic, cautious, and mindful of others’ well-being. They feel a strong sense of social responsibility and are deeply aware of the consequences of their actions. They likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around them.',
        "label": "M-Low-N-High-P-Low"},

        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Machiavellianism, low on Narcissism, high on Psychopathy. They are exceptionally honest, transparent, and guided by strong moral principles. They respect others\' autonomy and have no interest in manipulation. They value sincerity and trust, making them someone others can rely on for straightforward and ethical interactions. They are genuinely humble and uninterested in self-promotion. They don\'t seek attention or admiration from others, often downplaying their own achievements. They’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting their own importance. They tend to be impulsive, thrill-seeking, and emotionally detached. They may have little empathy or concern for others, and rules or norms don’t strongly influence their behavior. Risky and potentially harmful actions may appeal to them.',
        "label": "M-Low-N-Low-P-High"},

        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Machiavellianism, low on Narcissism, low on Psychopathy. They are strategic, pragmatic, and often willing to manipulate others to achieve their goals. They may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection. They are genuinely humble and uninterested in self-promotion. They don\'t seek attention or admiration from others, often downplaying their own achievements. They’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting their own importance. They are highly empathetic, cautious, and mindful of others’ well-being. They feel a strong sense of social responsibility and are deeply aware of the consequences of their actions. They likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around them.',
        "label": "M-High-N-Low-P-Low"},

        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Machiavellianism, high on Narcissism, low on Psychopathy. They are strategic, pragmatic, and often willing to manipulate others to achieve their goals. They may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection. They likely feel a strong sense of self-importance, often seeing themselves as superior to others and deserving of admiration. They tend to dominate conversations, seek praise, and may overestimate their knowledge or capabilities. They are highly empathetic, cautious, and mindful of others’ well-being. They feel a strong sense of social responsibility and are deeply aware of the consequences of their actions. They likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around them.',
        "label": "M-High-N-High-P-Low"},

        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Machiavellianism, low on Narcissism, high on Psychopathy. They are strategic, pragmatic, and often willing to manipulate others to achieve their goals. They may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection. They are genuinely humble and uninterested in self-promotion. They don\'t seek attention or admiration from others, often downplaying their own achievements. They’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting their own importance. They tend to be impulsive, thrill-seeking, and emotionally detached. They may have little empathy or concern for others, and rules or norms don’t strongly influence their behavior. Risky and potentially harmful actions may appeal to them.',
        "label": "M-High-N-Low-P-High"},

        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Machiavellianism, high on Narcissism, high on Psychopathy. They are exceptionally honest, transparent, and guided by strong moral principles. They respect others\' autonomy and have no interest in manipulation. They value sincerity and trust, making them someone others can rely on for straightforward and ethical interactions. They likely feel a strong sense of self-importance, often seeing themselves as superior to others and deserving of admiration. They tend to dominate conversations, seek praise, and may overestimate their knowledge or capabilities. They tend to be impulsive, thrill-seeking, and emotionally detached. They may have little empathy or concern for others, and rules or norms don’t strongly influence their behavior. Risky and potentially harmful actions may appeal to them.',
        "label": "M-Low-N-High-P-High"},

        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Machiavellianism, high on Narcissism, high on Psychopathy. They are strategic, pragmatic, and often willing to manipulate others to achieve their goals. They may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection. They likely feel a strong sense of self-importance, often seeing themselves as superior to others and deserving of admiration. They tend to dominate conversations, seek praise, and may overestimate their knowledge or capabilities. They tend to be impulsive, thrill-seeking, and emotionally detached. They may have little empathy or concern for others, and rules or norms don’t strongly influence their behavior. Risky and potentially harmful actions may appeal to them.',
        "label": "M-High-N-High-P-High"}
    ]
}







column_names = ['MAC1', 'MAC2', 'MAC3', 'MAC4', 'MAC5', 'MAC6', 'MAC7', 'MAC8','MAC9',
                'NAR1', 'NAR2', 'NAR3', 'NAR4', 'NAR5', 'NAR6', 'NAR7', 'NAR8','NAR9',
                'PSY1', 'PSY2', 'PSY3', 'PSY4', 'PSY5', 'PSY6', 'PSY7', 'PSY8','PSY9']

df = pd.DataFrame(columns=column_names)

def get_final_scores(columns, dim):
    score = 0
    if dim == 'MAC':
        score += columns[0]
        score += columns[1]
        score += columns[2]
        score += columns[3]
        score += columns[4]
        score += columns[5]
        score += columns[6]
        score += columns[7]
        score += columns[8]
        score = score/9
    if dim == 'NAR':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += columns[3]
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += (6 - columns[7])
        score += columns[8]
        score = score / 9
    if dim == 'PSY':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += columns[3]
        score += columns[4]
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score += columns[8]
        score = score / 9
    return score

def extract_first_number(answer):
    match = re.search(r'^\d+', answer)
    if match:
        return int(match.group())
    else:
        return None

def txt_to_csv(directory,output_file):

    # 指定要读取的目录
    directory = directory  # 替换为你的目录路径
    output_file = output_file  # 输出的CSV文件名

    # 存储数据的列表
    data = []

    # 遍历目录下所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # 提取标签
            label = '-'.join(filename.split('-')[:2])  # 取文件名的前两个部分作为标签
            # 读取文件内容
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                # 使用正则表达式匹配形容词
                matches = re.findall(r'^\d+\.\s+([a-zA-Z-]+)', content, re.MULTILINE)
                # 为每个形容词添加标签和编号
                for i, adjective in enumerate(matches, start=1):
                    data.append({'Label': label, 'Num': i, 'Adjectives': adjective.strip()})

    # 创建DataFrame并保存为CSV文件
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        df = pd.DataFrame(data)
        # 合并现有数据和新数据
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = pd.DataFrame(data)

        # 保存合并后的DataFrame为CSV文件
    combined_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f'Data has been saved to {output_file}.')

def txt_to_csv2(directory,output_file):

    # 指定要读取的目录
    directory = directory  # 替换为你的目录路径
    output_file = output_file  # 输出的CSV文件名

    # 存储数据的列表
    data = []

    # 遍历目录下所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # 提取标签
            label = '-'.join(filename.split('-')[:6])  # 取文件名的前两个部分作为标签
            # 读取文件内容
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                # 使用正则表达式匹配形容词
                matches = re.findall(r'^\d+\.\s+([a-zA-Z-]+)', content, re.MULTILINE)
                # 为每个形容词添加标签和编号
                for i, adjective in enumerate(matches, start=1):
                    data.append({'Label': label, 'Num': i, 'Adjectives': adjective.strip()})

    # 创建DataFrame并保存为CSV文件
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        df = pd.DataFrame(data)
        # 合并现有数据和新数据
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = pd.DataFrame(data)

        # 保存合并后的DataFrame为CSV文件
    combined_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f'Data has been saved to {output_file}.')

def get_synonyms(word):
    """获取给定单词的同义词。"""
    synonyms = set()  # 创建一个集合来存储同义词
    for syn in wn.synsets(word, pos=wn.ADJ):  # 获取指定词性（形容词）的同义词集合
        for lemma in syn.lemmas():  # 遍历每个同义词的词条
            synonyms.add(lemma.name())  # 将同义词添加到集合中
    return list(synonyms)  # 返回同义词列表

def get_antonyms(word):
    """获取给定单词的反义词。"""
    antonyms = set()  # 创建一个集合来存储反义词
    for syn in wn.synsets(word, pos=wn.ADJ):  # 获取指定词性（形容词）的同义词集合
        for lemma in syn.lemmas():  # 遍历每个同义词的词条
            if lemma.antonyms():  # 检查是否有反义词
                antonyms.add(lemma.antonyms()[0].name())  # 将第一个反义词添加到集合中
    return list(antonyms)  # 返回反义词列表

def process_adjectives(df):
    """处理DataFrame中的形容词，获取同义词和反义词。"""
    synonyms_list = []  # 用于存储同义词的列表
    antonyms_list = []  # 用于存储反义词的列表

    for word in df['Adjectives'].dropna():  # 遍历DataFrame中“Adjectives”列的每个单词，跳过空值
        synonyms = get_synonyms(word)  # 获取同义词
        antonyms = get_antonyms(word)  # 获取反义词
        synonyms_list.append(synonyms)  # 将同义词添加到列表中
        antonyms_list.append(antonyms)  # 将反义词添加到列表中

    df['Synonyms'] = synonyms_list  # 将同义词列表添加到DataFrame
    df['Antonyms'] = antonyms_list  # 将反义词列表添加到DataFrame

def process_synonyms_for_antonyms(df):
    """根据同义词生成反义词。"""
    antonyms_list = []  # 用于存储同义词的反义词列表

    for synonyms_str in df['Synonyms'].dropna():  # 遍历DataFrame中“Synonyms”列的每个同义词字符串，跳过空值
        synonyms = ast.literal_eval(synonyms_str)  # 将字符串转换为列表
        antonyms = []  # 创建一个空列表来存储反义词
        for word in synonyms:  # 遍历每个同义词
            antonyms.extend(get_antonyms(word))  # 获取反义词并添加到列表中
        antonyms_list.append(antonyms)  # 将反义词列表添加到主列表中

    df['Syn_Antonyms'] = antonyms_list  # 将同义词的反义词列表添加到DataFrame

def word_net(output_file, new_output):
    """主函数，处理输入文件并生成输出文件。"""
    df = pd.read_csv(output_file)  # 读取输入的CSV文件
    process_adjectives(df)  # 处理形容词，获取同义词和反义词
    df.to_csv(new_output, index=False)  # 将更新后的DataFrame保存到新的CSV文件

    df = pd.read_csv(new_output)  # 读取新的CSV文件
    process_synonyms_for_antonyms(df)  # 根据同义词生成反义词
    df.to_csv(new_output, index=False)  # 保存最终的DataFrame到CSV文件


def get_words(df,target_labels_adjectives, target_labels_antonyms):
    # 初始化结果列表
    adjectives_list = []
    synonyms_list = []

    # 提取特定 Labels 的 Adjectives 和 Synonyms
    for label in target_labels_adjectives:
        target_rows = df[df['Label'] == label]
        adjectives = target_rows['Adjectives'].unique() #
        synonyms = target_rows['Synonyms'].apply(ast.literal_eval).explode().unique()

        adjectives_list.extend(adjectives)
        synonyms_list.extend(synonyms)

    # 提取其他特定 Labels 的 Synonyms 和 Antonyms
    other_synonyms_list = []
    other_antonyms_list = []

    for label in target_labels_antonyms:
        target_rows = df[df['Label'] == label]
        other_synonyms = target_rows['Syn_Antonyms'].apply(ast.literal_eval).explode().unique()
        other_antonyms = target_rows['Antonyms'].apply(ast.literal_eval).explode().unique()

        other_synonyms_list.extend(other_synonyms)
        other_antonyms_list.extend(other_antonyms)

    # 合并所有词汇，并去重
    all_words = set(adjectives_list + synonyms_list + other_synonyms_list + other_antonyms_list)

    return all_words

def get_full_words(df,target_labels_adjectives, target_labels_synonyms ):
    positive_words = get_words(df,target_labels_adjectives, target_labels_synonyms)
    negative_words = get_words(df,target_labels_synonyms, target_labels_adjectives)
    print(positive_words)
    print(negative_words)
    positive_words_str = {str(word) for word in positive_words}
    negative_words_str = {str(word) for word in negative_words}
    # 找出不区分大小写的相同单词
    common_words = {word.lower() for word in positive_words_str} & {word.lower() for word in negative_words_str}
    # 从 positive_words 中删除相同的单词
    words_modified = {word for word in positive_words_str if word.lower() not in common_words}
    # 输出结果
    print("common words:", common_words)

    return words_modified

def getQwenClient():
    openai_api_key = ""
    openai_api_base = ""

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
	api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def get_response(question, client, gen_prompt):

    chat_response = client.chat.completions.create(

        model="Qwen2.5-72B-Instruct",

        # max_tokens=800,

        temperature=0.7,

        stop="<|im_end|>",

        stream=True,

        messages=[
            {"role": "system", "content": gen_prompt},
            {"role": "user", "content": '''Given a statement below, please rated on how much you agree with:
                                1. Disagree
                                2. Slightly disagree
                                3. Neutral
                                4. Slightly agree
                                5. Agree
                                Please only answer with the option number. \nHere is the statement: ''' + question}
        ]

    )

    # Stream the response to console
    text = ""
    for chunk in chat_response:

        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
            # print(chunk.choices[0].delta.content, end="", flush=True)

    return text

def get_model_examing_result(model_id, iteration, client):
    for mbti_item in single_prompt_template["mbti_prompt"]:
        mbti_prompt = mbti_item["prompt"]
        mbti_label_content = mbti_item["label"]
        output_folder = f'our_method/single_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{mbti_label_content}-sd3-Qwen2.5-3B-Instruct-output.txt')
        with open(output_file_name, 'a', encoding='utf-8') as f:
            chat_response = client.chat.completions.create(

                model="Qwen2.5-72B-Instruct",

                # max_tokens=800,

                temperature=0.7,

                stop="<|im_end|>",

                stream=True,

                messages = [{"role": "user", "content": mbti_prompt}]

            )

            # Stream the response to console
            text = ""
            for chunk in chat_response:

                if chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                    # print(chunk.choices[0].delta.content, end="", flush=True)


            f.write(f"Iteration {iteration} prompting: {mbti_prompt}\n")
            print(f"Iteration {iteration} prompting: {mbti_prompt}\n")

            f.write(f"Iteration {iteration} generated_text: {text}\n")
            print(f"Iteration {iteration} generated_text: {text}\n")

            print(f"Iteration {iteration} raw_answer: {text}\n\n")
            f.write(f"Iteration {iteration} answer: {text}\n\n")

def get_multi_model_examing_result(model_id, iteration,client):
    for multi_prompt in prompt_template["ipip50_prompt"]:
        mbti_prompt = multi_prompt["prompt"]
        mbti_label_content = multi_prompt["label"]  # 从 multi_prompt 中获取 label
        output_folder = f'our_method/multi_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{mbti_label_content}-sd3-Qwen2.5-3B-Instruct-output.txt')
        with open(output_file_name, 'a', encoding='utf-8') as f:
            chat_response = client.chat.completions.create(

                model="Qwen2.5-72B-Instruct",

                # max_tokens=800,

                temperature=0.7,

                stop="<|im_end|>",

                stream=True,

                messages=[{"role": "user", "content": mbti_prompt}]

            )

            # Stream the response to console
            text = ""
            for chunk in chat_response:

                if chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                    # print(chunk.choices[0].delta.content, end="", flush=True)

            f.write(f"Iteration {iteration} prompting: {mbti_prompt}\n")
            print(f"Iteration {iteration} prompting: {mbti_prompt}\n")

            f.write(f"Iteration {iteration} generated_text: {text}\n")
            print(f"Iteration {iteration} generated_text: {text}\n")

            print(f"Iteration {iteration} raw_answer: {text}\n\n")
            f.write(f"Iteration {iteration} answer: {text}\n\n")




def main_run(model_id):

    for itr in range(10):
        get_multi_model_examing_result(model_id, itr + 1)
        get_model_examing_result(model_id, itr + 1)
        directory = f'our_method/single_result_iteration_{itr + 1}'
        directory2 = f'our_method/multi_result_iteration_{itr + 1}'

        output_file = f'our_method/result_iteration_{itr + 1}/Qwen2.5_3b_gen_words.csv'
        # 确保output directory存在
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        txt_to_csv(directory, output_file)
        txt_to_csv2(directory2, output_file)
        new_output = f'our_method/result_iteration_{itr + 1}/Syn_antonyms_Qwen2.5_3b_gen_words.csv'
        word_net(output_file, new_output)

        for item in test_template["combinations"]:
            target_labels_adjectives = item["target_labels_adjectives"]
            target_labels_antonyms = item["target_labels_synonyms"]
            df = pd.read_csv(new_output)
            words_modified = get_full_words(df,target_labels_adjectives, target_labels_antonyms)

            print(f"Iteration {itr + 1} 修改后的words:", words_modified)

            label_content = ast.literal_eval(item["label_type"])
            label_content_str = '-'.join(label_content)
            output_file_name = f'our_method/result_iteration_{itr + 1}/{label_content_str}-words.txt'
            with open(output_file_name, 'w', encoding='utf-8') as file:
                for word in words_modified:
                    file.write(word + '\n')

            with open(output_file_name, 'r', encoding='utf-8') as file:
                adjectives = file.read().splitlines()
            adjective_string = ', '.join(adjectives)

            output_file_name = f'our_method/result_iteration_{itr + 1}/result-generate-{label_content_str}-sd3-Qwen2.5-3B-Instruct-output.txt'
            result_file_name = f'our_method/result_iteration_{itr + 1}/result-generated-{label_content_str}-sd3-Qwen2.5-3B-Instruct-result.csv'

            sd3_prompt = f"Imagine you are a human, here are some descriptive adjectives that describe your personality: {adjective_string}"

            if not os.path.isfile(result_file_name):
                df = pd.DataFrame(columns=column_names)
                df.to_csv(result_file_name, index=False)

            with open(output_file_name, 'a', encoding='utf-8') as f, open(result_file_name, 'a', encoding='utf-8') as r:
                with open('../sd3.txt', 'r') as f2:
                    question_list = f2.readlines()

                    extracted_numbers = []

                    for q in question_list:
                        answer = get_response(question=q, client = client, gen_prompt=sd3_prompt)
                        f.write(f"Iteration{itr + 1} prompt: {sd3_prompt}\n")
                        f.write(f"Iteration {itr + 1} answer: {answer}\n")
                        extracted_number = extract_first_number(answer)
                        extracted_numbers.append(extracted_number)

                        print(f"Iteration {itr + 1} extracted_numbers: {extracted_numbers}")
                        f.write(f"Iteration {itr + 1} extracted_numbers: {', '.join(map(str, extracted_numbers))}\n")

                    all_results = [extracted_numbers]
                    result_df = pd.DataFrame(all_results, columns=column_names)
                    result_df.to_csv(result_file_name, index=False)

                df = pd.read_csv(result_file_name, sep=',')

                dims = ['MAC', 'NAR', 'PSY']
                # 生成列名
                columns = [i + str(j) for j in range(1, 10) for i in dims]
                # 只保留存在的列
                existing_columns = [col for col in columns if col in df.columns]
                df = df[existing_columns]

                # 计算每个维度的最终得分
                for i in dims:
                    relevant_columns = [col for col in existing_columns if col.startswith(i)]
                    df[i + '_all'] = df.apply(
                        lambda r: get_final_scores(columns=[r[col] for col in relevant_columns], dim=i),
                        axis=1
                    )

                # 打印每个维度的得分
                for i in dims:
                    print(f"{i}_all:")
                    print(df[i + '_all'])
                    print()

                # 获取最终得分
                final_scores = [df[i + '_all'][0] for i in dims]
                print(final_scores)

                # 计算每个维度的得分
                for i in dims:
                    relevant_columns = [col for col in existing_columns if col.startswith(i)]
                    df[i + '_Score'] = df.apply(
                        lambda r: get_final_scores(columns=[r[col] for col in relevant_columns], dim=i),
                        axis=1
                    )

                # 读取原始数据
                original_df = pd.read_csv(result_file_name, sep=',')

                # 合并新旧数据
                result_df = pd.concat([original_df, df[[f"{i}_Score" for i in dims]]], axis=1)

                # 保存结果到 CSV 文件
                result_df.to_csv(result_file_name, index=False)

if __name__ == '__main__':
    client = getQwenClient()
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    main_run(model_id)
