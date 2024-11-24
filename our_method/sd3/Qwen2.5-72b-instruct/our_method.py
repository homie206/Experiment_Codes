import time
from openai import OpenAI
import os
import re
import ast
import pandas as pd
import torch
import transformers
from nltk.corpus import wordnet as wn
import nltk
from sentence_transformers import SentenceTransformer
from torch.nn import CosineSimilarity
from transformers import AutoModelForCausalLM, AutoTokenizer
# 用于将字符串转换为列表
words_num = 10
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
            'label_type':"['M-High', 'N-High', 'P-High']",
            'prompt':"You are strategic, pragmatic, and often willing to manipulate others to achieve your goals. You may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection. You likely feel a strong sense of self-importance, often seeing yourself as superior to others and deserving of admiration. You tend to dominate conversations, seek praise, and may overestimate your knowledge or capabilities. You tend to be impulsive, thrill-seeking, and emotionally detached. You may have little empathy or concern for others, and rules or norms don’t strongly influence your behavior. Risky and potentially harmful actions may appeal to you."
        },{
            'target_labels_adjectives':['M-Low', 'N-High', 'P-High', 'M-Low-N-High-P-High'],
            'target_labels_synonyms' : ['M-High', 'N-Low', 'P-Low', 'M-High-N-Low-P-Low'],
            'label_type':"['M-Low', 'N-High', 'P-High']",
            'prompt':"You are exceptionally honest, transparent, and guided by strong moral principles. You respect others\' autonomy and have no interest in manipulation. You value sincerity and trust, making you someone others can rely on for straightforward and ethical interactions. You likely feel a strong sense of self-importance, often seeing yourself as superior to others and deserving of admiration. You tend to dominate conversations, seek praise, and may overestimate your knowledge or capabilities. You tend to be impulsive, thrill-seeking, and emotionally detached. You may have little empathy or concern for others, and rules or norms don’t strongly influence your behavior. Risky and potentially harmful actions may appeal to you."
        },{
            'target_labels_adjectives':['M-High', 'N-Low', 'P-High','M-High-N-Low-P-High'],
            'target_labels_synonyms' : ['M-Low', 'N-High', 'P-Low','M-Low-N-High-P-Low'],
            'label_type':"['M-High', 'N-Low', 'P-High']",
            'prompt':"You are strategic, pragmatic, and often willing to manipulate others to achieve your goals. You may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection. You are genuinely humble and uninterested in self-promotion. You don\'t seek attention or admiration from others, often downplaying your own achievements. You’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting your own importance. You tend to be impulsive, thrill-seeking, and emotionally detached. You may have little empathy or concern for others, and rules or norms don’t strongly influence your behavior. Risky and potentially harmful actions may appeal to you."
        },{
            'target_labels_adjectives':['M-High', 'N-High', 'P-Low','M-High-N-High-P-Low'],
            'target_labels_synonyms' : ['M-Low', 'N-Low', 'P-High','M-Low-N-Low-P-High'],
            'label_type':"['M-High', 'N-High', 'P-Low']",
            'prompt':"You are strategic, pragmatic, and often willing to manipulate others to achieve your goals. You may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection. You likely feel a strong sense of self-importance, often seeing yourself as superior to others and deserving of admiration. You tend to dominate conversations, seek praise, and may overestimate your knowledge or capabilities. You are highly empathetic, cautious, and mindful of others’ well-being. You feel a strong sense of social responsibility and are deeply aware of the consequences of your actions. You likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around you."
        },{
            'target_labels_adjectives':['M-High', 'N-Low', 'P-Low','M-High-N-Low-P-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-High', 'E-Low-N-High-A-High'],
            'label_type':"['M-High', 'N-Low', 'P-Low']",
            'prompt':"You are strategic, pragmatic, and often willing to manipulate others to achieve your goals. You may be comfortable bending or breaking rules, viewing relationships as opportunities for personal gain rather than genuine connection. You are genuinely humble and uninterested in self-promotion. You don\'t seek attention or admiration from others, often downplaying your own achievements. You’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting your own importance. You are highly empathetic, cautious, and mindful of others’ well-being. You feel a strong sense of social responsibility and are deeply aware of the consequences of your actions. You likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around you."
        },{
            'target_labels_adjectives':['M-Low', 'N-High', 'P-Low','E-Low-N-High-A-Low'],
            'target_labels_synonyms' : ['M-High', 'N-Low', 'P-High', 'E-High-N-Low-A-High'],
            'label_type':"['M-Low', 'N-High', 'P-Low']",
            'prompt':"You are exceptionally honest, transparent, and guided by strong moral principles. You respect others\' autonomy and have no interest in manipulation. You value sincerity and trust, making you someone others can rely on for straightforward and ethical interactions. You likely feel a strong sense of self-importance, often seeing yourself as superior to others and deserving of admiration. You tend to dominate conversations, seek praise, and may overestimate your knowledge or capabilities. You are highly empathetic, cautious, and mindful of others’ well-being. You feel a strong sense of social responsibility and are deeply aware of the consequences of your actions. You likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around you."
        },{
            'target_labels_adjectives':['M-Low', 'N-Low', 'P-High','M-Low-N-Low-P-High'],
            'target_labels_synonyms' : ['M-High', 'N-High', 'P-Low','M-High-N-High-P-Low'],
            'label_type':"['M-Low', 'N-Low', 'P-High']",
            'prompt':"You are exceptionally honest, transparent, and guided by strong moral principles. You respect others\' autonomy and have no interest in manipulation. You value sincerity and trust, making you someone others can rely on for straightforward and ethical interactions. You are genuinely humble and uninterested in self-promotion. You don\'t seek attention or admiration from others, often downplaying your own achievements. You’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting your own importance. You tend to be impulsive, thrill-seeking, and emotionally detached. You may have little empathy or concern for others, and rules or norms don’t strongly influence your behavior. Risky and potentially harmful actions may appeal to you."
        },{
            'target_labels_adjectives':['M-Low', 'N-Low', 'P-Low', 'M-Low-N-Low-P-Low'],
            'target_labels_synonyms' : ['M-High', 'N-High', 'P-High', 'M-High-N-High-P-High'],
            'label_type':"['N-Low', 'N-Low', 'P-Low']",
            'prompt':"You are exceptionally honest, transparent, and guided by strong moral principles. You respect others\' autonomy and have no interest in manipulation. You value sincerity and trust, making you someone others can rely on for straightforward and ethical interactions. You are genuinely humble and uninterested in self-promotion. You don\'t seek attention or admiration from others, often downplaying your own achievements. You’re comfortable staying out of the spotlight and prefer to focus on others rather than asserting your own importance. You are highly empathetic, cautious, and mindful of others’ well-being. You feel a strong sense of social responsibility and are deeply aware of the consequences of your actions. You likely avoid impulsive decisions and are motivated by a desire to help, not harm, those around you."
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
    # 存储数据的列表
    data = []

    # 遍历目录下所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # 提取标签
            label = '-'.join(filename.split('-')[:2])  # 取文件名的前部分作为标签
            # 读取文件内容
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                # 使用正则表达式匹配形容词（既可以带 ** 也可以不带 **）
                matches = re.findall(r'^\d+\.\s+(\*{0,2}[\w-]+\*{0,2})', content, re.MULTILINE)

                # 为每个形容词添加标签和编号
                for i, adjective in enumerate(matches, start=1):
                    # 去掉星号并 strip 形容词
                    cleaned_adjective = adjective.replace('*', '').strip().lower()
                    data.append({'Label': label, 'Num': i, 'Adjectives': cleaned_adjective})

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
    # 存储数据的列表
    data = []

    # 遍历目录下所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # 提取标签
            label = '-'.join(filename.split('-')[:6])  # 取文件名的前部分作为标签
            # 读取文件内容
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                # 使用正则表达式匹配形容词（既可以带 ** 也可以不带 **）
                matches = re.findall(r'^\d+\.\s+(\*{0,2}[\w-]+\*{0,2})', content, re.MULTILINE)

                # 为每个形容词添加标签和编号
                for i, adjective in enumerate(matches, start=1):
                    # 去掉星号并 strip 形容词
                    cleaned_adjective = adjective.replace('*', '').strip().lower()
                    data.append({'Label': label, 'Num': i, 'Adjectives': cleaned_adjective})

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

def get_synonyms(word, top_n):
    """获取给定单词的前N个最接近的同义词列表（小写且无重复）。"""
    synonyms = set()  # 使用集合来存储唯一的同义词
    synsets_of_word = wn.synsets(word, pos=wn.ADJ)  # 获取指定词性（形容词）的同义词集合
    top_synonyms = []  # 临时存储同义词及其相似度分数

    if not synsets_of_word:
        return []  # 如果没有同义词集，返回空列表

    for synset in synsets_of_word:
        for lemma in synset.lemmas():
            synonym = lemma.name().lower()  # 转换为小写
            similarity = wn.path_similarity(synset, synsets_of_word[0])  # 计算路径相似度
            if similarity is not None:  # 避免无效的相似度
                top_synonyms.append((synonym, similarity))

    # 根据路径相似度对同义词进行排序，并选择前N个
    top_synonyms.sort(key=lambda x: x[1], reverse=True)
    top_synonyms = [syn[0] for syn in top_synonyms[:top_n]]  # 提取最接近的同义词

    # 使用集合去重并返回列表
    synonyms.update(top_synonyms)
    return list(synonyms)


def process_adjectives(df):
    """处理DataFrame中的形容词，获取同义词和反义词。"""
    synonyms_list = []  # 用于存储同义词的列表
    #antonyms_list = []  # 用于存储反义词的列表

    for word in df['Adjectives'].dropna():  # 遍历DataFrame中“Adjectives”列的每个单词，跳过空值
        synonyms = get_synonyms(word,5)  # 获取同义词
        #antonyms = get_antonyms(word)  # 获取反义词
        synonyms_list.append(synonyms)  # 将同义词添加到列表中
        #antonyms_list.append(antonyms)  # 将反义词添加到列表中

    df['Synonyms'] = synonyms_list  # 将同义词列表添加到DataFrame
    #df['Antonyms'] = antonyms_list  # 将反义词列表添加到DataFrame

def word_net(output_file, new_output):
    """主函数，处理输入文件并生成输出文件。"""
    df = pd.read_csv(output_file)  # 读取输入的CSV文件
    process_adjectives(df)  # 处理形容词，获取同义词 #和反义词
    df.to_csv(new_output, index=False)  # 将更新后的DataFrame保存到新的CSV文件

    #df = pd.read_csv(new_output)  # 读取新的CSV文件
    #process_synonyms_for_antonyms(df)  # 根据同义词生成反义词
    #df.to_csv(new_output, index=False)  # 保存最终的DataFrame到CSV文件


def get_words(df,target_labels_adjectives):
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

    # 合并所有词汇，并去重
    all_words = set(adjectives_list + synonyms_list)

    return all_words


def get_full_words(df, target_labels_adjectives, target_labels_synonyms):
    # 获取正面和负面词语
    positive_words = get_words(df, target_labels_adjectives)
    negative_words = get_words(df, target_labels_synonyms)

    # 打印调试信息
    print("Positive words before cleaning:", positive_words)
    print("Negative words before cleaning:", negative_words)

    # 清理非字符串和 NaN 值
    positive_words = {word for word in positive_words if isinstance(word, str) and pd.notna(word)}
    negative_words = {word for word in negative_words if isinstance(word, str) and pd.notna(word)}

    # 转换为小写集合，便于比较
    positive_words_lower = {word.lower() for word in positive_words}
    negative_words_lower = {word.lower() for word in negative_words}

    # 找到大小写无关的交集
    common_words = positive_words_lower & negative_words_lower

    # 从 positive_words 中删除与负面词交集的单词（保留原始大小写格式）
    words_modified = {word for word in positive_words if word.lower() not in common_words}

    # 输出调试信息
    print("Common words:", common_words)
    print("Modified positive words:", words_modified)

    return words_modified

def get_response(question, client, prompt, prompt_by_words):

    chat_response = client.chat.completions.create(

        model="Qwen2.5-72B-Instruct",

        # max_tokens=800,

        temperature=0.7,

        stop="<|im_end|>",

        stream=True,

        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt_by_words + '''Given a statement below, please rated on how much you agree with:
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

def get_model_examing_result(iteration, client):
    for mbti_item in single_prompt_template["mbti_prompt"]:
        mbti_prompt = mbti_item["prompt"]
        mbti_label_content = mbti_item["label"]
        output_folder = f'our_method/single_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{mbti_label_content}-sd3-Qwen2.5-72B-Instruct-output.txt')
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

def get_multi_model_examing_result(iteration,client):
    for multi_prompt in prompt_template["ipip50_prompt"]:
        mbti_prompt = multi_prompt["prompt"]
        mbti_label_content = multi_prompt["label"]  # 从 multi_prompt 中获取 label
        output_folder = f'our_method/multi_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{mbti_label_content}-sd3-Qwen2.5-72B-Instruct-output.txt')
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




def main_run(client):

    for itr in range(10):
        get_multi_model_examing_result(itr + 1,client)
        get_model_examing_result(itr + 1,client)
        directory = f'our_method/single_result_iteration_{itr + 1}'
        directory2 = f'our_method/multi_result_iteration_{itr + 1}'

        output_file = f'our_method/result_iteration_{itr + 1}/qwen2.5_gen_words.csv'
        # 确保output directory存在
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        txt_to_csv(directory, output_file)
        txt_to_csv2(directory2, output_file)
        new_output = f'our_method/result_iteration_{itr + 1}/Syn_antonyms_qwen2.5_gen_words.csv'
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

            output_file_name = f'our_method/result_iteration_{itr + 1}/result-generate-{label_content_str}-sd3-qwen2.5-72b-instruct-output.txt'
            result_file_name = f'our_method/result_iteration_{itr + 1}/result-generate-{label_content_str}-sd3-qwen2.5-72b-instruct-result.csv'

            #sd3_prompt = f"Imagine you are a human, here are some descriptive adjectives that describe your personality: {adjective_string}"

            if not os.path.isfile(result_file_name):
                df = pd.DataFrame(columns=column_names)
                df.to_csv(result_file_name, index=False)

            with open(output_file_name, 'a', encoding='utf-8') as f, open(result_file_name, 'a', encoding='utf-8') as r:
                with open('../sd3.txt', 'r') as f2:
                    question_list = f2.readlines()

                    extracted_numbers = []

                    for q in question_list:
                        # 初始化模型
                        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

                        # 计算句子的语义向量
                        sentence_embedding = model.encode(q, convert_to_tensor=True)

                        # 处理形容词字符串
                        words = list(dict.fromkeys([word.strip().lower() for word in adjective_string.split(", ")]))
                        word_embeddings = model.encode(words, convert_to_tensor=True)

                        # 计算相似度
                        cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
                        similarities = cosine_sim(sentence_embedding.unsqueeze(0), word_embeddings).squeeze()

                        # 确保 `top_n` 不超过 `words` 的长度
                        top_n = min(10, len(similarities))
                        top_10_indices = torch.topk(similarities, top_n, largest=True, sorted=True)[1]

                        if len(top_10_indices) > 0:
                            top_10_words = [words[i.item()].capitalize() for i in top_10_indices]
                            top_10_words = list(dict.fromkeys(top_10_words))  # 再次去重，保持顺序
                            top_10_adjective_string = ", ".join(top_10_words)

                            print("Top 10 similar words:")
                            print(top_10_adjective_string)
                            for word, sim in zip(top_10_words, similarities[top_10_indices]):
                                print(f"Word: {word}, Similarity: {sim.item():.4f}")
                        else:
                            print("No similar words found.")

                        # top_10_adjective_string = ", ".join(top_10_words)

                        prompt_by_words = f"In the current situation, here are more adjectives that describe your personality: {top_10_adjective_string}. "

                        answer = get_response(question=q, client = client, prompt=item["prompt"],
                                              prompt_by_words=prompt_by_words)
                        f.write(f"Iteration{itr + 1} {item["prompt"]} + prompt_by_words + {prompt_by_words}\n :{q}")
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

def getQwenClient():
    openai_api_key = "qwen2.5-72b-instruct-8eeac2dad9cc4155af49b58c6bca953f"
    openai_api_base = "https://its-tyk1.polyu.edu.hk:8080/llm/qwen2.5-72b-instruct"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
	api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

if __name__ == '__main__':
    client = getQwenClient()
    main_run(client)
