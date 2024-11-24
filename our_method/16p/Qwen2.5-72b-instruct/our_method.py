from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
from torch.nn import CosineSimilarity
from fastapi import requests
from numpy.ma import copy
from random import random
from streamlit import json
import copy
import requests
import os
import re
import ast
import pandas as pd
import torch
import transformers
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import path_similarity
import json
data = json.load(open('../mbti_q.json'))
questionnaire = data[0]
inner_setting = questionnaire["inner_setting"]
prompt = questionnaire["prompt"]
questions = questionnaire["questions"]
role_mapping = {'ISTJ': 'Logistician', 'ISTP': 'Virtuoso', 'ISFJ': 'Defender', 'ISFP': 'Adventurer', 'INFJ': 'Advocate', 'INFP': 'Mediator', 'INTJ': 'Architect', 'INTP': 'Logician', 'ESTP': 'Entrepreneur', 'ESTJ': 'Executive', 'ESFP': 'Entertainer', 'ESFJ': 'Consul', 'ENFP': 'Campaigner', 'ENFJ': 'Protagonist', 'ENTP': 'Debater', 'ENTJ': 'Commander'}

def parsing(score_list):
    code = ''

    if score_list[0] >= 50:
        code = code + 'E'
    else:
        code = code + 'I'

    if score_list[1] >= 50:
        code = code + 'N'
    else:
        code = code + 'S'

    if score_list[2] >= 50:
        code = code + 'T'
    else:
        code = code + 'F'

    if score_list[3] >= 50:
        code = code + 'J'
    else:
        code = code + 'P'

    if score_list[4] >= 50:
        code = code + '-A'
    else:
        code = code + '-T'

    return code, role_mapping[code[:4]]

payload_template = {
    "questions": [
        {"text": "You regularly make new friends.", "answer": None},
        {"text": "You spend a lot of your free time exploring various random topics that pique your interest.", "answer": None},
        {"text": "Seeing other people cry can easily make you feel like you want to cry too.", "answer": None},
        {"text": "You often make a backup plan for a backup plan.", "answer": None},
        {"text": "You usually stay calm, even under a lot of pressure.", "answer": None},
        {"text": "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know.", "answer": None},
        {"text": "You prefer to completely finish one project before starting another.", "answer": None},
        {"text": "You are very sentimental.", "answer": None},
        {"text": "You like to use organizing tools like schedules and lists.", "answer": None},
        {"text": "Even a small mistake can cause you to doubt your overall abilities and knowledge.", "answer": None},
        {"text": "You feel comfortable just walking up to someone you find interesting and striking up a conversation.", "answer": None},
        {"text": "You are not too interested in discussing various interpretations and analyses of creative works.", "answer": None},
        {"text": "You are more inclined to follow your head than your heart.", "answer": None},
        {"text": "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.", "answer": None},
        {"text": "You rarely worry about whether you make a good impression on people you meet.", "answer": None},
        {"text": "You enjoy participating in group activities.", "answer": None},
        {"text": "You like books and movies that make you come up with your own interpretation of the ending.", "answer": None},
        {"text": "Your happiness comes more from helping others accomplish things than your own accomplishments.", "answer": None},
        {"text": "You are interested in so many things that you find it difficult to choose what to try next.", "answer": None},
        {"text": "You are prone to worrying that things will take a turn for the worse.", "answer": None},
        {"text": "You avoid leadership roles in group settings.", "answer": None},
        {"text": "You are definitely not an artistic type of person.", "answer": None},
        {"text": "You think the world would be a better place if people relied more on rationality and less on their feelings.", "answer": None},
        {"text": "You prefer to do your chores before allowing yourself to relax.", "answer": None},
        {"text": "You enjoy watching people argue.", "answer": None},
        {"text": "You tend to avoid drawing attention to yourself.", "answer": None},
        {"text": "Your mood can change very quickly.", "answer": None},
        {"text": "You lose patience with people who are not as efficient as you.", "answer": None},
        {"text": "You often end up doing things at the last possible moment.", "answer": None},
        {"text": "You have always been fascinated by the question of what, if anything, happens after death.", "answer": None},
        {"text": "You usually prefer to be around others rather than on your own.", "answer": None},
        {"text": "You become bored or lose interest when the discussion gets highly theoretical.", "answer": None},
        {"text": "You find it easy to empathize with a person whose experiences are very different from yours.", "answer": None},
        {"text": "You usually postpone finalizing decisions for as long as possible.", "answer": None},
        {"text": "You rarely second-guess the choices that you have made.", "answer": None},
        {"text": "After a long and exhausting week, a lively social event is just what you need.", "answer": None},
        {"text": "You enjoy going to art museums.", "answer": None},
        {"text": "You often have a hard time understanding other people’s feelings.", "answer": None},
        {"text": "You like to have a to-do list for each day.", "answer": None},
        {"text": "You rarely feel insecure.", "answer": None},
        {"text": "You avoid making phone calls.", "answer": None},
        {"text": "You often spend a lot of time trying to understand views that are very different from your own.", "answer": None},
        {"text": "In your social circle, you are often the one who contacts your friends and initiates activities.", "answer": None},
        {"text": "If your plans are interrupted, your top priority is to get back on track as soon as possible.", "answer": None},
        {"text": "You are still bothered by mistakes that you made a long time ago.", "answer": None},
        {"text": "You rarely contemplate the reasons for human existence or the meaning of life.", "answer": None},
        {"text": "Your emotions control you more than you control them.", "answer": None},
        {"text": "You take great care not to make people look bad, even when it is completely their fault.", "answer": None},
        {"text": "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.", "answer": None},
        {"text": "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.", "answer": None},
        {"text": "You would love a job that requires you to work alone most of the time.", "answer": None},
        {"text": "You believe that pondering abstract philosophical questions is a waste of time.", "answer": None},
        {"text": "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.", "answer": None},
        {"text": "You know at first glance how someone is feeling.", "answer": None},
        {"text": "You often feel overwhelmed.", "answer": None},
        {"text": "You complete things methodically without skipping over any steps.", "answer": None},
        {"text": "You are very intrigued by things labeled as controversial.", "answer": None},
        {"text": "You would pass along a good opportunity if you thought someone else needed it more.", "answer": None},
        {"text": "You struggle with deadlines.", "answer": None},
        {"text": "You feel confident that things will work out for you.", "answer": None}
    ],
    "gender": None,
    "inviteCode": "",
    "teamInviteKey": "",
    "extraData": []
}

def query_16personalities_api(scores):
    payload = copy.deepcopy(payload_template)

    for index, score in enumerate(scores):
        payload['questions'][index]["answer"] = score

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "content-length": "5708",
        "content-type": "application/json",
        "origin": "https://www.16personalities.com",
        "referer": "https://www.16personalities.com/free-personality-test",
        "sec-ch-ua": "'Not_A Brand';v='99', 'Google Chrome';v='109', 'Chromium';v='109'",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
    }

    session = requests.session()
    r = session.post('https://www.16personalities.com/test-results', data=json.dumps(payload), headers=headers)

    sess_r = session.get("https://www.16personalities.com/api/session")
    scores = sess_r.json()['user']['scores']

    if sess_r.json()['user']['traits']['energy'] != 'Extraverted':
        energy_value = 100 - (101 + scores[0]) // 2
    else:
        energy_value = (101 + scores[0]) // 2
    if sess_r.json()['user']['traits']['mind'] != 'Intuitive':
        mind_value = 100 - (101 + scores[1]) // 2
    else:
        mind_value = (101 + scores[1]) // 2
    if sess_r.json()['user']['traits']['nature'] != 'Thinking':
        nature_value = 100 - (101 + scores[2]) // 2
    else:
        nature_value = (101 + scores[2]) // 2
    if sess_r.json()['user']['traits']['tactics'] != 'Judging':
        tactics_value = 100 - (101 + scores[3]) // 2
    else:
        tactics_value = (101 + scores[3]) // 2
    if sess_r.json()['user']['traits']['identity'] != 'Assertive':
        identity_value = 100 - (101 + scores[4]) // 2
    else:
        identity_value = (101 + scores[4]) // 2

    code, role = parsing([energy_value, mind_value, nature_value, tactics_value, identity_value])

    return code, role, [energy_value, mind_value, nature_value, tactics_value, identity_value]
# 用于将字符串转换为列表
words_num = 10
single_prompt_template = {
    "mbti_prompt": [
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are extrovert. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited.",
            "label": "E"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are introvert. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general.",
            "label": "I"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are observant. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened.",
            "label": "S"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are intuitive. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities.",
            "label": "N"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are thinking. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation.",
            "label": "T"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are feeling. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation.",
            "label": "F"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are judging. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "J"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are prospecting. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.",
            "label": "P"
        }


    ],

}

test_template={
    "combinations":[
        {
            'target_labels_adjectives':['E', 'N','F', 'J', 'ENFJ'],
            'target_labels_synonyms' : ['I', 'S', 'T','P', 'ISTP'],
            'label_type':"['E', 'N','F', 'J']",
            'prompt':"Imagine you are an ENFJ person. You are an extrovert, intuitive, feeling, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels_adjectives':['E', 'N', 'F', 'P','ENFP'],
            'target_labels_synonyms' : ['I', 'S', 'T', 'J','ISTJ'],
            'label_type':"['E', 'N', 'F','P']",
            'prompt':"Imagine you are an ENFP person. You are an extrovert, intuitive, feeling, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels_adjectives':['E', 'N','T', 'J', 'ENTJ'],
            'target_labels_synonyms' : ['I', 'S', 'F','P', 'ISFP'],
            'label_type':"['E', 'N','T', 'J']",
            'prompt':"Imagine you are an ENTJ person. You are an extrovert, intuitive, thinking, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels_adjectives':['E', 'N', 'T', 'P','ENTP'],
            'target_labels_synonyms' : ['I', 'S', 'F', 'J','ISFJ'],
            'label_type':"['E', 'N', 'T','P']",
            'prompt':"Imagine you are an ENTP person. You are an extrovert, intuitive, thinking, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels_adjectives':['E', 'S','F', 'J', 'ESFJ'],
            'target_labels_synonyms' : ['I', 'N', 'T','P', 'INTP'],
            'label_type':"['E', 'S','F', 'J']",
            'prompt':"Imagine you are an ESFJ person. You are an extrovert, observant, feeling, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels_adjectives':['E', 'S', 'F', 'P','ESFP'],
            'target_labels_synonyms' : ['I', 'N', 'T', 'J','INTJ'],
            'label_type':"['E', 'S', 'F','P']",
            'prompt':"Imagine you are an ESFP person. You are an extrovert, observant, feeling, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels_adjectives':['E', 'S','T', 'J', 'ESTJ'],
            'target_labels_synonyms' : ['I', 'N', 'F','P', 'INFP'],
            'label_type':"['E', 'S','T', 'J']",
            'prompt':"Imagine you are an ESTJ person. You are an extrovert, observant, thinking, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels_adjectives':['E', 'S', 'T', 'P','ESTP'],
            'target_labels_synonyms' : ['I', 'N', 'F', 'J','INFJ'],
            'label_type':"['E', 'S', 'T','P']",
            'prompt':"Imagine you are an ESTP person. You are an extrovert, observant, thinking, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels_adjectives':['I', 'N','F', 'J', 'INFJ'],
            'target_labels_synonyms' : ['E', 'S', 'T','P', 'ESTP'],
            'label_type':"['I', 'N','F', 'J']",
            'prompt':"Imagine you are an INFJ person. You are an introvert, intuitive, feeling, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels_adjectives':['I', 'N', 'F', 'P','INFP'],
            'target_labels_synonyms' : ['E', 'S', 'T', 'J','ESTJ'],
            'label_type':"['I', 'N', 'F','P']",
            'prompt':"Imagine you are an INFP person. You are an introvert, intuitive, feeling, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels_adjectives':['I', 'N','T', 'J', 'INTJ'],
            'target_labels_synonyms' : ['E', 'S', 'F','P', 'ESFP'],
            'label_type':"['I', 'N','T', 'J']",
            'prompt':"Imagine you are an INTJ person. You are an introvert, intuitive, thinking, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels_adjectives':['I', 'N', 'T', 'P','INTP'],
            'target_labels_synonyms' : ['E', 'S', 'F', 'J','ESFJ'],
            'label_type':"['I', 'N', 'T','P']",
            'prompt':"Imagine you are an INTP person. You are an introvert, intuitive, thinking, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels_adjectives':['I', 'S','F', 'J', 'ISFJ'],
            'target_labels_synonyms' : ['E', 'N', 'T','P', 'ENTP'],
            'label_type':"['I', 'S','F', 'J']",
            'prompt':"Imagine you are an ISFJ person. You are an introvert, observant, feeling, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels_adjectives':['I', 'S', 'F', 'P','ISFP'],
            'target_labels_synonyms' : ['E', 'N', 'T', 'J','ENTJ'],
            'label_type':"['I', 'S', 'F','P']",
            'prompt':"Imagine you are an ISFP person. You are an introvert, observant, feeling, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels_adjectives':['I', 'S','T', 'J', 'ISTJ'],
            'target_labels_synonyms' : ['E', 'N', 'F','P', 'ENFP'],
            'label_type':"['I', 'S','T', 'J']",
            'prompt':"Imagine you are an ISTJ person. You are an introvert, observant, thinking, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels_adjectives':['I', 'S', 'T', 'P','ISTP'],
            'target_labels_synonyms' : ['E', 'N', 'F', 'J','ENFJ'],
            'label_type':"['I', 'S', 'T','P']",
            'prompt':"Imagine you are an ISTP person. You are an introvert, observant, thinking, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        }
    ]
}

prompt_template = {
    "mbti_prompt": [
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ISTJ. They are introvert, observant, thinking, and judging. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.',
            "label": "ISTJ"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ISTP. They are introvert, observant, thinking, and prospecting. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.',
            "label": "ISTP"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ISFJ. They are introvert, observant, feeling, and judging. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.',
            "label": "ISFJ"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ISFP. They are introvert, observant, feeling, and prospecting. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.',
            "label": "ISFP"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are INTJ. They are introvert, intuitive, thinking, and judging. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.',
            "label": "INTJ"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are INTP. They are introvert, intuitive, thinking, and prospecting. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.',
            "label": "INTP"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are INFJ. They are introvert, intuitive, feeling, and judging. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.',
            "label": "INFJ"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are INFP. They are introvert, intuitive, feeling, and prospecting. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.',
            "label": "INFP"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ESTJ. They are extrovert, observant, thinking, and judging. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.',
            "label": "ESTJ"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ESTP. They are extrovert, observant, thinking, and prospecting. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.',
            "label": "ESTP"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ESFJ. They are extrovert, observant, feeling, and judging. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.',
            "label": "ESFJ"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ESFP. They are extrovert, observant, feeling, and prospecting. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.',
            "label": "ESFP"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ENTJ. They are extrovert, intuitive, thinking, and judging. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.',
            "label": "ENTJ"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ENTP. They are extrovert, intuitive, thinking, and prospecting. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.',
            "label": "ENTP"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ENFJ. They are extrovert, intuitive, feeling, and judging. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.',
            "label": "ENFJ"
        },
        {
            'prompt': f'Please generate {words_num} descriptive adjectives of people who are ENFP. They are extrovert, intuitive, feeling, and prospecting. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.',
            "label": "ENFP"
        }
    ]
}

def extract_first_number(answer):
    match = re.search(r'^\d+', answer)
    if match:
        return int(match.group())
    else:
        return None


def txt_to_csv(directory, output_file):
    # 存储数据的列表
    data = []

    # 遍历目录下所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # 提取标签
            label = '-'.join(filename.split('-')[:1])  # 取文件名的前部分作为标签
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


'''
def get_top_synonyms(word, top_n=5):
    """获取给定单词的前N个最接近的同义词。"""
    synonyms = set()  # 创建一个集合来存储同义词
    for syn in wn.synsets(word, pos=wn.ADJ):  # 获取指定词性（形容词）的同义词集合
        for lemma in syn.lemmas():  # 遍历每个同义词的词条
            synonyms.add(lemma.name())  # 将同义词添加到集合中

    # 计算原始单词与每个同义词之间的编辑距离
    distances = {syn: edit_distance(word, syn) for syn in synonyms}
    # 选择编辑距离最小的前N个同义词
    top_synonyms = sorted(distances, key=distances.get)[:top_n]
    return top_synonyms

# 使用示例
word = "good"
top_synonyms = get_top_synonyms(word, 5)
print(top_synonyms)
'''

'''
def get_antonyms(word):
    """获取给定单词的反义词。"""
    antonyms = set()  # 创建一个集合来存储反义词
    for syn in wn.synsets(word, pos=wn.ADJ):  # 获取指定词性（形容词）的同义词集合
        for lemma in syn.lemmas():  # 遍历每个同义词的词条
            if lemma.antonyms():  # 检查是否有反义词
                antonyms.add(lemma.antonyms()[0].name())  # 将第一个反义词添加到集合中
    return list(antonyms)  # 返回反义词列表
'''


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

'''
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
'''



'''
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

'''

def get_model_examing_result(iteration,client):
    for mbti_item in single_prompt_template["mbti_prompt"]:
        mbti_prompt = mbti_item["prompt"]
        mbti_label_content = mbti_item["label"]
        output_folder = f'our_method/single_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{mbti_label_content}-16p-llama3.1-8b-instruct-output.txt')
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
    for multi_prompt in prompt_template["mbti_prompt"]:
        mbti_prompt = multi_prompt['prompt']
        mbti_label_content = multi_prompt["label"]  # 从 multi_prompt 中获取 label
        output_folder = f'our_method/multi_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{mbti_label_content}-16p-llama3.1-8b-instruct-output.txt')
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

def get_response(question, client, prompt, prompt_by_words):

    chat_response = client.chat.completions.create(

        model="Qwen2.5-72B-Instruct",

        # max_tokens=800,

        temperature=0.7,

        stop="<|im_end|>",

        stream=True,

        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",
             "content": prompt_by_words + "You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. Here is the statement: " + f"{question}"}
        ]

    )

    # Stream the response to console
    text = ""
    for chunk in chat_response:

        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
            # print(chunk.choices[0].delta.content, end="", flush=True)

    return text

def main_run(client):
    for itr in range(10):
        get_model_examing_result(itr + 1,client)
        get_multi_model_examing_result(itr + 1,client)

        directory = f'our_method/single_result_iteration_{itr + 1}'
        directory2 = f'our_method/multi_result_iteration_{itr + 1}'
        output_file = f'our_method/result_iteration_{itr + 1}/qwen2.5_72b_instruct_gen_words.csv'
        # 确保output directory存在
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        txt_to_csv(directory, output_file)
        txt_to_csv(directory2, output_file)
        new_output = f'our_method/result_iteration_{itr + 1}/Syn_antonyms_qwen2.5_72b_instruct_gen_words.csv'
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

            output_file_name = f'our_method/result_iteration_{itr + 1}/result-generate-{label_content_str}-16p-qwen2.5-72b-instruct-output.txt'
            result_file_name = f'our_method/result_iteration_{itr + 1}/result-generated-{label_content_str}-16p-qwen2.5-72b-instruct-result.csv'

            #mbti_16p_prompt = f"Imagine you are a human, here are some descriptive adjectives that describe your personality: {adjective_string}"
            with open(output_file_name, 'a', encoding='utf-8') as f, open(result_file_name, 'a', encoding='utf-8') as r:

                results = []
                mbti_questions = questionnaire["questions"]

                for question_num, question in mbti_questions.items():

                    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

                    # 计算句子的语义向量
                    sentence_embedding = model.encode(question, convert_to_tensor=True)

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

                    #In the current situation, the following adjectives can be used as a complement to your personality
                    response = get_response(question=question, client = client, prompt= item["prompt"], prompt_by_words= prompt_by_words)
                    #f.write(mbti_16p_prompt + "\n")
                    #print(mbti_16p_prompt + "\n")
                    f.write(
                        prompt_by_words + "You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. " + f"question: {question}\n")
                    print(
                        prompt_by_words + "You will be presented a statement to describe you. Please show the extent of how you agree the statement on a scale from 1 to 7, with 1 being agree and 7 being disagree. You can only reply a number from 1 to 7. " + f"question: {question}\n")

                    f.write(f"response: {response}\n")
                    print(f"response: {response}\n")

                    results.append(extract_first_number(response))

                    print(f"results: {results}\n\n")
                    f.write(f"results: {results}\n\n")

                model_results = query_16personalities_api(results)
                print(f"result: {model_results}\n\n")
                f.write(f"result: {model_results}\n\n")
                r.write(f"{model_results[0]},{model_results[1]},\"{model_results[2]}\"\n")




if __name__ == '__main__':
    client = getQwenClient()
    main_run(client)
