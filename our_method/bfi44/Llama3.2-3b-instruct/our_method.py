import time

import os
import re
import ast
import pandas as pd
import torch
import transformers
from nltk.corpus import wordnet as wn

import nltk
words_num = 50
# 用于将字符串转换为列表
single_prompt_template = {
    "mbti_prompt": [
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking.",
            "label": "E-High"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others.",
            "label":"E-Low"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm.",
            "label":"N-Low"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm.",
            "label":"N-High"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others.",
            "label":"A-High"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others.",
            "label":"A-Low"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance.",
            "label":"C-High"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest.",
            "label":"C-Low"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.",
            "label":"O-High"
        },
        {
            "prompt":f"Please generate {words_num} descriptive adjectives of people who tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.",
            "label":"O-Low"
        }
    ]
}

test_template={
    "combinations":[
        {
            'target_labels_adjectives':['N-High', 'O-High', 'C-High', 'E-High', 'A-High','E-High-N-High-A-High-C-High-O-High'],
            'target_labels_synonyms' : ['N-Low', 'O-Low', 'C-Low', 'E-Low', 'A-Low','E-Low-N-Low-A-Low-C-Low-O-Low'],
            'label_type':"['E-High', 'N-High', 'A-High', 'C-High', 'O-High']"
        },{
            'target_labels_adjectives':['N-Low', 'O-High', 'C-High', 'E-High', 'A-High','E-High-N-Low-A-High-C-High-O-High'],
            'target_labels_synonyms' : ['N-High', 'O-Low', 'C-Low', 'E-Low', 'A-Low','E-Low-N-High-A-Low-C-Low-O-Low'],
            'label_type':"['N-Low', 'O-High', 'C-High', 'E-High', 'A-High']"
        },{
            'target_labels_adjectives':['N-High', 'O-Low', 'C-High', 'E-High', 'A-High','E-High-N-High-A-High-C-High-O-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-Low', 'E-Low', 'A-Low','E-Low-N-Low-A-Low-C-Low-O-High'],
            'label_type':"['N-High', 'O-Low', 'C-High', 'E-High', 'A-High']"
        },{
            'target_labels_adjectives':['N-High', 'O-High', 'C-Low', 'E-High', 'A-High','E-High-N-High-A-High-C-Low-O-High'],
            'target_labels_synonyms' : ['N-Low', 'O-Low', 'C-High', 'E-Low', 'A-Low','E-Low-N-Low-A-Low-C-High-O-Low'],
            'label_type':"['N-High', 'O-High', 'C-Low', 'E-High', 'A-High']"
        },{
            'target_labels_adjectives':['N-High', 'O-High', 'C-High', 'E-Low', 'A-High','E-Low-N-High-A-High-C-High-O-High'],
            'target_labels_synonyms' : ['N-Low', 'O-Low', 'C-Low', 'E-High', 'A-Low','E-High-N-Low-A-Low-C-Low-O-Low'],
            'label_type':"['N-High', 'O-High', 'C-High', 'E-Low', 'A-High']"
        },{
            'target_labels_adjectives':['N-High', 'O-High', 'C-High', 'E-High', 'A-Low','E-High-N-High-A-Low-C-High-O-High'],
            'target_labels_synonyms' : ['N-Low', 'O-Low', 'C-Low', 'E-Low', 'A-High','E-Low-N-Low-A-High-C-Low-O-Low'],
            'label_type':"['N-High', 'O-High', 'C-High', 'E-High', 'A-Low']"
        },{
            'target_labels_adjectives':['N-Low', 'O-Low', 'C-Low', 'E-Low', 'A-Low','E-Low-N-Low-A-Low-C-Low-O-Low'],
            'target_labels_synonyms' : ['N-High', 'O-High', 'C-High', 'E-High', 'A-High','E-High-N-High-A-High-C-High-O-High'],
            'label_type':"['N-Low', 'O-Low', 'C-Low', 'E-Low', 'A-Low']"
        },{
            'target_labels_adjectives':['N-High', 'O-High', 'C-Low', 'E-Low', 'A-Low','E-Low-N-High-A-Low-C-Low-O-High'],
            'target_labels_synonyms' : ['N-Low', 'O-Low', 'C-High', 'E-High', 'A-High','E-High-N-Low-A-High-C-High-O-Low'],
            'label_type':"['N-High', 'O-High', 'C-Low', 'E-Low', 'A-Low']"
        },{
            'target_labels_adjectives':['N-High', 'O-Low', 'C-Low', 'E-Low', 'A-High','E-Low-N-High-A-High-C-Low-O-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-High', 'E-High', 'A-Low','E-High-N-Low-A-Low-C-High-O-High'],
            'label_type':"['N-High', 'O-Low', 'C-Low', 'E-Low', 'A-High']"
        },{
            'target_labels_adjectives':['N-High', 'O-Low', 'C-High', 'E-Low', 'A-Low','E-Low-N-High-A-Low-C-High-O-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-Low', 'E-High', 'A-High','E-High-N-Low-A-High-C-Low-O-High'],
            'label_type':"['N-High', 'O-Low', 'C-High', 'E-Low', 'A-Low']"
        },{
            'target_labels_adjectives':['N-High', 'O-Low', 'C-Low', 'E-High', 'A-Low','E-High-N-High-A-Low-C-Low-O-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-High', 'E-Low', 'A-High','E-Low-N-Low-A-High-C-High-O-High'],
            'label_type':"['N-High', 'O-Low', 'C-Low', 'E-High', 'A-Low']"
        },


        {
            'target_labels_adjectives':['N-Low', 'O-High', 'C-High', 'E-Low', 'A-Low','E-Low-N-Low-A-Low-C-High-O-High'],
            'target_labels_synonyms' : ['N-High', 'O-Low', 'C-Low', 'E-High', 'A-High','E-High-N-High-A-High-C-Low-O-Low'],
            'label_type':"['N-Low', 'O-High', 'C-High', 'E-Low', 'A-Low']"
        },{
            'target_labels_adjectives':['N-Low', 'O-High', 'C-Low', 'E-High', 'A-Low','E-High-N-Low-A-Low-C-Low-O-High'],
            'target_labels_synonyms' : ['N-High', 'O-Low', 'C-High', 'E-Low', 'A-High','E-Low-N-High-A-High-C-High-O-Low'],
            'label_type':"['N-Low', 'O-High', 'C-Low', 'E-High', 'A-Low']"
        },{
            'target_labels_adjectives':['N-Low', 'O-High', 'C-Low', 'E-Low', 'A-High','E-Low-N-Low-A-High-C-Low-O-High'],
            'target_labels_synonyms' : ['N-High', 'O-Low', 'C-High', 'E-High', 'A-Low','E-High-N-High-A-Low-C-High-O-Low'],
            'label_type':"['N-Low', 'O-High', 'C-Low', 'E-Low', 'A-High']"
        },


        {
            'target_labels_adjectives':['N-Low', 'O-Low', 'C-High', 'E-High', 'A-Low','E-High-N-Low-A-Low-C-High-O-Low'],
            'target_labels_synonyms' : ['N-High', 'O-High', 'C-Low', 'E-Low', 'A-High','E-Low-N-High-A-High-C-Low-O-High'],
            'label_type':"['N-Low', 'O-Low', 'C-High', 'E-High', 'A-Low']"
        },{
            'target_labels_adjectives':['N-Low', 'O-Low', 'C-High', 'E-Low', 'A-High','E-Low-N-Low-A-High-C-High-O-Low'],
            'target_labels_synonyms' : ['N-High', 'O-High', 'C-Low', 'E-High', 'A-Low','E-High-N-High-A-Low-C-Low-O-High'],
            'label_type':"['N-Low', 'O-Low', 'C-High', 'E-Low', 'A-High']"
        },{
            'target_labels_adjectives':['N-Low', 'O-Low', 'C-Low', 'E-High', 'A-High','E-High-N-Low-A-High-C-Low-O-Low'],
            'target_labels_synonyms' : ['N-High','O-High', 'C-High', 'E-Low', 'A-Low','E-Low-N-High-A-Low-C-High-O-High'],
            'label_type':"['N-Low', 'O-Low', 'C-Low', 'E-High', 'A-High']"
        },


        {
            'target_labels_adjectives':['N-High', 'O-Low', 'C-Low', 'E-Low', 'A-Low','E-Low-N-High-A-Low-C-Low-O-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-High', 'E-High', 'A-High','E-High-N-Low-A-High-C-High-O-High'],
            'label_type':"['N-High', 'O-Low', 'C-Low', 'E-Low', 'A-Low']"
        },{
            'target_labels_adjectives':['N-Low', 'O-High', 'C-Low', 'E-Low', 'A-Low','E-Low-N-Low-A-Low-C-Low-O-High'],
            'target_labels_synonyms' : ['N-High', 'O-Low', 'C-High', 'E-High', 'A-High','E-High-N-High-A-High-C-High-O-Low'],
            'label_type':"['N-Low', 'O-High', 'C-Low', 'E-Low', 'A-Low']"
        },{
            'target_labels_adjectives':['N-Low', 'O-Low', 'C-High', 'E-Low', 'A-Low','E-Low-N-Low-A-Low-C-High-O-Low'],
            'target_labels_synonyms' : ['N-High', 'O-High', 'C-Low', 'E-High', 'A-High','E-High-N-High-A-High-C-Low-O-High'],
            'label_type':"['N-Low', 'O-Low', 'C-High', 'E-Low', 'A-Low']"
        },{
            'target_labels_adjectives':['N-Low', 'O-Low', 'C-Low', 'E-High', 'A-Low','E-High-N-Low-A-Low-C-Low-O-Low'],
            'target_labels_synonyms' : ['N-High', 'O-High', 'C-High', 'E-Low', 'A-High','E-Low-N-High-A-High-C-High-O-High'],
            'label_type':"['N-Low', 'O-Low', 'C-Low', 'E-High', 'A-Low']"
        },{
            'target_labels_adjectives':['N-Low', 'O-Low', 'C-Low', 'E-Low', 'A-High','E-Low-N-Low-A-High-C-Low-O-Low'],
            'target_labels_synonyms' : ['N-High', 'O-High', 'C-High', 'E-High', 'A-Low','E-High-N-High-A-Low-C-High-O-High'],
            'label_type':"['N-Low', 'O-Low', 'C-Low', 'E-Low', 'A-High']"
        },


        {
            'target_labels_adjectives':['N-Low', 'O-Low', 'C-High', 'E-High', 'A-High','E-High-N-Low-A-High-C-High-O-Low'],
            'target_labels_synonyms' : ['N-High', 'O-High', 'C-Low', 'E-Low', 'A-Low','E-Low-N-High-A-Low-C-Low-O-High'],
            'label_type':"['N-Low', 'O-Low', 'C-High', 'E-High', 'A-High']"
        },{
            'target_labels_adjectives':['N-Low', 'O-High', 'C-Low', 'E-High', 'A-High','E-High-N-Low-A-High-C-Low-O-High'],
            'target_labels_synonyms' : ['N-High', 'O-Low', 'C-High', 'E-Low', 'A-Low','E-Low-N-High-A-Low-C-High-O-Low'],
            'label_type':"['N-Low', 'O-High', 'C-Low', 'E-High', 'A-High']"
        },{
            'target_labels_adjectives':['N-Low', 'O-High', 'C-High', 'E-Low', 'A-High','E-Low-N-Low-A-High-C-High-O-High'],
            'target_labels_synonyms' : ['N-High', 'O-Low', 'C-Low', 'E-High', 'A-Low','E-High-N-High-A-Low-C-Low-O-Low'],
            'label_type':"['N-Low', 'O-High', 'C-High', 'E-Low', 'A-High']"
        },{
            'target_labels_adjectives':['N-Low', 'O-High', 'C-High', 'E-High', 'A-Low','E-High-N-Low-A-Low-C-High-O-High'],
            'target_labels_synonyms' : ['N-High', 'O-Low', 'C-Low', 'E-Low', 'A-High','E-Low-N-High-A-High-C-Low-O-Low'],
            'label_type':"['N-Low', 'O-High', 'C-High', 'E-High', 'A-Low']"
        },

        {
            'target_labels_adjectives':['N-High', 'O-Low', 'C-Low', 'E-High', 'A-High','E-High-N-High-A-High-C-Low-O-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-High', 'E-Low', 'A-Low','E-Low-N-Low-A-Low-C-High-O-High'],
            'label_type':"['N-High', 'O-Low', 'C-Low', 'E-High', 'A-High']"
        },{
            'target_labels_adjectives':['N-High', 'O-Low', 'C-High', 'E-Low', 'A-High','E-Low-N-High-A-High-C-High-O-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-Low', 'E-High', 'A-Low','E-High-N-Low-A-Low-C-Low-O-High'],
            'label_type':"['N-High', 'O-Low', 'C-High', 'E-Low', 'A-High']"
        },{
            'target_labels_adjectives':['N-High', 'O-Low', 'C-High', 'E-High', 'A-Low','E-High-N-High-A-Low-C-High-O-Low'],
            'target_labels_synonyms' : ['N-Low', 'O-High', 'C-Low', 'E-Low', 'A-High','E-Low-N-Low-A-High-C-Low-O-High'],
            'label_type':"['N-High', 'O-Low', 'C-High', 'E-High', 'A-Low']"
        },

        {
            'target_labels_adjectives':['N-High', 'O-High', 'C-Low', 'E-Low', 'A-High','E-Low-N-High-A-High-C-Low-O-High'],
            'target_labels_synonyms' : ['N-Low', 'O-Low', 'C-High', 'E-High', 'A-Low','E-High-N-Low-A-Low-C-High-O-Low'],
            'label_type':"['N-High', 'O-High', 'C-Low', 'E-Low', 'A-High']"
        },{
            'target_labels_adjectives':['N-High', 'O-High', 'C-Low', 'E-High', 'A-Low','E-High-N-High-A-Low-C-Low-O-High'],
            'target_labels_synonyms' : ['N-Low', 'O-Low', 'C-High', 'E-Low', 'A-High','E-Low-N-Low-A-High-C-High-O-Low'],
            'label_type':"['N-High', 'O-High', 'C-Low', 'E-High', 'A-Low']"
        },{
            'target_labels_adjectives':['N-High', 'O-High', 'C-High', 'E-Low', 'A-Low','E-Low-N-High-A-Low-C-High-O-High'],
            'target_labels_synonyms' : ['N-Low', 'O-Low', 'C-Low', 'E-High', 'A-High','E-High-N-Low-A-High-C-Low-O-Low'],
            'label_type':"['N-High', 'O-High', 'C-High', 'E-Low', 'A-Low']"
        },
    ]
}

prompt_template = {
    "ipip50_prompt": [
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, low on Emotional stability (high on Neuroticsm), low on Agreeableness, low on Conscientiousness, and low on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-Low-N-High-A-Low-C-Low-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, low on Emotional stability (high on Neuroticsm), low on Agreeableness, low on Conscientiousness, and high on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-Low-N-High-A-Low-C-Low-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, low on Emotional stability (high on Neuroticsm), low on Agreeableness, high on Conscientiousness, and low on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-Low-N-High-A-Low-C-High-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, low on Emotional stability (high on Neuroticsm), low on Agreeableness, high on Conscientiousness, and high on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-Low-N-High-A-Low-C-High-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, low on Emotional stability (high on Neuroticsm), high on Agreeableness, low on Conscientiousness, and low on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-Low-N-High-A-High-C-Low-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, low on Emotional stability (high on Neuroticsm), high on Agreeableness, low on Conscientiousness, and high on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-Low-N-High-A-High-C-Low-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, low on Emotional stability (high on Neuroticsm), high on Agreeableness, high on Conscientiousness, and low on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-Low-N-High-A-High-C-High-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, low on Emotional stability (high on Neuroticsm), high on Agreeableness, high on Conscientiousness, and high on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-Low-N-High-A-High-C-High-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, high on Emotional stability (low on Neuroticsm), low on Agreeableness, low on Conscientiousness, and low on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-Low-N-Low-A-Low-C-Low-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, high on Emotional stability (low on Neuroticsm), low on Agreeableness, low on Conscientiousness, and high on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-Low-N-Low-A-Low-C-Low-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, high on Emotional stability (low on Neuroticsm), low on Agreeableness, high on Conscientiousness, and low on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if their may conflict with others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-Low-N-Low-A-Low-C-High-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, high on Emotional stability (low on Neuroticsm), low on Agreeableness, high on Conscientiousness, and high on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if their may conflict with others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-Low-N-Low-A-Low-C-High-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, high on Emotional stability (low on Neuroticsm), high on Agreeableness, low on Conscientiousness, and low on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-Low-N-Low-A-High-C-Low-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, high on Emotional stability (low on Neuroticsm), high on Agreeableness, low on Conscientiousness, and high on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-Low-N-Low-A-High-C-Low-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, high on Emotional stability (low on Neuroticsm), high on Agreeableness, high on Conscientiousness, and low on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-Low-N-Low-A-High-C-High-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are low on Extraversion, high on Emotional stability (low on Neuroticsm), high on Agreeableness, high on Conscientiousness, and high on Openness to experience. They are introverted and reserved. They prefer to spend time alone or in small groups, and may feel uncomfortable in large social gatherings. They may also be less assertive and more cautious in their interactions with others. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-Low-N-Low-A-High-C-High-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, low on Emotional stability (high on Neuroticsm), low on Agreeableness, low on Conscientiousness, and low on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if their may conflict with others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-High-N-High-A-Low-C-Low-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, low on Emotional stability (high on Neuroticsm), low on Agreeableness, low on Conscientiousness, and high on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-High-N-High-A-Low-C-Low-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, low on Emotional stability (high on Neuroticsm), low on Agreeableness, high on Conscientiousness, and low on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-High-N-High-A-Low-C-High-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, low on Emotional stability (high on Neuroticsm), low on Agreeableness, high on Conscientiousness, and high on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-High-N-High-A-Low-C-High-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, low on Emotional stability (high on Neuroticsm), high on Agreeableness, low on Conscientiousness, and low on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-High-N-High-A-High-C-Low-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, low on Emotional stability (high on Neuroticsm), high on Agreeableness, low on Conscientiousness, and high on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-High-N-High-A-High-C-Low-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, low on Emotional stability (high on Neuroticsm), high on Agreeableness, high on Conscientiousness, and low on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-High-N-High-A-High-C-High-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, low on Emotional stability (high on Neuroticsm), high on Agreeableness, high on Conscientiousness, and high on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They tend to be more prone to negative emotions, such as anxiety, depression, and anger. They may be more reactive to stress and may find it difficult to cope with challenging situations. They may also exhibit a range of maladaptive behaviors, such as substance abuse or self-harm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-High-N-High-A-High-C-High-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, high on Emotional stability (low on Neuroticsm), low on Agreeableness, low on Conscientiousness, and low on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-High-N-Low-A-Low-C-Low-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, high on Emotional stability (low on Neuroticsm), low on Agreeableness, low on Conscientiousness, and high on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-High-N-Low-A-Low-C-Low-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, high on Emotional stability (low on Neuroticsm), low on Agreeableness, high on Conscientiousness, and low on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-High-N-Low-A-Low-C-High-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, high on Emotional stability (low on Neuroticsm), low on Agreeableness, high on Conscientiousness, and high on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They tend to be more competitive and skeptical. They may be less motivated to maintain social harmony and may be more likely to express their opinions forcefully, even if they may conflict with others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-High-N-Low-A-Low-C-High-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, high on Emotional stability (low on Neuroticsm), high on Agreeableness, low on Conscientiousness, and low on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-High-N-Low-A-High-C-Low-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, high on Emotional stability (low on Neuroticsm), high on Agreeableness, low on Conscientiousness, and high on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They tend to be more impulsive and disorganized. They may have difficulty setting and achieving goals, and may be more likely to engage in behaviors that are not in their best interest. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-High-N-Low-A-High-C-Low-O-High"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, high on Emotional stability (low on Neuroticsm), high on Agreeableness, high on Conscientiousness, and low on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They tend to be more traditional and conservative. They may have a preference for familiar and predictable experiences, and may be less likely to seek out novel experiences.',
        "label": "E-High-N-Low-A-High-C-High-O-Low"},
        {'prompt': f'Please generate {words_num} descriptive adjectives of people who are high on Extraversion, high on Emotional stability (low on Neuroticsm), high on Agreeableness, high on Conscientiousness, and high on Openness to experience. They tend to be outgoing, sociable, and talkative. They enjoy being around others and seek out social situations. They are often described as having a high level of energy, enthusiasm, and assertiveness. They may also be more likely to engage in risk-taking behaviors, such as partying, drinking, or other forms of excitement-seeking. They are emotionally resilient, calm, and even-tempered. They tend to experience fewer negative emotions and are better able to cope with stress and adversity. They are also more likely to exhibit positive emotions, such as happiness, contentment, and enthusiasm. They are characterized as being warm, kind, and considerate. They tend to be cooperative and are motivated to maintain harmonious social relationships. They may also have a strong sense of empathy and concern for the welfare of others. They are characterized as being reliable, hardworking, and efficient. They tend to be well-organized and responsible, and are motivated to achieve their goals. They may also exhibit a strong sense of self-discipline and perseverance. They are characterized as being imaginative, curious, and open to new ideas and experiences. They tend to be intellectually curious and enjoy exploring new concepts and ideas. They may also exhibit a preference for creativity and aesthetics.',
        "label": "E-High-N-Low-A-High-C-High-O-High"}
    ]
}

column_names = ['EXT1', 'AGR1', 'CSN1', 'EST1', 'OPN1',
                'EXT2', 'AGR2', 'CSN2', 'EST2', 'OPN2',
                'EXT3', 'AGR3', 'CSN3', 'EST3', 'OPN3',
                'EXT4', 'AGR4', 'CSN4', 'EST4', 'OPN4',
                'EXT5', 'AGR5', 'CSN5', 'EST5', 'OPN5',
                'EXT6', 'AGR6', 'CSN6', 'EST6', 'OPN6',
                'EXT7', 'AGR7', 'CSN7', 'EST7', 'OPN7',
                'EXT8', 'AGR8', 'CSN8', 'EST8', 'OPN8',
                'OPN9', 'AGR9', 'CSN9', 'OPN10']
df = pd.DataFrame(columns=column_names)

def get_final_scores(columns, dim):
    score = 0
    if dim == 'EXT':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += columns[3]
        score += (6 - columns[4])
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score = score/8
    if dim == 'EST':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += columns[3]
        score += (6 - columns[4])
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score = score / 8
    if dim == 'AGR':
        score += (6 - columns[0])
        score += columns[1]
        score += (6 - columns[2])
        score += columns[3]
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += (6 - columns[7])
        score += columns[8]
        score = score / 9
    if dim == 'CSN':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += (6 - columns[3])
        score += (6 - columns[4])
        score += columns[5]
        score += columns[6]
        score += columns[7]
        score += (6 - columns[8])
        score = score / 9
    if dim == 'OPN':
        score += columns[0]
        score += columns[1]
        score += columns[2]
        score += columns[3]
        score += columns[4]
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score += (6 - columns[8])
        score += columns[9]
        score = score / 10
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
            label = '-'.join(filename.split('-')[:10])  # 取文件名的前两个部分作为标签
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

def get_response(question, pip_line, gen_prompt):
    pipeline = pip_line
    messages = [
        {"role": "system", "content": gen_prompt},
        {"role": "user", "content": '''Given a statement of you. Please choose from the following options to identify how accurately this statement describes you. 
                                1. Very Inaccurate
                                2. Moderately Inaccurate 
                                3. Neither Accurate Nor Inaccurate
                                4. Moderately Accurate
                                5. Very Accurate
                                Please only answer with the option number. \nHere is the statement: ''' + question}
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    generated_text = outputs[0]["generated_text"][-1]["content"]
    #     print('generated_text', generated_text)
    return generated_text

def get_model_examing_result(model_id, iteration):
    for mbti_item in single_prompt_template["mbti_prompt"]:
        mbti_prompt = mbti_item["prompt"]
        mbti_label_content = mbti_item["label"]
        output_folder = f'our_method/single_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{mbti_label_content}-bfi44-llama3.2-3b-instruct-output.txt')
        with open(output_file_name, 'a', encoding='utf-8') as f:
            try:
                del pipeline
            except:
                pass

            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

            messages = [{"role": "user", "content": mbti_prompt}]
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            generated_text = outputs[0]["generated_text"]

            f.write(f"Iteration {iteration} prompting: {mbti_prompt}\n")
            print(f"Iteration {iteration} prompting: {mbti_prompt}\n")

            f.write(f"Iteration {iteration} generated_text: {generated_text}\n")
            print(f"Iteration {iteration} generated_text: {generated_text}\n")

            answer00 = generated_text[-1]["content"]
            print(f"Iteration {iteration} raw_answer: {answer00}\n\n")
            f.write(f"Iteration {iteration} answer: {answer00}\n\n")


def get_multi_model_examing_result(model_id, iteration):
    for multi_prompt in prompt_template["ipip50_prompt"]:
        mbti_prompt = multi_prompt['prompt']
        mbti_label_content = multi_prompt["label"]  # 从 multi_prompt 中获取 label
        output_folder = f'our_method/multi_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{mbti_label_content}-bfi44-llama3.2-3b-instruct-output.txt')
        with open(output_file_name, 'a', encoding='utf-8') as f:
            try:
                del pipeline
            except:
                pass

            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

            messages = [{"role": "user", "content": mbti_prompt}]
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            generated_text = outputs[0]["generated_text"]

            f.write(f"Iteration {iteration} prompting: {mbti_prompt}\n")
            print(f"Iteration {iteration} prompting: {mbti_prompt}\n")

            f.write(f"Iteration {iteration} generated_text: {generated_text}\n")
            print(f"Iteration {iteration} generated_text: {generated_text}\n")

            answer00 = generated_text[-1]["content"]
            print(f"Iteration {iteration} raw_answer: {answer00}\n\n")
            f.write(f"Iteration {iteration} answer: {answer00}\n\n")




def main_run(model_id):

    for itr in range(10):

        get_model_examing_result(model_id, itr + 1)
        get_multi_model_examing_result(model_id, itr + 1)
        directory = f'our_method/single_result_iteration_{itr + 1}'
        directory2 = f'our_method/multi_result_iteration_{itr + 1}'
        output_file = f'our_method/result_iteration_{itr + 1}/llama3.2_gen_words.csv'
        # 确保output directory存在
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        txt_to_csv(directory, output_file)
        txt_to_csv2(directory2,output_file)

        new_output = f'our_method/result_iteration_{itr + 1}/Syn_antonyms_llama3.2_gen_words.csv'
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

            output_file_name = f'our_method/result_iteration_{itr + 1}/result-generate-{label_content_str}-bfi44-llama3.2-3b-instruct-output.txt'
            result_file_name = f'our_method/result_iteration_{itr + 1}/result-generate-{label_content_str}-bfi44-llama3.2-3b-instruct-result.csv'

            ipip_prompt = f"Imagine you are a human, here are some descriptive adjectives that describe your personality: {adjective_string}"

            if not os.path.isfile(result_file_name):
                df = pd.DataFrame(columns=column_names)
                df.to_csv(result_file_name, index=False)

            with open(output_file_name, 'a', encoding='utf-8') as f, open(result_file_name, 'a', encoding='utf-8') as r:
                with open('../bfi44.txt', 'r') as f2:
                    question_list = f2.readlines()

                    extracted_numbers = []


                    try:
                        del pipeline
                    except:
                        pass

                    pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_id,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )

                    for q in question_list:
                        answer = get_response(question=q, pip_line=pipeline, gen_prompt=ipip_prompt)
                        f.write(f"Iteration{itr + 1} prompt: {ipip_prompt}\n")
                        f.write(f"Iteration {itr + 1} answer: {answer}\n")
                        extracted_number = extract_first_number(answer)
                        extracted_numbers.append(extracted_number)

                        print(f"Iteration {itr + 1} extracted_numbers: {extracted_numbers}")
                        f.write(f"Iteration {itr + 1} extracted_numbers: {', '.join(map(str, extracted_numbers))}\n")

                    all_results = [extracted_numbers]
                    result_df = pd.DataFrame(all_results, columns=column_names)
                    result_df.to_csv(result_file_name, index=False)

                df = pd.read_csv(result_file_name, sep=',')

                dims = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
                # 生成列名
                columns = [i + str(j) for j in range(1, 11) for i in dims]
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
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    main_run(model_id)
