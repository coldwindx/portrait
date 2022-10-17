import os
import sys
import jieba
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append(os.getcwd())
from src import settings
# step-1 分词
counts = CountVectorizer()          
with open(settings.HLM_TXT_PATH, 'r', encoding='utf-8') as f:
    words = jieba.lcut(f.read())        # 654136
    words = [word for word in words if 1 < len(word)]
    bag = counts.fit_transform(words)   # 43054
# print(len(counts.vocabulary_.keys()))
# step-2 向量化
