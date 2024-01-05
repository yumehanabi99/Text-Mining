import numpy as np
from gensim.models import KeyedVectors

word2=open('词频统计jp.txt','r',encoding='utf-8').read()[2:].split(',')
word=[]
for i in word2:
    i=i[:i.find(':')]
    word.append(i)

visualizeVecs=[]
visualizeWords=[]
word_list=list(set(word))

model100 = KeyedVectors.load("word2vec_model_jp", mmap='r')
# 因为有低频词过滤所以加了try except
for i in word_list:
    try:
        visualizeVecs.append(model100.wv[i])  # model100为w2v模型
        visualizeWords.append(i)
    except KeyError:
        continue

visualizeVecs = np.array(visualizeVecs).astype(np.float64)  # 词向量列表
import matplotlib.pyplot as plt
from matplotlib.font_manager import *

temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / visualizeVecs.shape[0] * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])
myfont = FontProperties(fname='C://Windows//Fonts//msmincho.ttc')
for i in range(len(visualizeWords)):
    # print (i)
    # print (coord[i, 0])
    # print (coord[i, 1])
    color = 'red'
    plt.text(coord[i, 0], coord[i, 1], visualizeWords[i], bbox=dict(facecolor=color, alpha=0.03),
             fontsize=6, fontproperties=myfont)
plt.xlim((np.min(coord[:, 0]) - 0.5, np.max(coord[:, 0]) + 0.5))
plt.ylim((np.min(coord[:, 1]) - 0.5, np.max(coord[:, 1]) + 0.5))
#plt.savefig('w2v100_test.png', format='png', dpi=1000, bbox_inches='tight')
plt.show()