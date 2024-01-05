#coding=utf-8
from collections import Counter

import jieba
import paddle
from nltk.parse import stanford
import jieba.posseg as pseg

#standFord句法分析
'''string = '高举中国特色社会主义伟大旗帜。'
seg_list = jieba.cut(string, cut_all=False, HMM=True)
seg_str = ' '.join(seg_list)

parser_path = './stanford-parser.jar'
model_path =  './stanford-parser-4.2.0-models.jar'
pcfg_path = 'edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'

parser = stanford.StanfordParser(path_to_jar=parser_path,path_to_models_jar=model_path,model_path=pcfg_path)

sentence = parser.raw_parse(seg_str)

s=str(next(sentence))
print(s)
print(s.split('\n'))

for line in sentence:
    print(line)
    #line.leaves()
    line.draw()'''

#jieba句法分析
'''paddle.enable_static()
l=open('二十大报告全文（中文）.txt','r',encoding='utf-8').read().replace('　','').replace('—','').replace('。','\n').replace('！','\n').replace('？','\n').replace('；','\n').replace('，','\n').split('\n')

n=['PER','LOC','ORG','TIME','f','s','t','nr','ns','nt','nw','nz','m','q','r']
v=['vd','vn']
a=['ad','an']

total=[]
totalt=[]
text=[]

for i in l:
    f=[]
    temp=[]
    print(i)
    jieba.enable_paddle()
    words = pseg.cut(i, use_paddle=True)
    for word, flag in words:
        #print('%s %s' % (word, flag))
        f.append(flag)
    print(f)
    f=['n' if i in n else i for i in f]
    f = ['v' if i in v else i for i in f]
    f = ['a' if i in a else i for i in f]
    print(f)
    for j in f:
        if temp==[]:
            temp.append(j)
        if temp[-1]!=j:
            temp.append(j)
    print(temp)
    text.append(i)
    total.append(temp)
    totalt.append(str(temp))

#print(Counter(total).most_common(10))
#[("['v', 'n']", 166), ('[]', 147), ("['v', 'n', 'v']", 115), ("['v', 'n', 'v', 'n']", 94), ("['n', 'v', 'n']", 39), ("['v', 'a', 'n']", 32), ("['v', 'n', 'v', 'n', 'v']", 31), ("['a', 'v', 'n']", 27), ("['v', 'n', 'a', 'n']", 25), ("['n', 'v']", 22)]
for i in range(len(text)):
    if total[i]==['v', 'n', 'v', 'n']:
        print(text[i])'''

#四字格
'''l=open('二十大报告全文（中文）.txt','r',encoding='utf-8').read().replace('　','').replace('—','').replace('。','\n').replace('！','\n').replace('？','\n').replace('；','\n').replace('，','\n').replace('、','\n').split('\n')
l2=[]
for i in l:
    if len(i)==4:
        print(i)
        l2.append(i)

print(Counter(l2).most_common(20))'''

