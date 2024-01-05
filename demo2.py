# -*- coding: utf-8 -*-
import os

'''t="中国:969,中国共産党:702,20回党大会:531,発展:464,20回全国代表大会:397,党:386,習近平:324,堅持:315,推進:300,世界:295,報告:264,全面的:260,人民:253,代表:215,時代:211,実現:207,北京:193,特色:189,強化:186,偉大:177,書記:170,重要:167,中国式現代化:167,構築:166,党大会:153,建設:153,記者:150,成果:131,人民大会堂:128,現代化:127,整備:121,中華民族:120,大会:119,実施:116,社会主義:116,19期中央委員会:113,指導:109,記者会見:108,復興:108,習総書記:107,開幕:105"
l=t.split(',')
t2=''
t3=''
t4=''
count=0
for i in l:
    if int(i[i.find(':')+1:])>=100:
        t2=t2+i[i.find(':')+1:]+';'
        t3=t3+"'"+i[:i.find(':')]+"',"
        t4=t4+"['"+i[:i.find(':')]+"' newline '"+i[i.find(':')+1:]+"'],"+i[i.find(':')+1:]+";"
        count=count+1

print(t2.replace(";","\n"))
print(t3.replace("'","").replace(",","\n"))
print(t4)
print(count)'''

'''from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import *

text='習近平（シーチンピン）総書記（国家主席）が掲げた「中国式現代化」によって発展を推し進めることなどが新たに明記された。'
token_filters = [CompoundNounFilter(), POSKeepFilter(['名詞']), POSStopFilter(["名詞,代名詞", "名詞,非自立", "名詞,数"])]
#token_filters = [CompoundNounFilter(), POSKeepFilter(['名詞'])]
t = Tokenizer("userdic2.csv", udic_type="simpledic", udic_enc="utf8")

a = Analyzer(token_filters=token_filters, tokenizer=t)
#a = Analyzer()
l=[]
for i in a.analyze(text):
    print(i)
    l.append(str(i).split('\t')[0])

print(' / '.join(l))'''


#dic_list=open('D://日本国语大辞典//日本国语大辞典.txt', 'r', encoding='utf-8').read().split('\n')
'''
def find_in_dic_new(line_n):
    text = ''
    line_n = int(line_n)
    c = 0
    start_read = 0
    for i in dic_list:
        if c == line_n:
            start_read = 1
        if start_read == 1:
            text = text + i+'\n'
            print(i)
            if i.find('</>') != -1:
                break
        c = c + 1
    return text

def find_in_dicx(line_n):
    text=''
    line_n=int(line_n)
    c = 0
    start_read=0
    with open('D://日本国语大辞典//日本国语大辞典.txt', 'r', encoding='utf-8') as w:
        line = w.readline()
        while line:
            line = w.readline()
            if c==line_n:
                start_read=1
            if start_read==1:
                text=text+line
                print(line)
                if line.find('</>\n')!=-1:
                    break
            c = c + 1
    w.close()
    return text'''

'''no_find=open('C://Users//97633//Desktop//readmdict//朱子语类二字词词表.txt', 'r', encoding='utf-8').read()
print(no_find[:100])
from opencc import OpenCC

#a=OpenCC('jp2t').convert(no_find)
a=OpenCC('t2s').convert(no_find)
a=OpenCC('s2t').convert(a)
a=OpenCC('t2jp').convert(a)
print(a[:100])
open('C://Users//97633//Desktop//readmdict//朱子语类二字词词表jp2.txt', 'w', encoding='utf-8').write(a)
print('ok')'''


'''l=open('C://Users//97633//Desktop//readmdict//朱子语类二字词词表jp2.txt','r',encoding='utf-8').read().split()
word_list=open('C://Users//97633//Desktop//readmdict//word_list3.txt','r',encoding='utf-8').read().split()
word_line=open('C://Users//97633//Desktop//readmdict//word_line3.txt','r',encoding='utf-8').read().split()
ok=open('C://Users//97633//Desktop//readmdict//output_已遍历的(第3轮).txt', 'r', encoding='utf-8').read().split()

for i in l:
    if i not in ok:
        temp_word_list=[]
        temp_word_line = []
        for j in range(len(word_list)):
            if word_list[j].find(i)!=-1 and len(word_list[j])==2:
                temp_word_list.append(i)
                temp_word_line.append(word_line[j])
        if len(temp_word_list)==1:
            print('find 1 word')
            print(temp_word_line[0])
            inputtxt=find_in_dic_new(temp_word_line[0])
            open('C://Users//97633//Desktop//readmdict//word//'+i+'.txt','w',encoding='utf-8').write(inputtxt)
            #open('C://Users//97633//Desktop//readmdict//output_仅有一例.txt','a',encoding='utf-8').write(i+'\t')
        if len(temp_word_list)>1:
            print('find more word')
            for k in range(len(temp_word_line)):
                inputtxt = find_in_dic_new(temp_word_line[k])
                open('C://Users//97633//Desktop//readmdict//word//' + i +'_'+str(k)+ '.txt', 'w', encoding='utf-8').write(inputtxt)
            #open('C://Users//97633//Desktop//readmdict//output_多例.txt', 'a', encoding='utf-8').write(i + '\t')
        if len(temp_word_list)==0:
            print('find no word')
            open('C://Users//97633//Desktop//readmdict//output_没找到(第3轮).txt', 'a', encoding='utf-8').write(i + '\t')
        open('C://Users//97633//Desktop//readmdict//output_已遍历的(第3轮).txt', 'a', encoding='utf-8').write(i + '\t')
'''

'''word_list=[]
word_line=[]
c=0
word_find=0
with open('D://日本国语大辞典//日本国语大辞典.txt', 'r', encoding='utf-8') as w:
    line=w.readline()
    while line:
        #print(c)
        #print(line)
        line = w.readline()
        if line.find('【')==-1 :
            word_find=0
        if word_find == 1 and line.find('【')!=-1 and line.find('・')!=-1:
            temp_wl=line[line.find('【')+1:line.find('】')].split('・')
            word_find = 0
            for i in temp_wl:
                if len(i)==2:
                    word_list.append(i)
                    word_line.append(str(c + 1))
            #new_line=line[line.find('【')+1:line.find('】')]
            #word_find = 0
            #word_list.append(new_line)
            #word_line.append(str(c+1))
        if line=='</>\n':
            word_find=1
        c = c + 1

print('finish1')'''

'''word_list=open('C://Users//97633//Desktop//readmdict//word_list3.txt','r',encoding='utf-8').read().split()
word_line=open('C://Users//97633//Desktop//readmdict//word_line3.txt','r',encoding='utf-8').read().split()

tn=open('C://Users//97633//Desktop//readmdict//output_没找到(第3轮).txt','r',encoding='utf-8').read()
from opencc import OpenCC

tn_t=OpenCC('jp2t').convert(tn)
tn_s=OpenCC('t2s').convert(tn_t)

find_in_list=0
l2=[]
for i in tn_t.split('\t'):
    if i in word_list:
        find_in_list=find_in_list+1
        l2.append(i)

print(find_in_list)
print(l2)
for i in tn_s.split('\t'):
    if i in word_list:
        find_in_list=find_in_list+1
        l2.append(i)
print(find_in_list)
print(l2)

print(len(tn.split()))
print(find_in_list)

print(len(set(l2)))
open('C://Users//97633//Desktop//readmdict//temp.txt','w',encoding='utf-8').write('\t'.join(l2))'''


#l=open('C://Users//97633//Desktop//readmdict//朱子语类二字词词表3.txt','r',encoding='utf-8').read().split('\t')

'''input_text=''
print('start write')
all_line=len(word_list)
for i in range(len(word_list)):
    text=word_list[i]+'\t'+str(word_line[i])
    if text.find('\n')!=-1:
        text=text.replace('\n','')
    input_text=input_text+text+'\n'
    print(text)
    print(str(i)+'/'+str(all_line))'''

#open('C://Users//97633//Desktop//readmdict//dic.txt','w',encoding='utf-8').write(input_text)
'''import re
regex = re.compile(r'<[^>]+>')

for i in os.listdir('C://Users//97633//Desktop//readmdict//word//'):
    t=open('C://Users//97633//Desktop//readmdict//word//'+i,'r',encoding='utf-8').read()
    open('C://Users//97633//Desktop//readmdict//word2//'+i,'w',encoding='utf-8').write(regex.sub('', t))'''
'''
txt=''
for i in os.listdir('C://Users//97633//Desktop//readmdict//word2//'):
    t=open('C://Users//97633//Desktop//readmdict//word2//'+i,'r',encoding='utf-8').read().split('\n')
    t2=[]
    for j in t:
        if j!='':
            t2.append(j)
    txt=txt+'\n\n'.join(t2)+'\n\n==========\n\n'
    print('y')

open('C://Users//97633//Desktop//readmdict//word2_all.txt','w',encoding='utf-8').write(txt)'''

'''l=open('C://Users//97633//Desktop//朱子语类//没找到的.txt','r',encoding='utf-8').read().split('\t')
l2=open('C://Users//97633//Desktop//朱子语类//朱子语类二字词词表（日语汉字）.txt','r',encoding='utf-8').read().split('\t')
l3=[]
for i in l2:
    if i not in l:
        l3.append(i)
open('C://Users//97633//Desktop//朱子语类//找到的.txt','w',encoding='utf-8').write('\t'.join(l3))'''

'''l=open('C://Users//97633//Desktop//朱子语类//找到的.txt','r',encoding='utf-8').read().split()
l2=open('C://Users//97633//Desktop//朱子语类//word_list3.txt','r',encoding='utf-8').read().split()
l3=open('C://Users//97633//Desktop//朱子语类//word_line3.txt','r',encoding='utf-8').read().split()
t=''
for i in l:
    a=l3[l2.index(i)]
    if l3.count(a)>1:
        temp=[]
        for j in range(len(l3)):
            if l3[j]==a:
                temp.append(j)
        text=''
        for j in temp:
            if l2[j].find('・')!=-1:
                text=text+l2[j]
        print(i+'\t'+a+'\t'+text)
        t=t+i+'\t'+a+'\t'+text+'\n'
open('C://Users//97633//Desktop//朱子语类//temp.txt','w',encoding='utf-8').write(t)'''

'''t=''
for i in os.listdir('C://Users//97633//Desktop//朱子语类//单个文件（带html标签）//'):
    word=i[:2]
    l=open('C://Users//97633//Desktop//朱子语类//单个文件（带html标签）//'+i,'r',encoding='utf-8').read().split('\n')
    for j in l:
        if j.find('<partspeech>')!=-1:
            cx=j[j.find('<partspeech>'):j.find('</partspeech>')].replace('<partspeech>','')
    t=t+word+'\t'+cx.replace('〔','').replace('〕','')+'\n'
    print(word,cx.replace('〔','').replace('〕',''))

open('C://Users//97633//Desktop//朱子语类//temp.txt','w',encoding='utf-8').write(t)'''

'''l1=[]
l2=[]
all=[]
for i in os.listdir('news//jp_all//'):
    l=open('news//jp_all//'+i,'r',encoding='utf-8').read().split('\n')
    for j in l:
        if j.find('共同富裕')!=-1:
            for k in j.split('。'):
                if k.find('共同富裕') != -1:
                    all.append(k)

                if k.find('共同富裕')!=-1 and k.find('格差')!=-1:
                    l1.append(k)
                if k.find('共同富裕')!=-1 and k.find('豊か')!=-1:
                    l2.append(k)

print(len(all))
print(len(l1))
print(len(l2))
for i in all:
    if i not in l1 and i not in l2:
        print(i)'''

'''l=[]
x=''
t=open('C://Users//97633//Desktop//中国式現代化.txt','r',encoding='utf-8').read().split('\n')
for i in t:
    if i.find('.txt')!=-1:
        print(i[:4]+'/'+i[4:6]+'/'+i[6:8])
        x=x+i[:4]+'/'+i[4:6]+'/'+i[6:8]+'\t'
    if i.find('label')!=-1:
        print(i[i.find('label')+9:i.find('score')-4])
        print(i[i.find('score')+8:-2])
        x=x+i[i.find('label')+9:i.find('score')-4]+'\t'
        x=x+i[i.find('score')+8:-2]+'\n'
    if i.find('.txt')==-1 and i.find('label')==-1:
        print(i)
        x=x+i+'\t'
print(x)'''

'''for i in os.listdir('news//jp_all//'):
    l=open('news//jp_all//'+i,'r',encoding='utf-8').read().split('\n')
    for j in l:
        if j.find('共同富裕')!=-1:
            for k in j.split('。'):
                if k.find('共同富裕') != -1:
                    print(i)
                    print(k)
                    print(kjbs__([k]))'''


'''l=['日経', '朝日', '毎日', '産経', '読売']
for a in l:
    print(a)
    c=0
    for i in os.listdir('news//jp//'+a+'//'):
        t=open('news//jp//'+a+'//'+i,'r',encoding='utf-8').read()
        if t.find('中国式の現代化')!=-1 or t.find('中国式現代化')!=-1:
            c=c+t.count('中国式現代化')
    print(c)'''

'''for i in os.listdir('news//jp//日経'):
    print(i)'''

for i in range(1,101):
    open('C://Users//97633//Desktop//录音//录音'+str(i)+'.aac','w',encoding='utf-8').write('test')