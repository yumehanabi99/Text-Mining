'''
t=open("queryFn.htm",'r',encoding='utf-8').read()

a=0
while True:
    a=t.find("http://japanese.china.org.cn/",a+1)
    print(t[a:t.find(".htm",a)+4])
    if a==-1:
        break

'''

'''
import requests

t=open("chn.txt",'r',encoding='utf-8').read()
for i in t.split('\n'):
    response = requests.get(url=i)
    response.encoding="utf-8"
    open("save//"+i[i.find("cn/")+3:].replace('/','_'), 'w', encoding='utf-8').write(response.text)
'''


import os
import re

files = os.listdir("save//")
for file in files:
    l = open("save//"+file, 'r', encoding='utf-8').read().split('\n')
    for i in l:
        if i.find("<!--enpcontent-->")!=-1:
            t=i[i.find("<!--enpcontent-->"):i.find("<!--/enpcontent-->")]
            title=i[i.find("<title>")+7:i.find("</title>")]
            open("save//" + file.replace('htm','txt'), 'w', encoding='utf-8').write(title+'\n'+re.sub("<[^>]*?>", "", t))
