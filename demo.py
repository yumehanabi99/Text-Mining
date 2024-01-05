#encoding:utf-8
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'MS Mincho'

def dg():
    colors = ['red', 'green', 'blue', 'yellow'] #有向图
    colors = ['#F0FFFF', '#C0FFFF', '#90FFFF', '#30FFFF']
    DG = nx.DiGraph() #一次性添加多节点，输入的格式为列表
    #pos = nx.spring_layout(DG)
    DG.add_nodes_from(['A', 'B', 'C', 'D']) #添加边，数据格式为列表
    #数越大越近
    DG.add_edge('A','B',weight=90)
    DG.add_edge('A','C',weight=1)
    DG.add_edge('A','D',weight=1)
    lz=[400,300,200,100]
    #nx.draw(DG,with_labels=True, node_size=lz, node_color = range(4),cmap=plt.cm.Reds,linewidths=None)
    nx.draw(DG,with_labels=True, node_size=lz, node_color = colors,linewidths=None,edgecolors='black')
    plt.show()

large_size=100
DG = nx.DiGraph()

wordcount=open('词频统计cn.txt','r',encoding='utf-8').read()
#word=['中国式現代化','発展モデル','中国独自','独自','我が国','国情','欧米','国家安全','党員ら','一線','人類']
word=['本質的要求','現代化','各国','復興','平和的発展路線','社会主義現代化','中華民族','状況','意義','成果']

DG.add_nodes_from(word)
lz=[60*large_size]

data=open('词向量：中国式现代化（中）.txt','r',encoding='utf-8').read().replace('\n','').split('===')[0][1:-1]
for i in data.split("), ("):
    print(i)
    DG.add_edge('中国式現代化', i.split(', ')[0], weight=i.split(', ')[1])
    size_data = open('词频统计cn.txt', 'r', encoding='utf-8').read().split(','+ i.split(', ')[0]+':')[1].split(',')[0]
    lz.append(int(size_data)*large_size)
'''
data2=open('词向量：中国式现代化（中）.txt','r',encoding='utf-8').read().replace('\n','').split('===')
for i in range(len(word)):
    if i!=0:
        list=data2[i][1:-1].split("), (")
        for j in list:
            if j.split(', ')[0] in word:
                DG.add_edge(word[i], j.split(', ')[0], weight=j.split(', ')[1])
'''

#nx.draw_circular
nx.draw(DG,font_color='#1e2022',with_labels=True, node_size=lz,linewidths=None,edgecolors='#52616b',node_color='#c9d6df')
plt.show()
