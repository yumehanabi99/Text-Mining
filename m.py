#encoding:utf-8
import collections
import os
from collections import Counter

from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import *

import pyLDAvis
import pyLDAvis.gensim_models
import random
import gensim
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from janome.charfilter import *

import pandas as pd
import pathlib
import jaconv

from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer

import wordcloud

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import nlplot

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

import MeCab

#读取保存的janome分词结果
def read_janome_save(file_name,return_value):
    df = pd.DataFrame()
    list=[]
    if return_value=='df':
        if file_name == 'jp':
            df = pd.DataFrame(open('janome_save_jp.txt', 'r', encoding='utf-8').read().split('\n'), columns=['text'])
        if file_name == 'cn':
            df = pd.DataFrame(open('janome_save_cn.txt', 'r', encoding='utf-8').read().split('\n'), columns=['text'])
        return df
    if return_value == 'list':
        if file_name == 'jp':
            for i in open('janome_save_jp.txt', 'r', encoding='utf-8').read().split('\n'):
                list.append(i.split(' '))
        if file_name == 'cn':
            for i in open('janome_save_cn.txt', 'r', encoding='utf-8').read().split('\n'):
                list.append(i.split(' '))
        return list

#读取保存的mecab分词结果
def read_mecab_save(file_name,return_value):
    df = pd.DataFrame()
    list=[]
    if return_value=='df':
        if file_name == 'jp':
            df = pd.DataFrame(open('mecab_save_jp.txt', 'r', encoding='utf-8').read().split('\n'), columns=['text'])
        if file_name == 'cn':
            df = pd.DataFrame(open('mecab_save_cn.txt', 'r', encoding='utf-8').read().split('\n'), columns=['text'])
        return df
    if return_value == 'list':
        if file_name == 'jp':
            for i in open('mecab_save_jp.txt', 'r', encoding='utf-8').read().split('\n'):
                list.append(i.split(' '))
        if file_name == 'cn':
            for i in open('mecab_save_cn.txt', 'r', encoding='utf-8').read().split('\n'):
                list.append(i.split(' '))
        return list

#词频统计，l二维数组
def wordCounter(l):
    wc = []
    for i in l:
        for j in i:
            wc.append(j)
    return Counter(wc)

#l二维数组，word字符串（'習近平'），ret二维数组
def word2vec_(l,word):
    model = word2vec.Word2Vec(l, vector_size=100, min_count=5, window=5, epochs=100)
    model.save('word2vec_model')
    #ret = model.wv.most_similar(positive=[word])
    #return ret
'''    for item in ret:
        print(item[0], item[1])'''

def doc2vec_(l,save_file):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(l)]
    print(documents)
    #model_dm = Doc2Vec(documents, min_count=1, window=3, vector_size=100, sample=1e-3, negative=5, workers=4)
    #model_dm.save(save_file)

    #model_dm = Doc2Vec.load(save_file)
    #print(model_dm.docvecs.most_similar(1))

#l二维数组
def lda(l):
    dictionary = gensim.corpora.Dictionary(l)
    corpus = [dictionary.doc2bow(text) for text in l]
    n_cluster = 9
    lda = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_cluster,
        minimum_probability=0.001,
        passes=20,
        update_every=0,
        chunksize=10000,
        random_state=1,
    )
    print(lda.print_topics())
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(vis, 'pyldavis_output_'+str(random.randint(1,999999))+'.html')

    # test
    N = sum(count for doc in corpus for id, count in doc)
    print("N: ", N)
    perplexity = np.exp2(-lda.log_perplexity(corpus))
    print("perplexity:", perplexity)

#l二维数组
def lda2(l):
    dictionary = gensim.corpora.Dictionary(l)
    corpus = [dictionary.doc2bow(text) for text in l]
    n_cluster = 7
    '''lda = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_cluster,
        minimum_probability=0.001,
        passes=20,
        update_every=0,
        chunksize=10000,
        random_state=1,
    ).save('doc2bow_jp')'''
    lda_model = gensim.models.LdaModel.load('doc2bow_jp')
    data = []
    for i in open('janome_save_jp_data.txt', 'r', encoding='utf-8').read().split('\n'):
        data.append(i)
    for i, doc in enumerate(corpus):
        topic = sorted(lda_model[doc], key=lambda x: x[1], reverse=True)
        # topic = sorted(lda_model[doc], key=lambda x: x[1], reverse=True)[0][0]
        # print('文本编号：{}，主题编号：{}'.format(i, topic))
        # print(data[i],topic)
        for j in topic:
            if j[0] == 5:
                print(data[i],j[1])


#情感分析，输入文章一维数组，返回二维数组（[[标题1,值1],[标题2,值2],...]）
def kjbs__(l):
    rl=[]
    for i in l:
        model = AutoModelForSequenceClassification.from_pretrained('C://Users//97633//Desktop//NICT_BERT-base_JapaneseWikipedia_100K//')#C://Users//97633//Desktop//NICT_BERT-base_JapaneseWikipedia_100K//
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, max_length=512, truncation=True)
        v=nlp(i)
        rl.append([i.split('\n')[0],v])
        print(v)
    return rl

#读文件，返回一维数组（['标题1\n内容1\n内容2','标题2\n内容1'...]）
#c国家（'cn'；'jp'），m媒体（['日経', '朝日', '毎日', '産経', '読売']；['CRI', '中国網', '人民中国', '人民網', '新華社']）
def readFile(c,m):
    l=[]
    s=''
    files=[]

    if c=='cn':
        for i in m:
            tl=os.listdir("news//cn//"+i+'//')
            for j in tl:
                files.append("news//cn//"+i+'//'+j)
    if c=='jp':
        for i in m:
            tl=os.listdir("news//jp//"+i+'//')
            for j in tl:
                files.append("news//jp//"+i+'//'+j)

    for i in files:
        l.append(open(i,'r',encoding='utf-8').read())
    return l

#分词，l一维数组（['标题1\n内容1\n内容2','标题2\n内容1'...]），返回二维数组
def janomeT(l):
    rl=[]

    #停用词
    stopw = open("JapaneseStopWord.txt", 'r', encoding='utf-8').read().split('\n')
    stopw.append('の')
    stopw.append('さ')
    stopw.extend(['1日', '2日', '3日', '4日', '5日', '6日', '7日', '8日', '9日', '10日', '11日', '12日', '13日', '14日', '15日', '16日', '17日', '18日', '19日', '20日', '21日', '22日', '23日', '24日', '25日', '26日', '27日', '28日', '29日', '30日', '31日'])
    stopw.extend(['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'])

    token_filters = [CompoundNounFilter(), POSKeepFilter(['名詞']), POSStopFilter(["名詞,代名詞", "名詞,非自立", "名詞,数"])]
    #token_filters = [CompoundNounFilter(), POSKeepFilter(['名詞'])]
    t = Tokenizer("userdic3.csv", udic_type="simpledic", udic_enc="utf8")
    char_filters = [UnicodeNormalizeCharFilter(),RegexReplaceCharFilter('習氏', '習近平'),RegexReplaceCharFilter('習近平氏', '習近平'),RegexReplaceCharFilter('中国式の現代化', '中国式現代化'),RegexReplaceCharFilter('2つの確立', '二つの確立'),RegexReplaceCharFilter('2つの擁護', '二つの擁護')]#,RegexReplaceCharFilter('100年の奮闘目標', '百年の奮闘目標'),RegexReplaceCharFilter('100年奮闘目標', '百年奮闘目標')
    a = Analyzer(token_filters=token_filters, tokenizer=t, char_filters=char_filters)

    for i in l:
        t=[]
        for token in a.analyze(i):
            #输出结果：書記	名詞,一般,*,*,*,*,書記,ショキ,ショキ
            #print(token)
            if str(token).split('\t')[1].split(',')[6] not in stopw:
                t.append(str(token).split('\t')[1].split(',')[6])
        rl.append(t)
    return rl

#mecab 分词
# -Owakati
def mecabT(l):
    rl=[]
    # 全角の文字列
    FULLWIDTH_DIGITS = "０１２３４５６７８９"
    FULLWIDTH_ALPHABET = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    FULLWIDTH_PUNCTUATION = "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～　"
    FULLWIDTH_ALPHANUMERIC = FULLWIDTH_DIGITS + FULLWIDTH_ALPHABET  # 英数字
    FULLWIDTH_ALL = FULLWIDTH_ALPHANUMERIC + FULLWIDTH_PUNCTUATION  # 英数字、記号

    # 半角の文字列
    HALFWIDTH_DIGITS = "0123456789"
    HALFWIDTH_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    HALFWIDTH_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ "
    HALFWIDTH_ALPHANUMERIC = HALFWIDTH_DIGITS + HALFWIDTH_ALPHABET  # 英数字
    HALFWIDTH_ALL = HALFWIDTH_ALPHANUMERIC + HALFWIDTH_PUNCTUATION  # 英数字、記号

    #停用词
    stopw = open("JapaneseStopWord.txt", 'r', encoding='utf-8').read().split('\n')
    stopw.append('の')
    stopw.append('さ')
    stopw.append('ら')
    stopw.append('氏')
    stopw.append('　')
    #stopw.append('nbsp')

    for i in l:

        # 数字、アルファベットを半角に変換する。
        conv_map = str.maketrans(FULLWIDTH_ALPHANUMERIC, HALFWIDTH_ALPHANUMERIC)

        i = i.translate(conv_map)

        # 数字の削除
        #num_regex = re.compile('\d+,?\d*')
        # 数字は全て0に置換する
        #i = num_regex.sub('　', i)
        #print(i)

        # 記号の削除
        code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％～①②③④⑤⑥⑦⑧⑨⑩]')
        i = code_regex.sub('　', i)
        #print(i)

        #mecab-dict-index -d "C:\Program Files\MeCab\dic\ipadic" -u foo.dic -f utf-8 -t utf-8 C:\Users\97633\Desktop\2.csv
        wakati = MeCab.Tagger("-r nul -d 'C:/Program Files/MeCab/dic/ipadic' -u 'C:/Program Files/MeCab/dic/ipadic/NEologd.dic,C:/Program Files/MeCab/dic/ipadic/foo.dic'")
        list=wakati.parse(i.replace('\n','　')).split('\n')[:-2]

        word = ''
        newlist=[]
        for k in range(len(list)):
            stop=1
            cx=list[k].split('\t')[1].split(',')[0]
            w=list[k].split('\t')[0]
            if cx=='名詞' and w not in stopw and re.findall(r"\b\d+\b", w)==[] and re.findall("\d+日|\d+年|\d+月|\d+人|\d+歳|\d+時|\d+分", w)==[]:
                word=word+w
            else:
                stop=0
            if stop==0 and word!='':
                if word=='習':
                    word='習近平'
                if word in ['2つの確立','2つの擁護']:
                    word=word.replace('2つの確立','二つの確立').replace('2つの擁護','二つの擁護')
                newlist.append(word)
                word=''
        rl.append(newlist)
    return rl

def mecabT_(l):
    rl=[]
    # 全角の文字列
    FULLWIDTH_DIGITS = "０１２３４５６７８９"
    FULLWIDTH_ALPHABET = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    FULLWIDTH_PUNCTUATION = "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～　"
    FULLWIDTH_ALPHANUMERIC = FULLWIDTH_DIGITS + FULLWIDTH_ALPHABET  # 英数字
    FULLWIDTH_ALL = FULLWIDTH_ALPHANUMERIC + FULLWIDTH_PUNCTUATION  # 英数字、記号

    # 半角の文字列
    HALFWIDTH_DIGITS = "0123456789"
    HALFWIDTH_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    HALFWIDTH_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ "
    HALFWIDTH_ALPHANUMERIC = HALFWIDTH_DIGITS + HALFWIDTH_ALPHABET  # 英数字
    HALFWIDTH_ALL = HALFWIDTH_ALPHANUMERIC + HALFWIDTH_PUNCTUATION  # 英数字、記号

    #停用词
    stopw = open("JapaneseStopWord.txt", 'r', encoding='utf-8').read().split('\n')
    stopw.append('の')
    stopw.append('さ')
    stopw.append('ら')
    stopw.append('氏')
    stopw.append('　')
    #stopw.append('nbsp')
    stopw.append('する')
    stopw.append('れる')
    stopw.append('られる')
    stopw.append('せる')
    stopw.append('させる')
    stopw.append('なる')
    stopw.append('いる')
    stopw.append('ある')
    stopw.append('ない')
    stopw.append('くる')
    stopw.append('いく')
    stopw.append('みる')
    stopw.append('できる')
    stopw.append('おる')

    for i in l:

        # 数字、アルファベットを半角に変換する。
        conv_map = str.maketrans(FULLWIDTH_ALPHANUMERIC, HALFWIDTH_ALPHANUMERIC)

        i = i.translate(conv_map)

        # 数字の削除
        #num_regex = re.compile('\d+,?\d*')
        # 数字は全て0に置換する
        #i = num_regex.sub('　', i)
        #print(i)

        # 記号の削除
        code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％～①②③④⑤⑥⑦⑧⑨⑩]')
        i = code_regex.sub('　', i)
        #print(i)

        #mecab-dict-index -d "C:\Program Files\MeCab\dic\ipadic" -u foo.dic -f utf-8 -t utf-8 C:\Users\97633\Desktop\2.csv
        wakati = MeCab.Tagger("-r nul -d 'C:/Program Files/MeCab/dic/ipadic' -u 'C:/Program Files/MeCab/dic/ipadic/NEologd.dic,C:/Program Files/MeCab/dic/ipadic/foo.dic'")
        list=wakati.parse(i.replace('\n','　')).split('\n')[:-2]

        word = ''
        newlist=[]
        for k in range(len(list)):
            stop=1
            cx=list[k].split('\t')[1].split(',')[0]
            w=list[k].split('\t')[0]
            if cx=='名詞' and w not in stopw and re.findall(r"\b\d+\b", w)==[] and re.findall("\d+日|\d+年|\d+月|\d+人|\d+歳|\d+時|\d+分", w)==[]:
                word=word+w
            else:
                stop=0
            if stop==0 and word!='':
                if word=='習':
                    word='習近平'
                if word in ['2つの確立','2つの擁護','100年の奮闘目標','100年奮闘目標']:
                    word=word.replace('2つの確立','二つの確立').replace('2つの擁護','二つの擁護').replace('100年の奮闘目標','百年の奮闘目標').replace('100年奮闘目標','百年奮闘目標')
                newlist.append(word)
                word=''
            if cx=='動詞' or cx=='形容詞' or cx=='副詞':
                w2=list[k].split('\t')[1].split(',')[6]
                if w2 not in stopw:
                    newlist.append(w2)
        rl.append(newlist)
    return rl

def text_clustering(l):
    # 1、加载语料
    corpus = []
    for i in l:
        corpus.append(' '.join(i))

    # 2、计算tf-idf设为权重
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # 3、获取词袋模型中的所有词语特征，如果特征数量非常多的情况下可以按照权重降维
    word = vectorizer.get_feature_names()
    print("word feature length: {}".format(len(word)))

    # 4、导出权重，到这边就实现了将文字向量化的过程，矩阵中的每一行就是一个文档的向量表示
    tfidf_weight = tfidf.toarray()

    # 5、对向量进行聚类
    # 指定分成7个类
    # 可以利用肘部原则确定最佳聚类个数
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(tfidf_weight)

    # 打印出各个族的中心点
    print(kmeans.cluster_centers_)
    for index, label in enumerate(kmeans.labels_, 1):
        print("index: {}, label: {}".format(index, label))

    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
    # k-means的超参数n_clusters可以通过该值来评估
    print("inertia: {}".format(kmeans.inertia_))

    # 6、可视化
    # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(tfidf_weight)

    x = []
    y = []

    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=kmeans.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    # plt.savefig('./sample.png', aspect=1)

#共现网络 co-occurrence networks
def co_occurrence_networks(df):
    npt = nlplot.NLPlot(df, target_col='text')
    # top_nで頻出上位単語, min_freqで頻出下位単語を指定できる
    # ストップワーズは設定しませんでした。。。
    stopwords = npt.get_stopword(top_n=0, min_freq=0)
    # ビルド（データ件数によっては処理に時間を要します）※ノードの数のみ変更(min_edge_frequency=100)
    npt.build_graph(stopwords=stopwords, min_edge_frequency=100)
    npt.co_network(title='Co-occurrence network', width=1024,height=768,save=True)

#旭日图
def npt_sunburst(df):
    npt = nlplot.NLPlot(df, target_col='text')
    stopwords = npt.get_stopword(top_n=0, min_freq=0)
    npt.build_graph(stopwords=stopwords, min_edge_frequency=100)
    npt.sunburst(
        title='All sentiment sunburst chart',
        colorscale=True,
        color_continuous_scale='Oryel',
        save=True
    )

#词频统计
#wordCounter(janomeT(readFile('cn',['CRI', '中国網', '人民中国', '人民網', '新華社'])))
#print(Counter(open('janome_save_cn.txt','r',encoding='utf-8').read().split()))

'''for i in janomeT(readFile('cn',['CRI', '中国網', '人民中国', '人民網', '新華社'])):
    t=' '.join(i)
    open('test.txt','a',encoding='utf-8').write(t+'\n')'''

#词云图
'''
text=''
for i in janomeT(readFile('cn',['CRI', '中国網', '人民中国', '人民網', '新華社'])):
    text=text+' '.join(i)+' '
wc = wordcloud.WordCloud(font_path="msmincho.ttc",width = 800,height = 600,background_color='white',max_words=100).generate(text)
#plt.imshow(wc, interpolation="bilinear")
#plt.axis("off")
#plt.show()
wc.to_file("2.png")
'''

#lda主题模型
#lda(janomeT(readFile('jp',['日経', '朝日', '毎日', '産経', '読売'])))
#lda(janomeT(readFile('cn',['CRI', '中国網', '人民中国', '人民網', '新華社'])))

#lda(read_janome_save('cn','list'))
lda2(read_mecab_save('jp','list'))

#词向量
#word2vec_(janomeT(readFile('cn', ['CRI', '中国網', '人民中国', '人民網', '新華社'])),'中国式現代化')
#doc2vec_(janomeT(readFile('jp',['日経', '朝日', '毎日', '産経', '読売'])),'doc2vec_model_jp')
#word2vec_(read_janome_save('cn','list'),'中国式現代化')

'''
for i in ['本質的要求','現代化','各国','復興','平和的発展路線','社会主義現代化','中華民族','状況','意義','成果']:
    a=word2vec_(janomeT(readFile('cn', ['CRI', '中国網', '人民中国', '人民網', '新華社'])), i)
    print(a)
    print('===')
'''

#情感分析
#ljp=readFile('jp',['日経', '朝日', '毎日', '産経', '読売'])
#lcn=readFile('cn',['CRI', '中国網', '人民中国', '人民網', '新華社'])
#print(kjbs__(ljp))

#LSA
'''l=janomeT(readFile('jp',['日経', '朝日', '毎日', '産経', '読売']))
detokenized_doc=[]
for i in l:
    s=''
    for j in i:
        s=s+j+' '
    detokenized_doc.append(s)

news_df = pd.DataFrame({'clean_doc':detokenized_doc})

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000,max_df=0.5,smooth_idf=True)
X = vectorizer.fit_transform(news_df['clean_doc'])
X.shape

from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors
svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
len(svd_model.components_)

terms = vectorizer.get_feature_names()
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
    print("Topic " + str(i) + ": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")

import umap.umap_ as umap
import matplotlib.pyplot as plt

X_topics = svd_model.fit_transform(X)
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)
plt.figure(figsize=(7, 5))
plt.scatter(embedding[:, 0], embedding[:, 1],c=None,s=10,edgecolor='none')
plt.show()'''

#文本聚类
#text_clustering(janomeT(readFile('jp',['日経', '朝日', '毎日', '産経', '読売'])))
#text_clustering(read_janome_save('jp','list'))

#保存分词结果
'''text=''
for i in mecabT(readFile('jp',['日経', '朝日', '毎日', '産経', '読売'])):
    s=' '.join(i)
    text=text+s+'\n'
open('test.txt','w',encoding='utf-8').write(text)'''

#共现网络
#co_occurrence_networks(read_janome_save('jp','df'))

#旭日图
#npt_sunburst(read_janome_save('jp','df'))

# LDA主题数确定
'''
word_list = read_janome_save('jp','list')
corpus = [
    " ".join(np.random.choice(word_list[topic], 100))
    for topic in range(len(word_list)) for i in range(100)
]

tf_vectorizer = CountVectorizer()
bow = tf_vectorizer.fit_transform(corpus)

for c_num in range(1, 9):
    lda = LatentDirichletAllocation(
        n_components=c_num,
    )
    lda.fit(bow)
    print(f"トピック数: {c_num}, Perplexity: {lda.perplexity(bow)}")
'''

'''corpus = []
for i in read_janome_save('jp','list'):
    corpus.append(' '.join(i))

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
tfidf_weight = tfidf.toarray()

distortions = []
for i in range(2,21):
    km = KMeans(n_clusters=i)
    km.fit(tfidf_weight)
    distortions.append(km.inertia_)
    print("inertia: {}".format(km.inertia_))

plt.plot(range(2,21),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()'''