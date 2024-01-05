# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# NLP
import glob
from tqdm import tqdm
import math
import urllib
import gensim

# pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models
#pyLDAvis.enable_notebook()

# Vis
from wordcloud import WordCloud
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
font = {'family': 'TakaoGothic'}
matplotlib.rc('font', **font)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
import logging
import pickle
from smart_open import open


# Any results you write to the current directory are saved as output.

def main():

    l=[]
    for i in open('mecab_save_jp.txt', 'r', encoding='utf-8').read().split('\n'):
        l.append(i.split(' '))
    dictionary = gensim.corpora.Dictionary(l)
    #dictionary.filter_extremes(no_below=2)

    corpus = [dictionary.doc2bow(text) for text in l]
    # tfidf
    tfidf = gensim.models.TfidfModel(corpus)
    # make corpus_tfidf
    corpus_tfidf = tfidf[corpus]
    new_corpus=[]
    for i in list(corpus_tfidf):
        temp_l=[]
        for j in i:
            temp_l.append((j[0],j[1]*10))
        new_corpus.append(temp_l)
    print(new_corpus[:10])

    #Metrics for Topic Models
    start = 2
    limit = 22#22
    step = 1

    coherence_vals = []
    perplexity_vals = []

    #(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)

    '''for n_topic in tqdm(range(start, limit, step)):
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topic,
            passes=20,
            chunksize=10000,
            random_state=0,
            update_every=0
        )
        perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus)))
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=l, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherence_model_lda.get_coherence())

    # evaluation
    x = range(start, limit, step)

    fig, ax1 = plt.subplots(figsize=(12,5))

    # coherence
    c1 = 'darkturquoise'
    ax1.plot(x, coherence_vals, 'o--', color=c1)
    ax1.set_xlabel('Num Topics')
    ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)

    # perplexity
    c2 = 'slategray'
    ax2 = ax1.twinx()
    ax2.plot(x, perplexity_vals, 'o-', color=c2)
    ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)

    # Vis
    ax1.set_xticks(x)
    fig.tight_layout()
    plt.show()'''

    # save as png
    #plt.savefig('metrics.png')
    lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=8,
            passes=20,
            chunksize=10000,
            random_state=0,
            update_every=0,
            minimum_probability=0.001
        )

    # reduce memory
    del l
    gc.collect()
    # test
    N = sum(count for doc in corpus for id, count in doc)#test_corpus
    print("N: ",N)
    perplexity = np.exp2(-lda_model.log_perplexity(corpus))#test_corpus
    print("perplexity:", perplexity)

    # Vis t-SNE
    vis_tsne = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, mds='tsne', sort_topics=True)
    vis_tsne

    print(lda_model.print_topics())

    # save as html
    pyLDAvis.save_html(vis_tsne, 'test.html')
    #open('test.html', 'w', encoding='utf-8').write(open('test.html', 'r', encoding='utf-8').read().replace('cdn','fastly'))

if __name__ == '__main__':
    main()
