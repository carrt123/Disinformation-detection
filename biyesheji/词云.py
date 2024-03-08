import wordcloud
import jieba
import matplotlib.pyplot as plt
import jieba.posseg as pseg
import pyLDAvis.gensim_models

'''插入之前的代码片段'''
import codecs
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary


def load_txt(file_path, stopword):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    fake_txt, real_txt = [], []
    for line in lines:
        sentence = line.strip().split('\t')[1]
        if int(line.strip().split('\t')[0]) == 1:
            fake_txt.append([word for word in jieba.cut(sentence.replace('‘',""), cut_all=True) if word not in stopword])
        else:
            real_txt.append([word for word in jieba.cut(sentence, cut_all=True) if word not in stopword])
    return fake_txt, real_txt


def wordcloud_fc(txt):
    w = wordcloud.WordCloud(width=1000, font_path="msyh.ttf", height=700)
    w.generate(txt)
    w.to_file("realcloud.png")


def lad_fc(train):
    dictionary = corpora.Dictionary(train)

    corpus = [dictionary.doc2bow(text) for text in train]

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, passes=100)
    # num_topics：主题数目
    # passes：训练伦次
    # num_words：每个主题下输出的term的数目

    for topic in lda.print_topics(num_words=20):
        termNumber = topic[0]
        print(topic[0], ':', sep='')
        listOfTerms = topic[1].split('+')
        for term in listOfTerms:
            listItems = term.split('*')
            print('  ', listItems[1], '(', listItems[0], ')', sep='')

    d = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
    pyLDAvis.show(d)


def stopword_fc(file_path1, file_path2):
    with open(file_path1, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    with open(file_path2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
    stopword = []
    for line in lines1:
        stopword.append(line.rstrip("\n"))
    for line in lines2:
        stopword.append(line.rstrip("\n"))
    return stopword

stopword = stopword_fc('baidu_stopwords.txt', 'cn_stopwords.txt')
f, r = load_txt('data.txt', stopword)
lad_fc(f)

