# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:05:30 2016

@author: ldy
"""
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import multiprocessing
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
from gensim.corpora.dictionary import Dictionary
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.ensemble import IsolationForest
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 5
window_size = 7
batch_size = 32
n_epoch = 100
input_length = 100
cpu_count = multiprocessing.cpu_count()
import tflearn

def buildWordVector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

# 加载文件，导入数据,分词
def loadfile():
    neg=pd.read_excel('./data/neg.xls',header=None,index=None)
    pos=pd.read_excel('./data/pos.xls',header=None,index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    combined=np.concatenate((pos['words'], neg['words']))

    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))
    np.save('./isoForest_data/y.npy', y)

    return combined,y

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab,
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


# 计算词向量
def get_train_vecs(combined):
    n_dim = 300

    # Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=5)
    imdb_w2v.build_vocab(combined)
    # Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(combined,total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    imdb_w2v.save('./isoForest_data/w2v_model/w2v_model.pkl')


    combined_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in combined])
    np.save('./isoForest_data/combined_vecs.npy', combined_vecs)


def get_data():
    combined_word2vec_model = np.load('./isoForest_data/w2v_model/w2v_model.pkl')

    combined_vecs = np.load('./isoForest_data/combined_vecs.npy')
    y = np.load('./isoForest_data/y.npy')
    return combined_vecs,y


##训练svm模型
def svm_train(vecs, y):
    #x_train, x_test, y_train, y_test = train_test_split(vecs, y, test_size=0.1)
    #clf = SVC(kernel='rbf', verbose=True)
    #clf.fit(x_train, y_train)
    x_train = vecs
    clf = IsolationForest(n_estimators=100,n_jobs=-1, verbose=2,contamination=0.01)
    clf.fit(x_train)

    joblib.dump(clf, './isoForest_data/isoForest_model/isoForest_model.pkl')
    print('accurcy =')
    #print(clf.score(x_test, y_test))

##得到待预测单个句子的词向量
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('./isoForest_data/w2v_model/w2v_model.pkl')
    # imdb_w2v.train(words)
    vecs = buildWordVector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return vecs

def get_data2(index_dict,word_vectors,combined):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    return embedding_weights

def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    #words = tokenizer(string)
    model=Word2Vec.load('./isoForest_data/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model,words)
    return index_dict, word_vectors, combined

def wordToWordVec(x, embedding_weights):
    x_new = np.zeros((len(x),100,100))
    for i in range(0,len(x)):
        for j in range(0,100):
            x_new[i][j] = embedding_weights[x[i][j]]
    return x_new


####对单个句子进行情感判断
def svm_predict(string,clf):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    result = clf.predict(words_vecs)
  #  print(result)
    if int(result[0]) == 1:
        pass
        #print(string, 'normal')
    else:
        print(string, 'unnormal:')



if __name__ == '__main__':


    ##导入文件，处理保存为向量
    combined,y=loadfile()
    print (len(combined),len(y))

    # 计算词向量
    print ('Training a Word2vec model...')
    get_train_vecs(combined)

    combined_vecs, y = get_data()  # 导入训练数据和测试数据
    svm_train(combined_vecs, y)  # 训练svm并保存模型

    clf = joblib.load('./isoForest_data/isoForest_model/isoForest_model.pkl')

    apple = []
    fp = open('./data/testwhoisUnormal.txt','r',encoding='utf8')
    for line in fp.readlines():
        svm_predict(line, clf)
        #apple.append(line)
    apple.append('非常差')
    apple.append('哈士奇是一条不错的狗')
    apple.append('手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了')
    apple.append('电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如')
    apple.append('蒙牛一般般吧')
    apple.append('昨天打开书才翻了几页，书页纷纷掉落了，请问怎么回事？')
    apple.append('这本书是我的人生导师')
    apple.append('你是SB')
    apple.append('你个小坏蛋')
    apple.append('你最漂亮，怎么可能？')
    apple.append('你最漂亮')
    apple.append('你没我漂亮')
    apple.append('我们不喜欢喝蒙牛')
    apple.append('谁还会买中兴手机')
    apple.append('虽然这部手机不怎么好看，但是性价比确实高')
    apple.append('手机运行速度是快，但是真的能不能别卖翻新机给我')
    apple.append('这部手机真的不错')
    apple.append('蒙牛牛奶真的不错')
    apple.append('好评')
    apple.append('差评')

    svm_predict(apple[0],clf)
    svm_predict(apple[1],clf)
    svm_predict(apple[2],clf)
    svm_predict(apple[3],clf)
    svm_predict(apple[4],clf)
    svm_predict(apple[5],clf)
    svm_predict(apple[6],clf)
    svm_predict(apple[7],clf)
    svm_predict(apple[8],clf)
    svm_predict(apple[9], clf)
    svm_predict(apple[10], clf)
    svm_predict(apple[11], clf)
    svm_predict(apple[12], clf)
    svm_predict(apple[13], clf)
    svm_predict(apple[14], clf)
    svm_predict(apple[15], clf)
    svm_predict(apple[16], clf)
    svm_predict(apple[17], clf)
    svm_predict(apple[18], clf)
    svm_predict(apple[19], clf)


