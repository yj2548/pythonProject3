
from konlpy.tag import Okt
from gensim import corpora
from gensim import models
import csv
import pandas as pd
import os

Okt = Okt()
for i in range (1, 6):
    f = open("C:/Users/yjjj0/OneDrive/바탕 화면/메모장 크롤링 테스트{}.txt".format(i), 'r', encoding='UTF8')
    content = f.read()

    filtered_content = content.replace('.', '').replace(',','').replace("'","").replace('·', ' ').replace('=','').replace('\n','')
    Okt_morphs = Okt.pos(filtered_content)
    Noun_words = []
    for word, pos in Okt_morphs:
        if pos == 'Noun':
            if len(word) >= 2:
                Noun_words.append(word)

    documents= Noun_words

    stoplist = ('.!?')                                        # 불용어 처리
    texts = [[word for word in document.split() if word not in stoplist]
            for document in documents]

    dictionary = corpora.Dictionary(texts)                    # 사전 생성 (토큰화)
    print(dictionary)

    corpus = [dictionary.doc2bow(text) for text in texts]     # 말뭉치 생성 (벡터화)
    print('corpus : {}'.format(corpus))

    #-----------------------------------------------------------------
    num_topics=1
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state = 1)                   # 모델구축

    for t in lda.show_topics():  # 주제마다 출현 확률이 높은 단어 순으로 출력
       print(t)


    top_words_per_topic = []
    for t in range(lda.num_topics):
        top_words_per_topic.extend([(t, ) + x for x in lda.show_topic(t, topn = 5)])

    word_df = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])

    top10 = word_df.head(10)
    if not os.path.exists('top_words.csv'):
        top10.to_csv('top_words.csv', index=False, mode='w', encoding='utf-8-sig')
    else:
        top10.to_csv('top_words.csv', index=False, mode='a', encoding='utf-8-sig', header=False)
    print(top_words_per_topic)