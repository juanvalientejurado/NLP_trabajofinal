from numba import jit, cuda
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.phrases import Phrases
import aux_functions as af
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import FastText , KeyedVectors, LdaModel, CoherenceModel
from sklearn.manifold import TSNE
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

#Descargamos los paquetes 
#af.check_nltk_packages()



#Lectura del dataframe
projects_df = pd.read_excel('dataset\projects.xlsx')

##############################     PREPROCESADO DEL TEXTO     ################################

#Creacion de corpus para tratar exclusivamente el texto (título y resumen)
raw_corpus= [(projects_df['title'][ind]+' '+ projects_df['summary'][ind]) for ind in projects_df.index]
raw_corpus_df = pd.DataFrame(raw_corpus,columns=['Title_summ'])
raw_corpus_df['Title_summ_clean'] = raw_corpus_df['Title_summ'].apply(af.prepare_data)

corpus = raw_corpus_df['Title_summ_clean']
n_grams = Phrases(corpus, min_count=2, threshold=20)
corpus = [el for  el in n_grams[corpus]]

#N-gram detection
corpus_df = pd.DataFrame(columns=['Title_summ', 'Title_summ_clean'])
corpus_df['Title_summ'] = raw_corpus_df['Title_summ'].copy()
corpus_df['Title_summ_clean'] = corpus

print(corpus_df.head())

############# VECTORIZACIÓN ###############################
no_below = 4
no_above = .80

D = Dictionary(corpus_df['Title_summ_clean'])
D.filter_extremes(no_below=no_below, no_above=no_above)

##### TFIDF ##############

my_corpus_bow = [D.doc2bow(doc) for doc in corpus]

tfidf = TfidfModel(my_corpus_bow)
my_corpus_tfidf = tfidf[my_corpus_bow]

corpus_df['TFIDF'] = my_corpus_tfidf

####### WORD EMBEDDINGS #############################
#Descomentar si no se tienen los word_vectors
'''
model_fast_text = FastText(sentences=corpus, vector_size= 300, window= 5, min_count = 5, seed= 1)

word_vectors = model_fast_text.wv

#Guardamos los word_vectors para que la ejecución no tarde tanto

word_vectors.save("model_wv.wordvectors")

'''
word_vectors = KeyedVectors.load("word_vectors/model_wv.wordvectors", mmap = 'r')

corpus_df['Embeddings'] = word_vectors

#Representación TSNE
'''
tsne = TSNE(init = 'random')
embed_tsne = tsne.fit_transform(word_vectors.vectors)

fig, ax = plt.subplots(figsize=(16,16))
for idx, word in enumerate((list(word_vectors.key_to_index.keys())[:500])):
    plt.scatter(*embed_tsne[idx, :], color= 'steelblue')
    plt.annotate(word, (embed_tsne[idx,0], embed_tsne[idx,1]), alpha = 0.7)

plt.grid()
plt.show()
'''

############## LDA ######################
'''
n_topics  = [5, 10 ,15, 20, 25, 50]
coherences= []

for n in n_topics:
    ldag = LdaModel(corpus = my_corpus_bow, id2word = D, num_topics =n)
    coherencemodel = CoherenceModel(ldag, texts = corpus, dictionary= D, coherence= 'c_v')
    print("Estimating coherence for a model with {} topics".format(n))
    coherences.append(coherencemodel.get_coherence())

plt.plot(n_topics, coherences)
plt.xlabel("Número de tópicos")
plt.ylabel('Coherencia media')
plt.show()
#ajustar el número de tópicos usando la coherencia
'''
ldag = LdaModel(corpus = my_corpus_bow, id2word = D, num_topics =20)
corpus_lda = ldag[my_corpus_bow]

corpus_df['LDA_20'] = corpus_lda
print(corpus_df.head())
'''
vis_data = pyLDAvis.sklearn.prepare(ldag, my_corpus_bow, D)
pyLDAvis.display(vis_data)
'''
corpus_df.to_csv('corpus_df.csv')



