import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
documents = ["Human machine interface for lab abc computer applications",
	              "A survey of user opinion of computer system response time",
	              "The EPS user interface management system",
	              "System and human system engineering testing of EPS",
	              "Relation of user perceived response time to error measurement",
	              "The generation of random binary unordered trees",
	              "The intersection graph of paths in trees",
	              "Graph minors IV Widths of trees and well quasi ordering",
		      "Graph minors A survey"]

#-------------------tokenizes the document and stores it in a list-------------------------------------------------#
texts=[]
for document in documents:
	temp=[]
	for word in document.lower().split():
		temp.append(word)
	texts.append(temp)
print texts
#------------------makes dictionary token with its frequency and assigns unique id to each token---------------------------------------------------------------------#
dictionary = corpora.Dictionary(texts)
print dictionary
dictionary.save('deerwester.dict')
print(dictionary.token2id)
#------------------makes a vector in which it returns a list which contains id along with its frequencies in a given document. Here the documetns is a dictionary--------------------#
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('deerwester.mm', corpus)
print(corpus)
#-----------------------a model is initialised----------------------------------------------------------#
tfidf = models.TfidfModel(corpus)
#-----------------------the entire id frequency is transformed to tfidef model--------------------------#
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
	print(doc)
#------------------------coverts tfidf to lsi model-----------------------------------------------------#
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) #initialises an LSI transformation
corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(2)
