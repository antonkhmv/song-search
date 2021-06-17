import random
import pandas as pd
import numpy as np
import multiprocessing as mp
from collections import defaultdict
import pickle
import os

class Document:
	def __init__(self, title, artist, lyrics, index):
		# можете здесь какие-нибудь свои поля подобавлять
		self.artist = artist
		self.title = title
		self.lyrics = lyrics
		self.index = index
    
	def format(self, query):
		lyrics = str(self.lyrics[:150]) + ' ...'
		# возвращает пару тайтл-текст, отформатированную под запрос
		return [self.artist + ' - ' + self.title, lyrics]

data = None
words = None
inverted_index = None
documents = None
word2vec = None

data_path = r'data/data.csv'
words_path = r'data/words.csv'
inverted_index_path = r'data/inverted_index.pkl'
filtered_stemmed_lyrics_path = r'data/filtered_stemmed_lyrics.pkl'

def transform_string(s):
	return ''.join(filter(lambda x: ord('a') <= ord(x) <= ord('z') or ord(x) == ord(' '), \
				map(str.lower, s)))

def create_doc(song):
	# song = (index, title, artist, lyrics)
	return Document(title=song[1], artist=song[2], lyrics=song[3], index=song[0])

def this(x):
	return x

def build_index(path):
	path = path.replace('\\', '/') + '/'

	global data_path, words_path, inverted_index_path, filtered_stemmed_lyrics_path, glob_path, documents, data,\
		inverted_index
	
	data_path, words_path, inverted_index_path, filtered_stemmed_lyrics_path = \
		path + data_path, path + words_path, path + inverted_index_path, path + filtered_stemmed_lyrics_path

	
	# считывает сырые данные и строит индекс
	print('reading data.csv...')
	data = pd.read_csv(data_path)
	
	print(data_path)
	
	print('making a list of documents...')
	documents = np.array(list(map(create_doc, data.itertuples())))
	
	print('building inverted index...')
		
	# кэширование (иначе каждый раз очень долго запускается)
	if os.path.isfile(inverted_index_path):
		print('inverted index cache file found')
		print('reading inverted index...')
		with open(inverted_index_path, 'rb') as f:
			 inverted_index = pickle.load(f)
	else:
		inverted_index = defaultdict(list)
		with mp.Pool(4) as pool:
			for doc in pool.map(this, documents):
				seen_words = set()
				for w in transform_string(doc.lyrics).split() + transform_string(doc.title + ' ' + doc.artist).split():
					if w not in seen_words:
						seen_words.add(w)
						inverted_index[w].append(doc.index)
		print('caching inverted index...')
		with open(inverted_index_path, 'wb') as f:
			pickle.dump(inverted_index, f, pickle.HIGHEST_PROTOCOL)
			
 
from sklearn.feature_extraction.text import TfidfVectorizer

def get_embedding_from_str(s):
	global word2vec
	emb = np.zeros(300)
	cnt = 0
	for w in transform_string(s).split():
		if w in word2vec:
			emb += word2vec[w]
			cnt += 1
	return emb / cnt if cnt > 0 else emb
	
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')

def stem_and_filter_str(s):
	global stemmer
	sent = transform_string(s)
	return ' '.join([stemmer.stem(word) for word in sent.split()])

def initialize_models():
	global data, word2vec, idf_vectorizer, inverted_index
	
	idf_vectorizer = TfidfVectorizer(stop_words='english')
	
	print('stemming and filtering...')
	
	filtered_stemmed_lyrics = None
	
	# кэширование (иначе каждый раз очень долго запускается)
	if os.path.isfile(filtered_stemmed_lyrics_path) :
		print('filtered_stemmed_lyrics.pkl file found')
		print('reading filtered_stemmed_lyrics.pkl...')
		with open(filtered_stemmed_lyrics_path, 'rb') as f:
			 filtered_stemmed_lyrics = pickle.load(f)
	else:
		with mp.Pool(4) as pool:
			filtered_stemmed_lyrics = list(pool.map(stem_and_filter_str, data.lyrics))
		#print(len(filtered_stemmed_lyrics))
		print('caching filtered_stemmed_lyrics.pkl...')
		with open(filtered_stemmed_lyrics_path, 'wb') as f:
			pickle.dump(filtered_stemmed_lyrics, f, pickle.HIGHEST_PROTOCOL)
			
	print('initialzing tf-idf...')
	idf_vectorizer.fit(filtered_stemmed_lyrics)
    
	print('reading words.csv...')
	words = pd.read_csv(words_path)
	
	print('converting words to a dictionary...')
	word2vec = {row[1] : list(row)[2:] for row in words.itertuples()}
	
import scipy
	
def normalize(arr):
	length = 0
	
	if type(arr) == scipy.sparse.csr_matrix:
		cx = scipy.sparse.coo_matrix(arr)
		length = np.sqrt(np.dot(cx.data, cx.data))
		return { a : b / length for a, b in zip(cx.col, cx.data)}
	else:
		length = np.sqrt(np.dot(arr, arr))
		if length == 0:
			length = 1
		return arr / length
	
def cosine_simularity(vec1, vec2):
	if type(vec1) == scipy.sparse.csr_matrix:
		d1 = normalize(vec1)
		d2 = normalize(vec2)
		keys = d1.keys() & d2.keys()
		return np.dot(list(map(d1.get, keys)), list(map(d2.get, keys)))
	else:
		return np.dot(normalize(vec1), normalize(vec2))
	
def score(query, document):
	query_emb = get_embedding_from_str(query)
	
	title_emb = get_embedding_from_str(document.title)
	author_emb = get_embedding_from_str(document.artist)
	
	#print(query_emb, document_emb)
	title_score = cosine_simularity(query_emb, title_emb)
	author_score = cosine_simularity(query_emb, author_emb)
	
	idf_embeds = idf_vectorizer.transform([document.lyrics, query])
	
	lyrics_score = cosine_simularity(idf_embeds[0,:], idf_embeds[1,:])
	
	# возвращает какой-то скор для пары запрос-документ
	# больше -- релевантнее
	return 0.25 * author_score + 0.5 * title_score + 0.25 * lyrics_score

def retrieve(query):
	global inverted_index
	# возвращает начальный список релевантных документов
	# (желательно, не бесконечный)
	candidates = set()
	try:
		for i, w in enumerate(transform_string(query).split()):
			if w in inverted_index:
				if i == 0:
					candidates = candidates.union(inverted_index[w])
				else:
					candidates = candidates.intersection(inverted_index[w])
	except:
		print('except')
	print(len(inverted_index))
	return documents[list(candidates)[:3000]]
