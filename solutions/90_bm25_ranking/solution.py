import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):

	df = {term:0 for term in query}
	for term in query:
		for d in corpus:
			if term in d:
				df[term] = df[term] + 1
		
	avg_doc_len = np.mean([len(d) for d in corpus])
	N = len(corpus)

	scores = []
	for document in corpus:
		c = Counter(document)

		tf = [c[term] for term in query]
		tf = np.array(tf)

		dl = len(document)

		idf = [np.log((N+1)/(df[term]+1)) for term in query]
		bm25 = idf*tf*(k1+1)
		bm25 /= tf + k1*(1-b+b*dl/avg_doc_len)
		bm25 = sum(bm25)
		scores.append(bm25)
	
	return np.round(scores,3)