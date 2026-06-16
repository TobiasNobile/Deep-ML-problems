import numpy as np

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""
	tf = {}
	df = {}
	for d in range(len(corpus)):
		for term in query:
			document = corpus[d]
			count = document.count(term)
			tf[(term, d)] = count / len(document)

			if count > 0:
				df[term] = df.get(term, 0) + 1
			else:
				df[term] = df.get(term, 0)

	for term in df:
		N = len(corpus)
		df[term] = np.log((N+1)/(df[term]+1))+1
	
	tf_idf_scores = []
	for d in range(len(corpus)):
		tf_idf_doc = []
		for term in query:
			tf_idf = tf[(term, d)]*df[term]
			tf_idf_doc.append(tf_idf)
		tf_idf_scores.append(tf_idf_doc)

	return tf_idf_scores


