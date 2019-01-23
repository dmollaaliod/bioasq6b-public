from nnmodels import lstm

def test_sentences_to_ids():
	"""Convert each sentence to a list of word IDs."""
	sentences = [['my','first','sentence'],['my','ssecond','sentence'],['yes']]
	result = (([11095, 121], [11095, 1], [21402, 0]), (2, 2, 1))
	assert lstm.sentences_to_ids(sentences, 2) == result

def test_one_sentence_to_ids():
	"""Convert one sentence to a list of word IDs."""
	assert lstm.one_sentence_to_ids(['my','first','sentence'], 2) == ([11095, 121], 2)
	assert lstm.one_sentence_to_ids(['my','ssecond','sentence'], 2) == ([11095, 1], 2)
	assert lstm.one_sentence_to_ids(['yes'], 2) == ([21402, 0], 1)

def test_parallel_sentences_to_ids():
	"""Convert each sentence to a list of word IDs."""
	sentences = [['my','first','sentence'],['my','ssecond','sentence'],['yes']]
	results = (([11095, 121], [11095, 1], [21402, 0]), (2, 2, 1))
	assert lstm.parallel_sentences_to_ids(sentences, 2) == results

def test_snippets_to_ids():
	"""Convert the snippets to lists of word IDs."""
	snippets = [['sentence', 'one'], ['sentence'], ['two']]
	result = (([12205, 68, 0], [12205, 0, 0]), (2, 1))
	assert lstm.snippets_to_ids(snippets, 3, 2) == result

	snippets = [['sentence', 'three']]
	result = (([12205, 98, 0], [0, 0, 0]), (2, 0))
	assert lstm.snippets_to_ids(snippets, 3, 2) == result

def test_parallel_snippets_to_ids():
	"""Convert the batch of snippets to lists of word IDs."""
	snippets = [[['sentence', 'one'], ['sentence'], ['two']],[['sentence', 'three']]]
	result = ((([12205, 68, 0], [12205, 0, 0]), ([12205, 98, 0], [0, 0, 0])), ((2, 1), (2, 0)))
	assert lstm.parallel_snippets_to_ids(snippets, 3, 2) == result

