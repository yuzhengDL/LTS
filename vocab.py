"""
Constructing and loading dictionaries
"""
import numpy
from collections import OrderedDict
from collections import Counter

def build_dictionary(text):
	"""
	Build a dictionary
	text: list of sentences (pre-tokenized)
	"""
	counter = Counter()
	captions = []
	for s in text:
		tokenized_captions = []
		s = s.lower()
		tokenized_captions.extend(s.split())
		#tokenized_captions.append("</s>")
		captions.append(tokenized_captions)

	for cc in captions:
		counter.update(cc)
	print("Total words:", len(counter))

	word_counts = [x for x in counter.items() if x[1] >= 4]
	word_counts.sort(key=lambda x:x[1], reverse=True)
	print("Words in vocabulary:", len(word_counts))
	reverse_vocab = [x[0] for x in word_counts]
	worddict = OrderedDict([(x, y+2) for (y,x) in enumerate(reverse_vocab)])
	worddict['<eos>'] = 0
	worddict['UNK'] = 1
	wordcount = OrderedDict(word_counts)

	return worddict, word_counts

def load_dictionary(loc):
	"""
	Load a dictionary
	"""
	with open(loc, 'rb') as f:
		worddict = pkl.load(f)
	return worddict

def save_dictionary(worddict, wordcount, loc):
	"""
	Save a dictionary to the specified location
	"""
	with open(loc, 'wb') as f:
		pkl.dump(worddict, f)
		pkl.dump(wordcount, f)


