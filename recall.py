import numpy as np
import tensorflow as tf

def i2t(images, captions, npts=None):
	'''
	Image->Text(image annotation)
	Images: (5N, K) matrix of images
	Captions: (5N, K) matrix of captions
	'''
	if npts == None:
		npts = images.shape[0] // 5

	ranks = np.zeros(npts)
	for index in range(npts):
		#get query image
		im = images[5 * index].reshape(1, images.shape[1])

		#compute scores
		d = np.dot(im, captions.T).flatten()
		inds = np.argsort(d)[::-1]

		#score
		rank = 1e20
		for i in range(5 * index, 5 * index + 5, 1):
			tmp = np.where(inds == i)[0][0]
			if tmp < rank:
				rank = tmp
		ranks[index] = rank

	#compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	return (r1, r5, r10, medr)

def t2i(images, captions, npts=None):
	'''
	text->images(image search)
	images: (5N, K) matrix of images
	captions: (5N, K) matrix of captions
	'''
	if npts == None:
		npts = images.shape[0] // 5
	ims = np.array([images[i] for i in range(0, len(images), 5)])

	ranks = np.zeros(5 * npts)
	for index in range(npts):

		#get query captions
		queries = captions[5*index : 5*index + 5]

		d = np.dot(queries, ims.T)
		inds = np.zeros(d.shape)
		for i in range(len(inds)):
			inds[i] = np.argsort(d[i])[::-1]
			ranks[5 * index + i] = np.where(inds[i] == index)[0][0]

	#compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	return (r1, r5, r10, medr)


