import numpy as np
import tensorflow as tf
import cPickle


FEATURE_SIZE =[1024, 1024]
HIDDEN_SIZE = 10
BATCH_SIZE = 2048
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.001
G_LEARNING_RATE = 0.001
TEMPERATURE = 0.2
LAMBDA = 0.5

workdir = 'f8k_4096'
GAN_MODEL_BEST_FILE = workdir + 'gan_best_nn.model'

def evalrank(data, split='dev'):
	'''
	evaluate a trained model on either dev or test
	data options: f8k, f30k, coco
	'''
	print 'loading gan model best parameters ...'
	param_best = cPickle.load(open(GAN_MODEL_BEST_FILE))
	assert param_best is not None

	print 'loading dataset ...'
	if split == 'dev':
		caps = np.load(data + '/f8k_dev_rnn.npy')
		ims = np.load(data + '/f8k_dev_ims.npy')
	else:
		ims = np.load(data + '/f8k_test_ims.npy')
		caps = np.load(data + '/f8k_test_rnn.npy')

	generator_best = GEN(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, G_LEARNING_RATE, temperature=TEMPERATURE, param=param_best)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.initialize_all_variables())

	(r1, r5, r10, medr) = i2t(sess, generator_best, ims, caps)
	print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
	(r1i, r5i, r10i, medri) = t2i(sess, generator_best, ims, caps)
	print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)

def i2t(images, captions, npts=None):
	'''
	Image->Text(image annotation)
	Images: (5N, K) matrix of images
	Captions: (5N, K) matrix of captions
	'''
	if npts == None:
		npts = images.shape[0] / 5

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
		npts = images.shape[0] / 5
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


