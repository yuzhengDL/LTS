import cPickle as pkl
import random
import tensorflow as tf
import numpy as np
import utils as ut
import math
import datetime
import sys

from collections import OrderedDict, defaultdict

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import homogeneous_data
from datasets import load_dataset
from vocab import build_dictionary
from generator import GEN
import recall

FEATURE_SIZE = [4096, 512, 1024]	 #image feature size: 4096, txt feature size: 1024
G_HIDDEN_SIZE = [2048, 1024, 512, 256]

workdir = 'checkpoint_files'
GAN_MODEL_BEST_FILE = workdir + '/gan_best_nn.ckpt'

def main(data='f30k',	#f8k, f30k, coco
		 margin=0.3,
		 dim=1024,		#word embedding dimension
		 dim_image=[4096,512],
		 dim_word=1024,
		 hidden_size=G_HIDDEN_SIZE,
		 max_epochs=30,
		 weight_decay=0.0,
		 maxlen_w=100,
		 batch_size=128,
		 lrate=0.0002):

	# Model options
	model_options = {}
	model_options['data'] = data
	model_options['margin'] = margin
	model_options['dim'] = dim
	model_options['dim_image'] = dim_image
	model_options['dim_word'] = dim_word
	model_options['hidden_size'] = hidden_size
	model_options['max_epochs'] = max_epochs
	model_options['weight_decay'] = weight_decay
	model_options['maxlen_w'] = maxlen_w
	model_options['batch_size'] = batch_size
	model_options['lrate'] = lrate

	#loading dataset
	print 'Loading dataset ...'
	(train_caps, train_ims), (test_caps, test_ims) = load_dataset(name=data, load_train=True)[:2]

	#create and save dictionary
	print 'creating dictionary'
	worddict = build_dictionary(train_caps+test_caps)[0]
	n_words = len(worddict)
	model_options['n_words'] = n_words
	print 'dictionary size: ' + str(n_words)
	with open('f8k.dictionary.pkl', 'wb') as f:
		pkl.dump(worddict, f)

	#inverse dictionary
	word_idict = dict()
	for kk, vv in worddict.iteritems():
		word_idict[vv] = kk

	#initialize G and D
	print 'Building G and D ...'
	generator = GEN(model_options, param=None)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	saver = tf.train.Saver()
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())

	print('start adversarial training')
	curr = 0.
	uidx = 0.
	train_iter = homogeneous_data.HomogeneousData([train_caps, train_ims], batch_size=batch_size, maxlen=maxlen_w)

	ims_label = np.array([0., 1.])
	ls_label = np.array([1., 0.])

	for epoch in xrange(max_epochs):
		# Train G
		print 'Epoch ', epoch

		if epoch == 15:
			lrate = lrate / 10

		for x, im in train_iter:
			uidx += 1
                        lamb = (2 / (1 + np.exp(-10 * (epoch / max_epochs)))) - 1

			ls, mask, im = homogeneous_data.prepare_data(x, im, worddict, maxlen=maxlen_w, n_words=n_words)
			vgg_feature = im[:,:4096]
			NIC_feature = im[:,4096:]
			pred_data_label = np.concatenate((np.tile(ims_label, (len(im),1)), np.tile(ls_label, (len(ls.T),1))), axis=0)

			# Start domain classifier training
			_, cost = sess.run([generator.d_updates, generator.accuracy],
							 feed_dict={generator.vgg_pred_data: vgg_feature,
										generator.NIC_pred_data: NIC_feature,
										generator.ls_pred_data: ls.T,
										generator.input_mask: mask.T,
										generator.pred_data_label: pred_data_label,
										generator.gen_keep_prob: 1.0,
										generator.gen_phase: 0,
										generator.dis_keep_prob: 0.5,
										generator.learning_rate: lrate})
			#print(cost)

			#embedding training
			_, cost_E = sess.run([generator.g_updates_E, generator.loss_E],
							 feed_dict={generator.vgg_pred_data: vgg_feature,
										generator.NIC_pred_data: NIC_feature,
										generator.ls_pred_data: ls.T,
										generator.input_mask: mask.T,
										generator.pred_data_label: pred_data_label,
										generator.lamb: lamb,
										generator.gen_keep_prob: 0.5,
										generator.gen_phase: 1,
										generator.dis_keep_prob: 1.0,
										generator.learning_rate: lrate})
			'''
			# Adversarial training
			pred_data_label_reverse = 1 - pred_data_label
			_, cost = sess.run([generator.g_updates_D, generator.accuracy],
							 feed_dict={generator.vgg_pred_data: vgg_feature,
										generator.NIC_pred_data: NIC_feature,
										generator.ls_pred_data: ls.T,
										generator.input_mask: mask.T,
										generator.pred_data_label: pred_data_label_reverse,
										generator.gen_keep_prob: 0.5,
										generator.gen_phase: 1,
										generator.dis_keep_prob: 1.0,
										generator.learning_rate: lrate})
			#print(cost)
			'''

			if np.mod(uidx, 10) == 0:
				print 'Epoch ', epoch, 'Update ', uidx, 'Cost ', cost_E

			if np.mod(uidx, 100) == 0:
				print('test ...')

				# encode images into the text embedding space
				test_vgg_feature = test_ims[:,:4096]
				test_NIC_feature = test_ims[:,4096:]
				images = sess.run(generator.gen_ims_layer_2,
								  feed_dict={generator.vgg_pred_data: test_vgg_feature,
											 generator.NIC_pred_data: test_NIC_feature,
											 generator.gen_keep_prob: 1.0,
											 generator.gen_phase: 0})

				# encode sentences into the text embedding space
				features = np.zeros((len(test_caps), dim), dtype='float32')

				# length dictionary
				ds = defaultdict(list)

				captions = []
				for s in test_caps:
					s = s.lower()
					captions.append(s.split())

				for i,s in enumerate(captions):
					ds[len(s)].append(i)

				#quick check if a word is in the dictionary
				d = defaultdict(lambda : 0)
				for w in worddict.keys():
					d[w] = 1

				# Get features
				for k in ds.keys():
					numbatches = len(ds[k]) / batch_size + 1
					for minibatch in range(numbatches):
						caps = ds[k][minibatch::numbatches]
						caption = [captions[c] for c in caps]

						seqs = []
						for i, cc in enumerate(caption):
							seqs.append([worddict[w] if w in worddict.keys() else 1 for w in cc])

						x = np.zeros((k+1, len(caption))).astype('int64')
						x_mask = np.zeros((k+1, len(caption))).astype('float32')
						for idx, s in enumerate(seqs):
							x[:k,idx] = s
							x_mask[:k+1,idx] = 1.

						ff = sess.run(generator.gen_ls_layer_2,
									  feed_dict={generator.ls_pred_data: x.T,
												 generator.input_mask: x_mask.T,
												 generator.gen_keep_prob: 1.0,
												 generator.gen_phase: 0})
						for ind, c in enumerate(caps):
							features[c] = ff[ind]

				(r1, r5, r10, medr) = recall.i2t(images, features)
				print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
				(r1i, r5i, r10i, medri) = recall.t2i(images, features)
				print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)

				currscore = r1 + r5 + r10 + r1i + r5i + r10i
				if currscore > curr:
					curr = currscore

					# Save model
					print ('Saving...')
					saver.save(sess, GAN_MODEL_BEST_FILE)
					print('done.')

    		if epoch == 29:
			ims = np.array([images[i] for i in range(0, len(images), 5)])
			#ls = np.array([features[i] for i in range(0, len(features), 5)])
			plt.close()
			model1 = TSNE(n_components=2)
			tsne1 = model1.fit_transform(ims)
			model2 = TSNE(n_components=2)
			tsne2 = model2.fit_transform(features)
			plt.figure(1)
			plt.scatter(tsne1[:, 0], tsne1[:, 1], c='r')
			plt.scatter(tsne2[:, 0], tsne2[:, 1], c='b')
			plt.show()

if __name__ == '__main__':
	main()
