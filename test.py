from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import configuration

from sklearn import preprocessing

from datasets import load_dataset
from vocab import build_dictionary
from model import LTS
from recall import i2t, t2i

from collections import OrderedDict, defaultdict

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", "checkpoint_files/",
					   "Directory containing model checkpoints.")

tf.flags.DEFINE_string("test_dataset_name", "f8k", 
					   "name of the testing dataset")

def test_model():
	"""Evaluate the LTS model."""
	model_config = configuration.ModelConfig()
	model_config.data = FLAGS.test_dataset_name

	test_ims_path = model_config.data + '/' + model_config.data + '_test_ims.npy'
	test_caps_path = model_config.data + '/' + model_config.data + '_test_caps.txt'

	print ('Loading dataset ...')
	test_caps = []
	with open(test_caps_path) as f:
		for line in f:
			test_caps.append(line.strip())

    test_ims = numpy.load(test_path)

    # Normalization
    test_vgg_feature = test_ims[:,:1536]
	test_NIC_feature = preprocessing.scale(test_ims[:,1536:])

	#load dictionary
	print ('loading dictionary')
	with open('%s.dictionary.pkl'%model_config.data, 'rb') as f:
		worddict = pkl.load(f)
	n_words = len(worddict)
	model_config.n_words = n_words
	model_config.worddict = worddict

	# build the model for evaluation
	model = LTS(model_config)
	model.build()

	# Create the Saver to restore model Variables.
	#saver = tf.train.import_meta_graph('checkpoint_files/model.ckpt-901.meta')
	saver = tf.train.Saver()

	with tf.Session() as sess:
		model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
		if not model_path:
			print("Skipping testing. No checkpoint found in: %s" % FLAGS.checkpoint_dir)
			return

		print("Loading model from checkpoint: %s" % model_path)
		saver.restore(sess, model_path)
		print("Successfully loaded checkpoint: %s" % model_path)

		# encode images into the text embedding space
		images = getTestImageFeature(sess, model, test_vgg_feature, test_NIC_feature)

		# encode sentences into the text embedding space
		features = getTestTextFeature(sess, model, model_config, test_caps)

		(r1, r5, r10, medr) = i2t(images, features)
		print ("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
		(r1i, r5i, r10i, medri) = t2i(images, features)
		print ("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))


def main():
	assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
	assert FLAGS.test_dataset_name, "--test_dataset_name is required"
	
	test_model()

def getTestImageFeature(sess, model, VGG_feature, NIC_feature):
	"""Encode images into the text embedding space.

	Inputs:
		sess: current session 
		model: model graph
		VGG_feature: VGG features in test dataset
		NIC_feature: NIC features in test dataset

	Output:
		image embedding vectors
	"""
	images = sess.run(model.image_embedding,
					  feed_dict={model.VGG_pred_data: VGG_feature,
								 model.NIC_pred_data: NIC_feature,
								 model.keep_prob: 1.0,
								 model.phase: False})
	return images


def getTestTextFeature(sess, model, model_config, test_caps):
	"""Encode sentences into the text embedding soace

	Inputs:
		sess: current session
		model: current model graph
		model_config: configurations for model parameters
		test_caps: sentences in test dataset

	Output:
		sentence embedding vectors
	"""
	features = np.zeros((len(test_caps), model_config.dim), dtype='float32')

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
	for w in model_config.worddict.keys():
		d[w] = 1

	# Get features
	for k in ds.keys():
		numbatches = len(ds[k]) // model_config.batch_size + 1
		for minibatch in range(numbatches):
			caps = ds[k][minibatch::numbatches]
			caption = [captions[c] for c in caps]

			seqs = []
			for i, cc in enumerate(caption):
				seqs.append([model_config.worddict[w] if w in model_config.worddict.keys() else 1 for w in cc])

			x = np.zeros((k+1, len(caption))).astype('int64')
			x_mask = np.zeros((k+1, len(caption))).astype('float32')
			for idx, s in enumerate(seqs):
				x[:k,idx] = s
				x_mask[:k+1,idx] = 1.

			ff = sess.run(model.text_embedding,
						  feed_dict={model.ls_pred_data: x.T,
									 model.input_mask: x_mask.T,
									 model.keep_prob: 1.0,
									 model.phase: False})
			for ind, c in enumerate(caps):
				features[c] = ff[ind]

	return features



if __name__ == '__main__':
	main()