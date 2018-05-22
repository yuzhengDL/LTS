from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class LTS:
	"""Cross-media retrieval base on https://doi.org/10.1007/978-3-319-64689-3_24.

	"Learning a Limited Text Space for Cross-Media Retrieval"
	Zheng Yu, Wenmin Wang, Mengdi Fan
	"""


	def __init__(self, options):
		"""Basic setup

		Args:
			options: Object containing configuration parameters.
		"""

		self.config = options

		self.initializer = tf.truncated_normal_initializer(
			mean=0.0, stddev=0.1)

		# Dropout rate
		self.keep_prob = tf.placeholder(tf.float32, name="dropout")
		# Training phase
		self.phase = tf.placeholder(tf.bool, name="traning_phase")
		self.learning_rate = tf.placeholder(tf.float32, name='lrate')

		# Text space embedding vectors fot image features.
		self.image_embedding = None

		# Text space embedding vectors for sentences.
		self.text_embedding = None

		# The embedding loss for the optimizer to optimize.
		self.embedding_loss = None

		# The update operation to optimize the embedding loss
		self.updates = None

	def build_image_embeddings(self):
		"""Generate image embedding in a limited text space

		Inputs: 
			self.VGG_pred_data
			self.NIC_pred_data

		Output:
			self.image_embeddings
		"""
		self.VGG_pred_data = tf.placeholder(tf.float32, shape=[None, self.config.dim_VGG], name="ims_vgg_pred_data")
		self.NIC_pred_data = tf.placeholder(tf.float32, shape=[None, self.config.dim_NIC], name="ims_NIC_pred_data")

		# Map image space features intoa limited text space.
		with tf.variable_scope("IncV4_embedding") as IncV4_scope:
			IncV4_embedding = tf.contrib.layers.fully_connected(
				inputs=self.VGG_pred_data,
				num_outputs=self.config.dim,
				activation_fn=None,
				weights_initializer=self.initializer,
				biases_initializer=None,
				scope=IncV4_scope)

			IncV4_embedding = tf.nn.relu(tf.layers.batch_normalization(IncV4_embedding, training=self.phase, name="IncBN"))

		with tf.variable_scope("NIC_embedding") as NIC_scope:
			NIC_embedding = tf.contrib.layers.fully_connected(
				inputs=self.NIC_pred_data,
				num_outputs=self.config.dim,
				activation_fn=None,
				weights_initializer=self.initializer,
				biases_initializer=None,
				scope=NIC_scope)
			NIC_embedding = tf.nn.relu(tf.layers.batch_normalization(NIC_embedding, training=self.phase, name="NicBN"))

		self.image_embedding = tf.nn.l2_normalize(tf.layers.batch_normalization(
									tf.add(IncV4_embedding, NIC_embedding), 
									training=self.phase, name="SumBN"), 1, name="image_embedding")


	def build_seq_embeddings(self):
		"""Generate text embeddings

		Inputs:
			self.ls_pred_data
			self.input_mask

		Output:
			self.text_embedding
		"""
		self.ls_pred_data = tf.placeholder(tf.int64, shape=[None, None], name="ls_pred_data")
		self.input_mask = tf.placeholder(tf.int64, shape=[None, None], name='mask')

		with tf.variable_scope("seq_embedding"):
			embedding_map = tf.get_variable(
				name="word_embedding", 
				shape=[self.config.n_words, self.config.dim_word],
				initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
			seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.ls_pred_data)

		# BLSTM
		lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.dim, state_is_tuple=True)
		lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.dim, state_is_tuple=True)

		lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(
			lstm_cell_fw,
			output_keep_prob=self.keep_prob)
		lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(
			lstm_cell_bw, 
			output_keep_prob=self.keep_prob)

		with tf.variable_scope("lstm") as lstm_scope:
			initial_state_fw = lstm_cell_fw.zero_state(batch_size=tf.shape(self.ls_pred_data)[0], dtype=tf.float32)
			initial_state_bw = lstm_cell_bw.zero_state(batch_size=tf.shape(self.ls_pred_data)[0], dtype=tf.float32)

			sequence_length = tf.reduce_sum(self.input_mask, 1)
			
			_, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
		                                                     cell_bw=lstm_cell_bw,
		                                                     inputs=seq_embeddings,
		                                                     sequence_length=sequence_length,
		                                                     initial_state_fw=initial_state_fw,
		                                                     initial_state_bw=initial_state_bw,
		                                                     scope=lstm_scope)

			self.text_embedding = tf.nn.l2_normalize(tf.add(final_state[0][1], final_state[1][1]) / 2, 1, name="text_embedding")


	def build_loss(self):
		"""Builds the pairwise ranking loss function.

		Inputs:
			self.image_embedding
			self.text_embedding

		Output:
			self.embedding_loss
		"""
		
		with tf.name_scope("pairwise_ranking_loss"):
			# Compute losses
			pred_score = tf.matmul(self.image_embedding, self.text_embedding, transpose_b=True)
			diagonal = tf.diag_part(pred_score)
			cost_s = tf.maximum(0., self.config.margin - diagonal + pred_score)
			cost_im = tf.maximum(0., self.config.margin - tf.reshape(diagonal, [-1, 1]) + pred_score)
			cost_s = tf.multiply(cost_s, (tf.ones([tf.shape(self.image_embedding)[0], tf.shape(self.image_embedding)[0]]) - tf.eye(tf.shape(self.image_embedding)[0])))
			cost_im = tf.multiply(cost_im, (tf.ones([tf.shape(self.image_embedding)[0], tf.shape(self.image_embedding)[0]]) - tf.eye(tf.shape(self.image_embedding)[0])))

			self.embedding_loss = tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)


	def build_optimizer(self):
		"""Initialize an optimizer.

		Inputs:
			All trainable variables

		Output:
			self.updates
		"""
		tvars = tf.trainable_variables()
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.embedding_loss, tvars), self.config.clip_gradients)
			optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.updates = optimizer.apply_gradients(zip(grads, tvars))


	def build(self):
		"""Create all ops for training."""
		self.build_seq_embeddings()
		self.build_image_embeddings()
		self.build_loss()
		self.build_optimizer()

		
