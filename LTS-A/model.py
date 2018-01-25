from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import cPickle

class GEN:
	def __init__(self, options, param=None):
		self.ims_vgg_feature_size = options['dim_image'][0]		#image feature size, default 4096D
		self.ims_NIC_feature_size = options['dim_image'][1]		#image feature size, default 512D

		self.dim_image_1 = options['hidden_size'][0]		#4096->1024
		self.dim_image_2 = options['hidden_size'][1]		#1024->512

		self.dim_merge_1 = options['hidden_size'][2]		#512->512
		self.dim_merge_2 = options['hidden_size'][3]		#512->256

		self.weight_decay = options['weight_decay']
		self.margin = options['margin']
		self.g_params = []
		self.d_params = []
		self.params_total = []

		#input data
		self.vgg_pred_data = tf.placeholder(tf.float32, shape=[None, self.ims_vgg_feature_size], name="ims_vgg_pred_data")
		self.NIC_pred_data = tf.placeholder(tf.float32, shape=[None, self.ims_NIC_feature_size], name="ims_NIC_pred_data")
		self.ls_pred_data = tf.placeholder(tf.int64, shape=[None, None], name="ls_pred_data")
		self.input_mask = tf.placeholder(tf.int64, shape=[None, None], name='mask')

		self.pred_data_label = tf.placeholder(tf.float32, shape=[None, 2], name="pred_data_label")
		self.learning_rate = tf.placeholder(tf.float32, name='lrate')
		self.lamb = tf.placeholder(tf.float32, name='lambda')

		self.gen_keep_prob = tf.placeholder(tf.float32, name="gen_dropout")
		self.gen_phase = tf.placeholder(tf.bool, name="gen_phase")
		self.dis_keep_prob = tf.placeholder(tf.float32, name="dis_dropout")

		self.initializer = tf.random_uniform_initializer(minval=-0.08, maxval=0.08)

		with tf.variable_scope('generator'):
			if param == None:
				self.embedding = tf.get_variable("word_embedding", [options['n_words'], options['dim_word']], initializer=self.initializer)

				self.W_ims_1 = tf.get_variable('ims_1_weight', [self.ims_vgg_feature_size, self.dim_image_1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.b_ims_1 = tf.get_variable('ims_1_b', [self.dim_image_1], initializer=tf.constant_initializer(0.0))

				self.W_ims_2 = tf.get_variable('ims_2_weight', [self.dim_image_1, self.dim_image_2], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.b_ims_2 = tf.get_variable('ims_2_b', [self.dim_image_2], initializer=tf.constant_initializer(0.0))

				self.W_ims_3 = tf.get_variable('ims_3_weight', [self.ims_NIC_feature_size, self.dim_image_2], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.b_ims_3 = tf.get_variable('ims_3_b', [self.dim_image_2], initializer=tf.constant_initializer(0.0))

				#domain classifier parameters
				self.W_merge_1 = tf.get_variable('merge_1_weight', [self.dim_image_2, self.dim_merge_1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), trainable=False)
				self.b_merge_1 = tf.get_variable('merge_1_b', [self.dim_merge_1], initializer=tf.constant_initializer(0.0), trainable=False)
				self.W_merge_2 = tf.get_variable('merge_2_weight', [self.dim_merge_1, self.dim_merge_2], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), trainable=False)
				self.b_merge_2 = tf.get_variable('merge_2_b', [self.dim_merge_2], initializer=tf.constant_initializer(0.0), trainable=False)
				self.W_merge_3 = tf.get_variable('merge_3_weight', [self.dim_merge_2, 2], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), trainable=False)
				self.b_merge_3 = tf.get_variable('merge_3_b', [2], initializer=tf.constant_initializer(0.0), trainable=False)
			else:
				self.W_ims_1 = tf.Variable(param[0])
				self.b_ims_1 = tf.Variable(param[1])
				self.W_ims_3 = tf.Variable(param[4])
				self.b_ims_3 = tf.Variable(param[5])
				self.embedding = tf.Variable(param[6])

				self.W_merge_1 = tf.Variable(param[7])
				self.b_merge_1 = tf.Variable(param[8])
				self.W_merge_2 = tf.Variable(param[9])
				self.b_merge_2 = tf.Variable(param[10])
				self.W_merge_3 = tf.Variable(param[11])
				self.b_merge_3 = tf.Variable(param[12])

			self.g_params.append(self.W_ims_1)
			self.g_params.append(self.b_ims_1)
			self.g_params.append(self.W_ims_3)
			self.g_params.append(self.b_ims_3)
			self.g_params.append(self.embedding)

			self.d_params.append(self.W_merge_1)
			self.d_params.append(self.b_merge_1)
			self.d_params.append(self.W_merge_2)
			self.d_params.append(self.b_merge_2)
			self.d_params.append(self.W_merge_3)
			self.d_params.append(self.b_merge_3)

		self.ls_inputs = tf.nn.embedding_lookup(self.embedding, self.ls_pred_data)
                '''
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(options['dim'], state_is_tuple=True)
		lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.gen_keep_prob)
		self.initial_state = lstm_cell.zero_state(batch_size=tf.shape(self.ls_pred_data)[0], dtype=tf.float32)

		self.sequence_length = tf.reduce_sum(self.input_mask, 1)
		_, self.final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
												inputs=self.ls_inputs,
												sequence_length=self.sequence_length,
												initial_state=self.initial_state,
												dtype=tf.float32)

		self.gen_ls_layer_2 = tf.nn.l2_normalize(self.final_state[1], 1)
                '''
                lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(options['dim'] / 2, state_is_tuple=True)
                lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=self.gen_keep_prob)
                lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(options['dim'] / 2, state_is_tuple=True)
                lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=self.gen_keep_prob)
                self.initial_state_fw = lstm_cell_fw.zero_state(batch_size=tf.shape(self.ls_pred_data)[0], dtype=tf.float32)
                self.initial_state_bw = lstm_cell_bw.zero_state(batch_size=tf.shape(self.ls_pred_data)[0], dtype=tf.float32)

                self.sequence_length = tf.reduce_sum(self.input_mask, 1)

                _, self.final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                                      cell_bw=lstm_cell_bw,
                                                                      inputs=self.ls_inputs,
                                                                      sequence_length=self.sequence_length,
                                                                      initial_state_fw=self.initial_state_fw,
                                                                      initial_state_bw=self.initial_state_bw)

                self.gen_ls_layer_2 = tf.nn.l2_normalize(tf.concat((self.final_state[0][1], self.final_state[1][1]), 1), 1)

		# image feature embedding, embed img features into text space from 4096->1024
		self.ims_vgg_layer_1 = tf.nn.xw_plus_b(self.vgg_pred_data, self.W_ims_1, self.b_ims_1)		#ims: [batch_size, 4096] -> [batch_size, 2048]
		self.ims_vgg_layer_1 = tf.nn.dropout(tf.nn.relu(self.ims_vgg_layer_1), self.gen_keep_prob)
		self.ims_vgg_layer_2 = tf.nn.xw_plus_b(self.ims_vgg_layer_1, self.W_ims_2, self.b_ims_2)

		self.ims_NIC_layer = tf.nn.xw_plus_b(self.NIC_pred_data, self.W_ims_3, self.b_ims_3)
                #self.gen_ims_layer_2 = batch_norm(self.ims_NIC_layer, center=True, scale=True, is_training=self.gen_phase, scope='gen_ims_layer_bn_1')
		#self.gen_ims_layer_2 = tf.nn.l2_normalize(self.gen_ims_layer_2, 1)
                self.gen_ims_layer_2 = batch_norm(tf.add(self.ims_vgg_layer_2, self.ims_NIC_layer), center=True, scale=True, is_training=self.gen_phase, scope='gen_ims_layer_bn_1')
		self.gen_ims_layer_2 = tf.nn.l2_normalize(self.gen_ims_layer_2, 1)
		#self.gen_ims_layer_2 = tf.nn.l2_normalize(tf.add(self.ims_vgg_layer_2, self.ims_NIC_layer), 1)

		# domain adaptation loss branch
		self.dis_data_merge = tf.concat([self.gen_ims_layer_2, self.gen_ls_layer_2], 0)
		self.dis_merge_layer_1 = tf.nn.xw_plus_b(self.dis_data_merge, self.W_merge_1, self.b_merge_1)
		self.dis_merge_layer_1 = tf.nn.dropout(tf.nn.relu(self.dis_merge_layer_1), self.dis_keep_prob)
		self.dis_merge_layer_2 = tf.nn.xw_plus_b(self.dis_merge_layer_1, self.W_merge_2, self.b_merge_2)
		self.dis_merge_layer_2 = tf.nn.dropout(tf.nn.relu(self.dis_merge_layer_2), self.dis_keep_prob)
		self.dis_merge_layer_3 = tf.nn.xw_plus_b(self.dis_merge_layer_2, self.W_merge_3, self.b_merge_3)

		with tf.name_scope('domain_loss'):
			self.loss_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pred_data_label, logits=self.dis_merge_layer_3))
			self.weight_decay_D = self.weight_decay * (tf.nn.l2_loss(self.W_merge_1) + tf.nn.l2_loss(self.b_merge_1) + tf.nn.l2_loss(self.W_merge_2) \
								  + tf.nn.l2_loss(self.b_merge_2) + tf.nn.l2_loss(self.W_merge_3) + tf.nn.l2_loss(self.b_merge_3))
			self.domain_loss = self.loss_D + self.weight_decay_D

		#embedding loss branch
		self.pred_score = tf.matmul(self.gen_ims_layer_2, self.gen_ls_layer_2, transpose_b=True)
		self.diagonal = tf.diag_part(self.pred_score)
		self.cost_s = tf.maximum(0., self.margin - self.diagonal + self.pred_score)
		self.cost_im = tf.maximum(0., self.margin - tf.reshape(self.diagonal, [-1, 1]) + self.pred_score)
		self.cost_s = tf.multiply(self.cost_s, (tf.ones([tf.shape(self.vgg_pred_data)[0], tf.shape(self.vgg_pred_data)[0]]) - tf.eye(tf.shape(self.vgg_pred_data)[0])))
		self.cost_im = tf.multiply(self.cost_im, (tf.ones([tf.shape(self.vgg_pred_data)[0], tf.shape(self.vgg_pred_data)[0]]) - tf.eye(tf.shape(self.vgg_pred_data)[0])))

                #self.cost_s = tf.reduce_max(self.cost_s, 1)
                #self.cost_im = tf.reduce_max(self.cost_im, 0)

		with tf.name_scope('embedding_loss'):
			self.loss_E = tf.reduce_sum(self.cost_s) + tf.reduce_sum(self.cost_im)
			self.weight_decay_E = self.weight_decay * (tf.nn.l2_loss(self.W_ims_1) + tf.nn.l2_loss(self.b_ims_1) + tf.nn.l2_loss(self.W_ims_3) + tf.nn.l2_loss(self.b_ims_3))
			self.embedding_loss = self.loss_E + self.weight_decay_E - self.lamb * self.loss_D

		self.tvars = tf.trainable_variables()
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(self.update_ops):
			# domain classifier optimize operation
			self.optimizer_d = tf.train.AdamOptimizer(self.learning_rate)
			self.d_updates = self.optimizer_d.minimize(self.domain_loss, var_list=self.d_params)	  #domain adaptor update

			self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.embedding_loss, self.tvars), 2.0)
			self.optimizer_g = tf.train.AdamOptimizer(self.learning_rate)
			self.g_updates_E = self.optimizer_g.apply_gradients(zip(self.grads, self.tvars))

			#self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate / 4)
			#self.g_updates_D = self.optimizer.minimize(self.loss_D, var_list=self.tvars)	   #domain adaptor update

		self.correct_pred = tf.equal(tf.argmax(self.pred_data_label, 1), tf.argmax(self.dis_merge_layer_3, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
	def save_model(self, sess, filename):
		param = sess.run(self.params_total)
		cPickle.dump(param, open(filename, 'w'))

