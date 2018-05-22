from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig:

	def __init__(self):
		"""sets the default model hyperparameters."""

		# Name of the training dataset
		self.data = None
		self.worddict = None
		self.n_words = None

		# margin used in pairwise ranking loss
		self.margin = 0.3
		# Dimensions for the learned limited text space
		self.dim = 1024
		# Dimensions for the VGG feature
		self.dim_VGG = 1536
		# Dimensions for the NIC feature
		self.dim_NIC = 512
		# Dimensions for the wording embedding
		self.dim_word = 1024
		# Output dimensions for image embedding FC layers
		self.hidden_size = [2048, 1024]


		# training epochs
		self.max_epochs = 30
		# weight decay term
		self.weight_decay = 0.0
		self.maxlen_w = 100
		self.batch_size = 128
		self.lrate = 0.0002

		self.clip_gradients = 2.0

		self.max_checkpoints_to_keep = 5
