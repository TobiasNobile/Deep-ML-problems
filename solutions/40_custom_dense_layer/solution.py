
import numpy as np
import copy
import math

np.random.seed(42)

class Layer(object):

	def set_input_shape(self, shape):
    
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None
	
	def initialize(self, optimizer):
		self.W = np.random.uniform(-1/np.sqrt(self.input_shape[0]), 1/np.sqrt(self.input_shape[0]), size = (self.input_shape[0], self.n_units))
		self.w0 = np.zeros(self.n_units)

		self.W_opt = copy.copy(optimizer)
		self.w0_opt = copy.copy(optimizer)

	def forward_pass(self, X, training=True):
		self.layer_input = X
		return self.layer_input @ self.W + self.w0

	def backward_pass(self, accum_grad):
		grad_input = accum_grad @ self.W.T
		grad_w = self.layer_input.T @ accum_grad
		grad_b = np.sum(accum_grad, axis=0)
		if self.trainable:
			self.W = self.W_opt.update(self.W, grad_w)
			self.w0 = self.w0_opt.update(self.w0, grad_b)
		return grad_input

	def parameters(self):
		return np.prod(self.W.shape) + np.prod(self.w0.shape)
	
	def output_shape(self):
		return ((self.n_units,))

    