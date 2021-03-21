'''
Author: Dr. Abder-Rahman Ali
Email: abder.rahman.ali@gmail.com

This code is an Implementation of the Hopfield network. It is applied on
an image.
'''

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('/path-to-image/cat.png')
image = image.resize((100,100)) 
pattern = list(image.getdata())

def find_weights_matrix(x):
	w = np.zeros([len(x),len(x)])
	for i in range(len(x)):
		for j in range(i,len(x)):
			if i == j:
				w[i,j] = 0
			else:
				w[i,j] = (2*x[i]-1)*(2*x[j]-1)
				w[j,i] = w[i,j]
	return w

def update_nodes(x,x_copy,w):
	m = 0
	index = 0
	while index < 1:
		for i in range(len(pattern)):
			w_i = 0
			for j in range(len(x)):
				w_i = w_i + w[j][i] * x[j]

			if w_i >=0:
				x[i] = 1
			else:
				x[i] = 0

			if x[i] == x_copy[i]:
				m = m + 1
				
			if m == len(pattern):
				index = 1
	
	pattern_reshape = np.reshape(x,(100,100))
	pattern_reshape = 255 - pattern_reshape
	plt.imsave('result.png',pattern_reshape,cmap=plt.cm.binary)

weights_matrix = find_weights_matrix(pattern)

blurred_image = Image.open('/path-to-image/blurred_image.png')
blurred_image = image.resize((100,100)) 
blurred_image_pattern = list(image.getdata())
start_state = blurred_image_pattern
start_state_copy = start_state.copy()

new_vector = update_nodes(start_state,start_state_copy,weights_matrix)