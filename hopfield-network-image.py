'''
Author: Dr. Abder-Rahman Ali
Email: abder.rahman.ali@gmail.com
This code is an implementation of the Hopfield network. It is applied on
an image.
'''
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time

image = Image.open('/Users/abder/Desktop/machine-learning-code/cat.png')

# convert matrix to a vector
def matrix_to_vector(x):
    m = x.shape[0]*x.shape[1]
    temp = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            temp[c] = x[i,j]
            c +=1
    return temp

image = np.asarray(image,dtype=np.uint8) 
pattern = matrix_to_vector(image)

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
	vector_length = len(x)
	m = 0
	index = 0
	while index < 1:
		if m == vector_length:
			index = 1
		else:
			m = 0
			for i in range(len(pattern)):
				v_i = 0
				for j in range(len(x)):
					v_i = v_i + w[j][i] * x[j]

				if v_i >= 0:
					x[i] = 255
				else:
					x[i] = 0

				if x[i] == x_copy[i]:
					m = m + 1
				else:
					x_copy[i] = x[i]
					m = 0
					
	pattern_reshape = np.reshape(x,(100,100))
	pattern_reshape = 255 - pattern_reshape
	plt.imsave('result.png',pattern_reshape,cmap=plt.cm.binary)

weights_matrix = find_weights_matrix(pattern)

blurred_image = Image.open('/Users/abder/Desktop/machine-learning-code/blurred_image.png')
blurred_image = np.asarray(image,dtype=np.uint8) 
blurred_image_pattern = matrix_to_vector(blurred_image)
start_state = blurred_image_pattern
start_state_copy = start_state.copy()

new_vector = update_nodes(start_state,start_state_copy,weights_matrix)
