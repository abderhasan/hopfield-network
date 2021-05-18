'''
Author: Dr. Abder-Rahman Ali
Email: abder.rahman.ali@gmail.com

This code is an Implementation of the Hopfield network. It is applied on a 
single pattern. 
'''

import numpy as np
import matplotlib.pyplot as plt

# A pattern
pattern = [0,1,1,1,0,0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,1,0]

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
			v_i = 0
			for j in range(len(x)):
				v_i = v_i + w[j][i] * x[j]

			if v_i >=0:
				x[i] = 1
			else:
				x[i] = 0

			if x[i] == x_copy[i]:
				m = m + 1
				
			if m == 25:
				index = 1
	
	print('This is the final stable state:')
	print(x)
	
	pattern_reshape = np.reshape(x,(5,5))

weights_matrix = find_weights_matrix(pattern)

start_state = [0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,0,0,1,1,1,0]
start_state_copy = start_state.copy()

new_vector = update_nodes(start_state,start_state_copy,weights_matrix)