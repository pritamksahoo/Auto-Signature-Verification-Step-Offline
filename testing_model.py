from math import exp
import cv2
import numpy as np
from angle_feature import angle
from sign_extract import extract
import os
from os import listdir
from os.path import isfile, join

def activate(weights, inputs):
	'''Calculating input for hidden layers'''
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer(activation):
	'''Evaluating Activating ( Sigmoid ) Function'''
	return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, row):
	'''Calculating Output of the Neural Network'''
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs) 
			new_inputs.append(transfer(activation))
		inputs = new_inputs
	return inputs

def predict(network, row):
	'''Making Prediction'''
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Taking Test Images 
images = [f for f in listdir("./Testing/") if isfile(join("./Testing/", f))]
network = list(np.load("weights_after_training.npy"))

for i in images:
	print("\n" + str(i))
	extract("./Testing/"+i, "./Testing/testmodfd.jpeg")

	dataset = list(angle("./Testing/testmodfd.jpeg"))

	img = cv2.imread("./Testing/"+i, 0)
	pre = int(input("Enter prediction (1 for yes / 0 for no) : "))
	dataset.append(pre)
	'''Main Function Call'''
	prediction = predict(network, dataset)
	
	print('Expected=%d, Got=%d' % (pre, prediction))

	os.remove("./Testing/testmodfd.jpeg")