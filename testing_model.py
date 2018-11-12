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
	# print(outputs)
	return outputs.index(max(outputs)),max(outputs)

# Taking Test Images 
images = [f for f in listdir("./Testing/") if isfile(join("./Testing/", f))]
network = list(np.load("weights_after_training.npy"))
# print(network)
for i in images:
	print("\n" + str(i))
	extract("./Testing/"+i, "./Testing/testmodfd.jpeg")

	dataset = list(angle("./Testing/testmodfd.jpeg"))

	# img = cv2.imread("./Testing/"+i, 0)
	# cv2.imshow("Test", img)
	# cv2.waitKey(0)
	# pre = int(input("Enter prediction (1 for yes / 0 for no) : "))
	# dataset.append(pre)
	'''Main Function Call'''
	prediction, match = predict(network, dataset)
	if prediction == 11:
		print("!!! Not a signature !!!")
	else:
		if match <= 0.7:
			print("### Does not matches with any signature ###")
		else:
			print("Matches with img%.2d.jpg" % (prediction))
	# print(prediction)
	# print('Expected=%d, Got=%d' % (pre, prediction))
	print()
	os.remove("./Testing/testmodfd.jpeg")