import cv2
import numpy as np
from angle_feature import angle
from sign_extract import extract
import os
from os import listdir
from os.path import isfile, join

weights = np.load("weights_for_ANN.npy")
bias = np.load("bias_for_ANN.npy")

images = [f for f in listdir("./Testing/") if isfile(join("./Testing/", f))]

for i in images:
	extract("./Testing/"+i, "./Testing/testmodfd.jpeg")
	yin = np.dot(angle("./Testing/testmodfd.jpeg"),weights) + bias
	#print(yin)
	img = cv2.imread("./Testing/"+i, 0)
	if yin > 0:
		print("\n", i + " : This may be a signature:-)")
	else:
		print("\n", i + " : Not a signature:-(")
	os.remove("./Testing/testmodfd.jpeg")
	cv2.imshow("Test", img)
	cv2.waitKey(0)
		
print()