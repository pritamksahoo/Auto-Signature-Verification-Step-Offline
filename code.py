import numpy as np
from angle_feature import angle
from perceptron import perceptron_sgd
from sign_extract import extract
from os import listdir
from os.path import isfile, join
import os

print("\nExtracting signatures......\n")

trainimages1 = [f for f in listdir("./Training/YES/") if isfile(join("./Training/YES/", f))]
count=1
for i in trainimages1:
	name, ext = i.split(".")
	print(i)
	extract("./Training/YES/"+i, "./Training/MODFD/modfd"+str(count)+".jpeg")
	count = count+1

print()

trainimages2 = [f for f in listdir("./Training/NO/") if isfile(join("./Training/NO/", f))]
for i in trainimages2:
	name, ext = i.split(".")
	print(i)
	extract("./Training/NO/"+i, "./Training/MODFD/modfd"+str(count)+".jpeg")
	count = count+1

print("\nExtracting angle features......", end="")

modfdimages = [f for f in listdir("./Training/MODFD") if isfile(join("./Training/MODFD", f))]
no_of_images = len(modfdimages)

input_to_ANN = []
for i in range(1,len(trainimages1)+1):
	t = list(angle("./Training/MODFD/modfd"+str(i)+".jpeg"))
	t.append(1)
	input_to_ANN.append(t)

for i in range(1, len(trainimages2)+1):
	t = list(angle("./Training/MODFD/modfd"+str(len(trainimages1)+i)+".jpeg"))
	t.append(0)
	input_to_ANN.append(t)

print("Done")
np.save('data_for_ANN', input_to_ANN)


# print(input_to_ANN[:2])
# input_to_ANN = np.array([angle("./Training/MODFD/modfd"+str(i)+".jpeg") for i in range(1,no_of_images+1)])
# output = np.append(np.array([1]*len(trainimages1)), np.array([-1]*len(trainimages2)))
# print(input_to_ANN)
# print(output)
# print(output)

# print("Done\n\nTraining perceptron model......\n")

# w, b = perceptron_sgd(input_to_ANN, output)
# print("Bias :",b)
# print("\nWeights : \n",w,"\n")

# np.save('weights_for_ANN', w)
# np.save('bias_for_ANN', b)

# for i in modfdimages:
# 	os.remove("./Training/MODFD/"+i)