####################################################
# 
# Muhammed Furkan YAGBASAN - 2099505
# 20.04.2019
# Ceng499 - THE2
# PART 2
#
####################################################
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import log
import random # to generate validation set randomly from trainig set

if(len(sys.argv)!=3 and len(sys.argv)!=4):
	print("usage: python part2.py <trainingset_file> <attribute selection strategy index> <testset_file>" +
		"\n attribute selection strategy indices" +
		"\n1: Information Gain" +
		"\n2: Gain Ratio" +
		"\n3: Average Gini Index" +
		"\n4: Gain Ratio with Chi-squared Pre-pruning" +
		"\n5: Gain Ratio with Reduced error post-pruning\n")
	sys.exit(1)

attributeValues = [["vhigh","high","med","low"],
["vhigh","high","med","low"],
["2","3","4","5more"],
["2","4","more"],
["small","med","big"],
["low","med","high"]]

classes = ["unacc","acc","good","vgood"]
attributeNames = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# get training set
examples = []
with open(sys.argv[1]) as fp:
	cnt = 0
	for line in fp:
		e = []
		attributes = line.strip().split(' ')
		e.append(attributes[0])
		e.append(attributes[1])
		e.append(attributes[2])
		e.append(attributes[3])
		e.append(attributes[4])
		e.append(attributes[5])
		e.append(attributes[6])

		examples.append(e)

validationSet = []
if(int(sys.argv[2])==5):
	validationSetSize = int(len(examples)/20)
	for i in range(validationSetSize):
		randx = random.randint(0,len(examples)-1)
		validationSet.append(examples.pop(randx))

# get test set
testExamples = []
if(len(sys.argv)==4):
	with open(sys.argv[3]) as fp:
		cnt = 0
		for line in fp:
			e = []
			attributes = line.strip().split(' ')
			e.append(attributes[0])
			e.append(attributes[1])
			e.append(attributes[2])
			e.append(attributes[3])
			e.append(attributes[4])
			e.append(attributes[5])
			e.append(attributes[6])

			testExamples.append(e)

class TreeNode:
	def __init__(self, attribute, classDist):
		self.attribute= attribute
		self.classDist = classDist
		self.children = []

	def addChild(self, child):
		self.children.append(child)


def searchInList(list, attribute, value):
	result = []
	for e in list:
		if(e[attribute]==value):
			result.append(e)
	return result

# returns the distribution of classes 
# among the set that has given attribute and value
def select(list, attribute, value):
	result = [0,0,0,0]

	for e in list:
		if(e[attribute]==value):
			for i,c in enumerate(classes):
				if(e[-1]==c):
					result[i]+=1
					break
	return result

# returns the distribution of classes
def select2(list):
	result = [0,0,0,0]

	for e in list:
		for i,c in enumerate(classes):
			if(e[-1]==c):
				result[i]+=1
				break

	return result

# returns the distribution set that has given attribute
def select_attribute(list, attribute):
	result = [0] * len(attributeValues[attribute])
	for e in list:
		for i,value in enumerate(attributeValues[attribute]):
			if(e[attribute]==value):
				result[i]+=1
	return result

# retruns entropy according to given distribution set
def entropy(list):
	result = 0
	S = 0
	for l in list:
		S += l
	if(S==0):
		return 1
	for l in list:
		p = l/S
		if(p!=0):
			result += (-1)*p*log(p, len(list))
	return result

def average_entropy(list, attribute):
	result = 0
	distribution = select_attribute(list, attribute)
	totalsum = 0
	for num in distribution:
		totalsum += num
	for i,d in enumerate(distribution):
		result += (d/totalsum)*entropy(select(list ,attribute, attributeValues[attribute][i]))
	return result

def info_gain(list, attribute):
	return entropy(select2(list))-average_entropy(list,attribute)

def intrinsic_info(list, attribute):
	result = 0
	distribution = select_attribute(list, attribute)
	totalsum = 0
	for num in distribution:
		totalsum += num
	for d in distribution:
		if(d!=0):
			result -= (d/totalsum)*log((d/totalsum),2)
	return result

def gain_ratio(list, attribute):
	return info_gain(list, attribute)/intrinsic_info(list, attribute)

def gini(list):
	result = 1
	S = 0
	for l in list:
		S += l
	if(S==0):
		return 1
	for l in list:
		p = l/S
		if(p!=0):
			result -= p*p
	return result

def gini_index(list, attribute):
	result = 0
	distribution = select_attribute(list, attribute)
	totalsum = 0
	for num in distribution:
		totalsum += num
	for i,d in enumerate(distribution):
		result += (d/totalsum)*gini(select(list ,attribute, attributeValues[attribute][i]))
	return result

def distributionResult(list):
	for i in range(len(classes)):
		if(list[i]!=0):
			return i
	return -1

def chi_test(list, attribute):
	currentDist = select2(list)
	subDists = []
	for value in attributeValues[attribute]:
		subDists.append(select(list, attribute, value))

	result = 0

	sumCurrentDist = sum(currentDist)
	for childDist in subDists:
		sumChildDist = sum(childDist)
		p = (sumChildDist/sumCurrentDist)
		for i in range(4):
			expV = currentDist[i]*p
			if(expV==0):
				continue
			result += ((expV - childDist[i])**2)/expV

	if(result>6.251):
		return 1
	else:
		return 0

def post_pruning(root):

	if(root.attribute == -1):
		return testDataSet(validationSet)

	scoresOfChildren = []
	for child in root.children:
		scoresOfChildren.append(post_pruning(child))

	holdAttribute = root.attribute
	root.attribute = -1

	selfscore = testDataSet(validationSet)

	if(selfscore<max(scoresOfChildren)):
		root.attribute = holdAttribute
		return max(scoresOfChildren)
	else:
		return selfscore

def chooseStrategy(list, a):
	idx = int(sys.argv[2])
	if(idx == 1):
		return info_gain(list, a)
	elif(idx == 2):
		return gain_ratio(list, a)
	elif(idx == 3):
		return gini_index(list, a)
	elif(idx == 4):
		return gain_ratio(list, a)
	elif(idx == 5):
		return gain_ratio(list, a)

# Tree Construction
def constructTree(list, passedAttributes):
	if(len(list)==0):
		node = TreeNode(-1, [0,0,0,0])
		return node

	distribution = select2(list)

	if(entropy(distribution)==0):
		node = TreeNode(-1, distribution)
		return node

	if(len(passedAttributes)==6):
		node = TreeNode(-1, distribution)
		return node

	minNum = float("Inf")
	maxNum = -1
	bestAttribute = -1
	for a in range(6):
		if(a in passedAttributes):
			continue
		strategyScore = chooseStrategy(list,a)
		if(int(sys.argv[2])==3):
			if(strategyScore<minNum):
				minNum = strategyScore
				bestAttribute = a
		else:
			if(strategyScore>maxNum):
				maxNum = strategyScore
				bestAttribute = a

	if(int(sys.argv[2])==4):
		if(chi_test(list, bestAttribute)==0):
			node = TreeNode(-1, distribution)
			return node

	node = TreeNode(bestAttribute, distribution)

	newPassedAttributes = []
	for i in passedAttributes:
		newPassedAttributes.append(i)
	if(bestAttribute not in newPassedAttributes):
		newPassedAttributes.append(bestAttribute)

	for value in attributeValues[bestAttribute]:
		node.addChild(constructTree(searchInList(list, bestAttribute, value), newPassedAttributes))

	return node

def printTree(root, depth, lineList):
	if(root.attribute==-1):
		print("LEAF", end='')
		print(root.classDist)
		return
	else:
		print(attributeNames[root.attribute], end='')
	print(root.classDist)

	for i,c in enumerate(root.children):
		if(i==len(root.children)-1):
			lineList[depth] = 0
		else:
			lineList[depth] = 1
		for t in range(depth):
			if(lineList[t]!=0):
				print("|", end='')
			print("\t", end='')
		print("|")
		for t in range(depth):
			if(lineList[t]!=0):
				print("|", end='')
			print("\t", end='')
		print("|",end='')
		print("->",end='')
		print("(" + attributeValues[root.attribute][i] + ")" ,end="->")
		printTree(c, depth+1,lineList)

decisionTreeRoot = constructTree(examples, [])
printTree(decisionTreeRoot,0,[1,1,1,1,1,1])

######################################################
## Testing functions - START
######################################################
def getMaxFromDist(distribution):
	maxNum = -1
	result = -1
	for i,d in enumerate(distribution):
		if(d>maxNum):
			maxNum=d
			result = i
	if(maxNum==0):
		return -1
	return result

def testData(e, root):
	if(root.attribute==-1):
		output = getMaxFromDist(root.classDist)
		if(output!=-1):
			if(e[-1]==classes[output]):
				return 1
		return 0

	else:
		idx = attributeValues[root.attribute].index(e[root.attribute])
		return testData(e, root.children[idx])
		
def testDataSet(testSet):
	cnttest = 0
	for e in testSet:
		cnttest += testData(e, decisionTreeRoot)
	return cnttest

######################################################
## Testing functions - END
######################################################

if(len(sys.argv)==4):
	testAccuracy = (testDataSet(testExamples)/len(testExamples))
	if(int(sys.argv[2])!=5):
		print("\nTest Accuracy: %lf\n" %(testAccuracy))
	else:
		post_pruning(decisionTreeRoot)
		print("\nPruned Tree:\n")
		printTree(decisionTreeRoot,0,[1,1,1,1,1,1])
		print("\nTest Accuracy: %lf" %(testAccuracy))
		print("\nTest Accuracy: %lf\n" %(testDataSet(testExamples)/len(testExamples)))
else:
	if(int(sys.argv[2])==5):
		post_pruning(decisionTreeRoot)
		print("\nPruned Tree:\n")
		printTree(decisionTreeRoot,0,[1,1,1,1,1,1])