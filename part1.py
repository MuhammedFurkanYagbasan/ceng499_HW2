####################################################
# 
# Muhammed Furkan YAGBASAN - 2099505
# 21.04.2019
# Ceng499 - THE2
# PART 1
#
####################################################
import numpy as np
import matplotlib.pyplot as plt
import sys

# Usage
if(len(sys.argv)!=4):
	print("usage: python part1.py <dateset_file> <Criterion type index> <# of final clusters>" +
		"\n Criterion type indices" +
		"\n1: Single-Linkage" +
		"\n2: Complete-Linkage" +
		"\n3: Average-Linkage" +
		"\n4: Centroid\n")
	sys.exit(1)

###############################################################
## distance functions -START
###############################################################
def single_linkage_dist(list1, list2):
	min_dist = float("Inf")
	for l1 in list1:
		for l2 in list2:
			dist = ((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)
			if(dist < min_dist):
				min_dist = dist
	return min_dist

def complete_linkage_dist(list1, list2):
	max_dist = -1
	for l1 in list1:
		for l2 in list2:
			dist = ((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)
			if(dist > max_dist):
				max_dist = dist
	return max_dist

def average_linkage_dist(list1, list2):
	dist = 0
	for l1 in list1:
		for l2 in list2:
			dist += ((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)
	dist /= (len(list1)*len(list2))
	return dist

def centroid_dist(list1, list2):
	xsum1 = 0
	ysum1 = 0
	xsum2 = 0
	ysum2 = 0
	for l1 in list1:
		xsum1 += l1[0]
		ysum1 += l1[1]
	for l2 in list2:
		xsum2 += l2[0]
		ysum2 += l2[1]

	xsum1 /= len(list1)
	ysum1 /= len(list1)
	xsum2 /= len(list2)
	ysum2 /= len(list2)

	dist = ((xsum1 - xsum2)**2 + (ysum1 - ysum2)**2)

	return dist

###############################################################
## distance functions - END
###############################################################

# general dist function
def dist(l1, l2):
	idx = int(sys.argv[2])
	if(idx==1):
		return single_linkage_dist(l1, l2)
	elif(idx==2):
		return complete_linkage_dist(l1,l2)
	elif(idx==3):
		return average_linkage_dist(l1, l2)
	elif(idx==4):
		return centroid_dist(l1, l2)
	else:
		print("Criterion type index is wrong!")
		sys.exit(1)

def minDist(list):
	min_dist = float("Inf")
	cnt1 = 0
	p1idx = 0
	p2idx = 0
	for l1 in list:
		cnt2 = cnt1+1 
		for l2 in list[(cnt1+1):]:
			result = dist(l1, l2)
			if(result < min_dist):
				min_dist = result
				p1idx = cnt1
				p2idx = cnt2
			cnt2+=1
		cnt1+=1
	return [p1idx, p2idx]

activeSet = []

# Read Data
with open(sys.argv[1]) as fp:
	cnt = 0
	for line in fp:
		coords = line.strip().split(' ')
		x = round(float(coords[0]),5)
		y = round(float(coords[1]),5)
		activeSet.append([[x,y]])

# run the algorithm
while(len(activeSet)>int(sys.argv[3])):
	closest_pnts = minDist(activeSet)
	union_pnts = activeSet[closest_pnts[0]]
	union_pnts.extend(activeSet[closest_pnts[1]])

	del activeSet[closest_pnts[0]]
	if(closest_pnts[0]<closest_pnts[1]):
		del activeSet[closest_pnts[1]-1]
	else:
		del activeSet[closest_pnts[1]]
	activeSet.append(union_pnts)

# plot the graph
for i in range(int(sys.argv[3])):
	xs = np.zeros((len(activeSet[i]),))
	ys = np.zeros((len(activeSet[i]),))
	cnt=0
	for f in activeSet[i]:
		xs[cnt] = f[0]
		ys[cnt] = f[1]
		cnt+=1
	plt.plot(xs, ys, 'o')

plt.show()
