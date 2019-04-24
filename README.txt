Muhammed Furkan YAĞBASAN
2099505
CENG499 THE2
21.04.2019
---------------------------------------------------------------------------
<PART 1>

part1.py -> plots the graph according to given parameters
Usage:
python part1.py <dateset_file> <Criterion type index> <# of final clusters>
Criterion type indices
1: Single-Linkage
2: Complete-Linkage
3: Average-Linkage
4: Centroid

part1Images folder -> contains outputs of part1.py for 4 data sets.


---------------------------------------------------------------------------
<PART 2>

part2.py -> generates an decision tree according to given parameters
Usage:
python part2.py <trainingset_file> <attribute selection strategy index> <testset_file>
attribute selection strategy indices
1: Information Gain
2: Gain Ratio
3: Average Gini Index
4: Gain Ratio with Chi-squared Pre-pruning
5: Gain Ratio with Reduced error post-pruning

(giving a testset_file is optional)

part2trees folder -> contains output tree representations and test accuracy results of part2.py

In tree representations every node contains "[a,b,c,d]" that represents number of classes.
( a=> # of unacc
  b=> # of acc
  c=> # of good
  d=> # of vgood )
