************************************************************************
file with basedata            : c1550_.bas
initial value random generator: 29301
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  124
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       23       15       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           7  16
   3        3          2          11  12
   4        3          1           5
   5        3          2           6  14
   6        3          3           8   9  10
   7        3          2           9  10
   8        3          2          11  16
   9        3          2          12  17
  10        3          1          11
  11        3          1          13
  12        3          1          13
  13        3          1          15
  14        3          1          15
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       6    0    7    9
         2     5       0    1    4    9
         3     5       3    0    2    7
  3      1     2       0    5    6    6
         2     5       4    0    6    5
         3    10       3    0    6    1
  4      1     4       8    0   10   10
         2     6       5    0    9   10
         3     8       0    9    6   10
  5      1     2       0    7    7    9
         2     6       0    4    6    9
         3     8       6    0    5    8
  6      1     6       0    2    8    5
         2     6       4    0    9    5
         3     8       3    0    6    3
  7      1     2       5    0    5    8
         2     5       4    0    3    6
         3     6       2    0    2    4
  8      1     2       0    6    5    9
         2     2       4    0    5    8
         3     8       0    6    4    8
  9      1     6       8    0    9    6
         2     6       0    8    9    6
         3    10       0    6    9    2
 10      1     5       3    0    5    2
         2     8       0    6    3    2
         3     8       1    0    2    2
 11      1     1       4    0    9    8
         2     2       0    3    9    7
         3     8       3    0    8    6
 12      1     3       1    0    3    9
         2     7       0    7    2    8
         3     9       0    6    2    7
 13      1     1       0    4    8    3
         2     1       9    0   10    2
         3     1       0    3   10    3
 14      1     3       1    0    4    1
         2     5       0    6    3    1
         3     9       1    0    2    1
 15      1     1       3    0    8    5
         2     3       1    0    5    5
         3     6       0    4    5    5
 16      1     1       7    0    7    9
         2     7       0    9    4    7
         3    10       0    9    2    6
 17      1     5       0    4    6   10
         2     6       9    0    4    8
         3    10       9    0    2    4
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10   13  100  101
************************************************************************
