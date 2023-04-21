************************************************************************
file with basedata            : c1544_.bas
initial value random generator: 2072430910
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  142
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       28        7       28
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           7   9
   3        3          1           5
   4        3          2          10  17
   5        3          3           6   8  11
   6        3          1           9
   7        3          3          14  16  17
   8        3          1           9
   9        3          2          10  12
  10        3          1          15
  11        3          1          13
  12        3          2          16  17
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
  2      1     6      10    0   10    7
         2    10       0    2   10    7
         3    10       9    0   10    6
  3      1     5       3    0    6    4
         2     5       0    7    6    4
         3     7       4    0    6    3
  4      1     1       0    2    7    6
         2     8       0    2    3    3
         3     8       0    1    3    4
  5      1     2       9    0    4    7
         2     3       9    0    3    7
         3    10       0    8    2    4
  6      1     1       0    2    3    5
         2     5      10    0    3    3
         3    10       8    0    1    3
  7      1     2       4    0    6   10
         2     7       3    0    5    9
         3     8       3    0    2    9
  8      1     4       7    0    7    7
         2     5       6    0    6    6
         3    10       0    7    6    3
  9      1     8       0    4    9    3
         2    10       7    0    8    1
         3    10       0    3    7    1
 10      1     5       0    5    6    8
         2     8       0    4    5    5
         3     9       0    4    5    4
 11      1     6       0    8    8    4
         2     7       0    7    8    4
         3     8       0    7    6    4
 12      1     6       8    0    7    8
         2     6       0    9    3    6
         3     6       0    9    2    9
 13      1     3       6    0    8    9
         2     5       0    8    8    6
         3    10       6    0    5    5
 14      1     9       8    0    6    4
         2     9       0    7    6    5
         3     9       0    9    6    4
 15      1     4       0    3    7    5
         2     5       3    0    7    3
         3     9       0    1    7    3
 16      1     1       4    0    9    6
         2     6       0    6    8    4
         3     8       0    4    8    1
 17      1     1       7    0    2    3
         2     1       0    9    2    3
         3    10       6    0    1    3
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   25   26   91   80
************************************************************************
