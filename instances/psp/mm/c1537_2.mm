************************************************************************
file with basedata            : c1537_.bas
initial value random generator: 460063472
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  125
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       28       15       28
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   6
   3        3          2           7  13
   4        3          1          16
   5        3          3          10  11  13
   6        3          2           8   9
   7        3          2           9  14
   8        3          1          11
   9        3          1          15
  10        3          1          17
  11        3          3          12  15  16
  12        3          1          17
  13        3          1          14
  14        3          1          17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     7       3    6    9    7
         2    10       1    5    4    6
         3    10       2    5    6    4
  3      1     3       6    8    6    7
         2     3       6    7   10    9
         3     6       4    5    2    7
  4      1     1       4    5    3    4
         2     3       3    4    2    3
         3     9       3    2    2    2
  5      1     6       9    6    8    1
         2     9       6    4    5    1
         3    10       6    2    4    1
  6      1     3       5    9    7    7
         2     6       4    8    6    6
         3    10       4    7    6    6
  7      1     3      10    6   10   10
         2     5       6    4   10    5
         3     5       8    5    9    7
  8      1     5       3   10    9    5
         2     6       3    9    5    3
         3    10       2    7    2    2
  9      1     1       8    9    6    9
         2     5       7    9    6    7
         3     8       7    9    5    6
 10      1     1       7    4    7    7
         2     4       7    3    7    5
         3     5       4    2    5    5
 11      1     1       7    4    8    9
         2     6       6    4    8    9
         3     7       4    4    6    9
 12      1     2      10    5    5    6
         2     3      10    5    3    5
         3     6       9    5    3    4
 13      1     5      10    8    9    6
         2     5       9    7    9    7
         3     6       6    7    8    4
 14      1     5       6    5   10    9
         2     8       6    5    9    9
         3     9       3    3    9    8
 15      1     7       3    8    9    8
         2     7       2    6   10    6
         3     9       2    2    5    5
 16      1     6       9    5    6    5
         2     7       5    4    4    5
         3     8       4    4    3    5
 17      1     5      10    4    6    7
         2     6       9    4    5    6
         3     7       9    4    2    3
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   13   87   85
************************************************************************
