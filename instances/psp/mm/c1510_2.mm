************************************************************************
file with basedata            : c1510_.bas
initial value random generator: 227300182
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
    1     16      0       15        4       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           8
   3        3          3           9  14  15
   4        3          3           5   6   7
   5        3          1          15
   6        3          1          10
   7        3          2          12  14
   8        3          1          14
   9        3          1          11
  10        3          2          13  17
  11        3          2          16  17
  12        3          1          16
  13        3          1          15
  14        3          2          16  17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     4       0    8    7    0
         2     7       0    3    4    0
         3     7       0    6    0    8
  3      1     1       0    7    8    0
         2     5       0    7    7    0
         3    10       3    0    0    7
  4      1     2       4    0    0    3
         2     2       4    0    4    0
         3     4       0    3    2    0
  5      1     6       9    0    5    0
         2     8       6    0    0    8
         3    10       0    7    0    5
  6      1     1       0    5    8    0
         2     3       0    4    4    0
         3     6       8    0    0    6
  7      1     2       7    0    0    5
         2     6       4    0    0    2
         3     8       3    0    6    0
  8      1     2       0    9    0    8
         2     3       9    0    1    0
         3     7       0    6    1    0
  9      1     2       0    3    6    0
         2     4       0    3    0    8
         3     7       0    2    0    6
 10      1     4       3    0    0    6
         2     7       0    8    0    6
         3     9       0    3    0    4
 11      1     5       6    0    6    0
         2     5       0    9    0    4
         3     7       0    8    0    1
 12      1     1       0    6    0    7
         2     4       0    6    0    6
         3    10       0    6    0    5
 13      1     1       0    8    7    0
         2     2       3    0    0    3
         3     9       3    0    0    2
 14      1     4       0   10    0    6
         2     8       0    9    8    0
         3    10       0    8    7    0
 15      1     6       6    0   10    0
         2     7       0    4    0    1
         3     8       5    0    6    0
 16      1     5       0   10    0    9
         2     6      10    0    0    9
         3     7       0   10    0    7
 17      1     3       0    7    0    4
         2     6       3    0    0    3
         3     6       0    7    4    0
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   24   40   55
************************************************************************
