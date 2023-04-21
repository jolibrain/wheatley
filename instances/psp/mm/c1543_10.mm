************************************************************************
file with basedata            : c1543_.bas
initial value random generator: 1858046924
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  123
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       25       10       25
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           6
   3        3          1           7
   4        3          3           5   6   8
   5        3          2          10  11
   6        3          1          13
   7        3          2           9  17
   8        3          1          14
   9        3          3          10  12  15
  10        3          1          16
  11        3          3          12  15  17
  12        3          1          13
  13        3          1          14
  14        3          1          16
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       8    0    4   10
         2     4       0    8    2    7
         3    10       0    8    2    4
  3      1     3       0    8    4    6
         2     4       0    8    2    5
         3     7       0    7    2    1
  4      1     3       4    0    8    4
         2     7       0    4    5    4
         3     9       4    0    5    3
  5      1     7       8    0    5    8
         2     7       8    0    6    7
         3     9       8    0    1    4
  6      1     1       0    9    5    7
         2     2       0    7    3    7
         3     4       0    3    3    7
  7      1     2       0    8   10   10
         2     4       1    0    5    8
         3     5       0    4    2    7
  8      1     2       9    0    9    9
         2     6       8    0    6    7
         3     7       7    0    2    4
  9      1     1       0    3    3   10
         2     3       8    0    3   10
         3     8       4    0    2   10
 10      1     4       5    0    5    6
         2     5       0    7    4    4
         3    10       0    7    3    3
 11      1     3       8    0    1    8
         2     8       5    0    1    7
         3     9       0    3    1    7
 12      1     9       1    0    8    9
         2     9       0    8    8    8
         3     9       1    0    9    4
 13      1     1       0   10    8    4
         2     2       9    0    8    3
         3     5       0    9    8    2
 14      1     1       0    6    7    8
         2     6       5    0    7    7
         3    10       0    4    7    6
 15      1     6       3    0    7    5
         2     6       3    0    6    6
         3     7       2    0    5    3
 16      1     1       0    8    7    8
         2     4       0    5    6    8
         3    10       9    0    5    7
 17      1     3       9    0    7   10
         2     4       6    0    7   10
         3     4       0    6    6    9
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   17   81  102
************************************************************************
