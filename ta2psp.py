#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#
import argparse
import numpy as np
from utils.utils import load_problem

parser = argparse.ArgumentParser(description="taillard to psp converter")
parser.add_argument("inputfile", type=str, help="taillard problem to read")
parser.add_argument("outputfile", type=str, help="where to write psp problem")

args = parser.parse_args()
separator = "************************************************************************\n"
separator2 = (
    "------------------------------------------------------------------------\n"
)

n_j, n_m, affectations, durations = load_problem(args.inputfile)
durations = durations[:, :, 0]
sink = n_j * n_m + 2
with open(args.outputfile, "w") as f:
    f.write(separator)
    f.write(f"file with basedata            : {args.inputfile}\n")
    f.write("initial value random generator: 1\n")
    f.write(separator)
    f.write("projects                      :  1\n")
    f.write(f"jobs (incl. supersource/sink ):  {n_j*n_m+2}\n")
    f.write(f"horizon                       :  {np.sum(durations)}\n")
    f.write("RESOURCES\n")
    f.write(f"- renewable                 :  {n_m}   R\n")
    f.write("- nonrenewable              :  0   N\n")
    f.write("- doubly constrained        :  0   D\n")
    f.write(separator)
    f.write("PROJECT INFORMATION:\n")
    f.write("pronr.  #jobs rel.date duedate tardcost  MPM-Time\n")
    f.write(f"1     {n_j*n_m+2}      0       41       21       41\n")
    f.write(separator)
    f.write("PRECEDENCE RELATIONS:\n")
    f.write("jobnr.    #modes  #successors   successors\n")
    f.write(f"1           1           {n_j}           ")
    for j in range(0, n_j):
        f.write(f"{2+j*n_m}  ")
    f.write("\n")
    for j in range(n_j):
        for m in range(n_m - 1):
            f.write(f"{2+j*n_m + m}           1           1           {2+j*n_m+m+1}\n")
        f.write(f"{(j+1)*n_m+1}           1           1           {sink}\n")
    f.write(f"{sink}           1           0\n")
    f.write(separator)
    f.write("REQUESTS/DURATIONS:\n")
    f.write("jobnr. mode duration  ")
    for m in range(n_m):
        f.write(f"R {m+1}  ")
    f.write("\n")
    f.write(separator2)
    f.write(" 1     1     0        ")
    for m in range(n_m):
        f.write("0    ")
    f.write("\n")
    for j in range(n_j):
        for m in range(n_m):
            rusage = [0] * n_m
            rusage[affectations[j, m] - 1] = 1
            rusagestring = ""
            for r in rusage:
                rusagestring += str(r) + "    "
            f.write(
                f" {2 + j * n_m + m}     1    {int(durations[j, m])}      {rusagestring}\n"
            )
    f.write(f" {sink}     1    0        ")
    for m in range(n_m):
        f.write("0    ")
    f.write("\n")
    f.write(separator)
    f.write("RESOURCEAVAILABILITIES:\n  ")
    for m in range(n_m):
        f.write(f"R {m+1}  ")
    f.write("\n   ")
    for m in range(n_m):
        f.write("1    ")
    f.write("\n")
    f.write(separator)
