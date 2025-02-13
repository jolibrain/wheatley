import argparse
import json
import glob
import os


def read_stats(directory):
    min_crit = None
    with open(directory + "/progress.json", "r") as f:
        for n, line in enumerate(f):
            j = json.loads(line)
            crit = j["validation/ppo_criterion"]
            if min_crit is None or crit < min_crit:
                min_crit = crit
                it = n
    print(f"min makespan: {min_crit} at iteration {it}")


def read_sols(directory, valid=True):
    sols = {120: {}, 90: {}, 60: {}, 30: {}}
    if valid:
        files = glob.glob(directory + "/sol_*.txt")
    else:
        files = glob.glob(directory + "/*.sol")
    for f in files:
        with open(f, "r") as ff:
            pbline = ff.readline()
            if not valid:
                pbline = pbline.split(" ")[1].rstrip()

            bn = os.path.basename(pbline)
            pbfile = os.path.basename(pbline).split(".")[0]

            inst = int(pbfile.split("_")[1])
            if pbfile[1:4] == "120":
                size = 120
                try:
                    param = int(pbfile[4:6])
                except ValueError:
                    param = int(pbfile[4:5])

            elif pbfile[1:3] in ["60", "30", "90"]:
                size = int(pbfile[1:3])
                try:
                    param = int(pbfile[3:5])
                except ValueError:
                    param = int(pbfile[3:4])

            else:
                print("error")

            if param not in sols[size]:
                sols[size][param] = {}

            critline = ff.readline()
            crit = float(critline.split(" ")[2].rstrip())
            sols[size][param][inst] = crit
    return sols


def read_pspsols(fname):
    sols = {}
    with open(fname, "r") as f:
        for line in f:
            try:
                param = int(line[:2])
                vals = line.split()
                inst = int(vals[1])
                crit = int(vals[2])
                if param not in sols:
                    sols[param] = {}
                sols[param][inst] = crit
            except ValueError:
                pass
    return sols


def compute_perfs(wsol, psol):
    gaps = {30: [], 60: [], 90: [], 120: []}
    for size in [30, 60, 90, 120]:
        for param in wsol[size].keys():
            for inst in wsol[size][param].keys():
                better = wsol[size][param][inst] < psol[size][param][inst]
                if better:
                    print("BETTER !!!!")
                gap = (wsol[size][param][inst] - psol[size][param][inst]) / psol[size][
                    param
                ][inst]
                gaps[size].append(gap)
    mean_gaps = {}
    mg = 0
    nmg = 0
    for size in gaps.keys():
        if len(gaps[size]) != 0:
            mean_gaps[size] = sum(gaps[size]) / len(gaps[size])
            mg += sum(gaps[size])
            nmg += len(gaps[size])
    mean_gaps["all"] = mg / nmg
    return mean_gaps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_unknown")
    parser.add_argument("--dir_unseen", default=None)
    args = parser.parse_args()

    if args.dir_unknown is not None:
        read_stats(args.dir_unknown)
        wheatley_sols_unknown = read_sols(args.dir_unknown)

    if args.dir_unseen is not None:
        wheatley_sols_unseen = read_sols(args.dir_unseen, valid=False)
    psplibsols = {}
    psplibsols[30] = read_pspsols("j30opt.sm")
    psplibsols[60] = read_pspsols("j60hrs.sm")
    psplibsols[90] = read_pspsols("j90hrs.sm")
    psplibsols[120] = read_pspsols("j120hrs.sm")

    if args.dir_unknown is not None:
        mean_gaps_unknown = compute_perfs(wheatley_sols_unknown, psplibsols)
        print(f"mean_gaps_unknown: {mean_gaps_unknown}")

    if args.dir_unseen is not None:
        mean_gaps_unseen = compute_perfs(wheatley_sols_unseen, psplibsols)
        print(f"mean_gaps_unseen: {mean_gaps_unseen}")
