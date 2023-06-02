import argparse
import os
import glob
import random

parser = argparse.ArgumentParser(description="train/test set generator")

parser.add_argument("--problem_path", nargs="+", type=str, help="instances paths")
parser.add_argument("--set_path", type=str, help="where to put train/ and test/")
parser.add_argument("--split", type=float, default=0.1, help="test fraction")

args = parser.parse_args()


train_files = []
for pb_path in args.problem_path:
    train_files.extend(glob.glob(pb_path + "/*"))

test_files = []
for i in range(int(len(train_files) * args.split)):
    tp = random.choice(train_files)
    test_files.append(tp)
    train_files.remove(tp)

print("#train_files", len(train_files))
print("#test_files", len(test_files))

train_path = args.set_path + "/train"
test_path = args.set_path + "/test"
os.makedirs(train_path)
os.makedirs(test_path)

for f in train_files:
    basename = os.path.basename(f)
    os.symlink(os.path.realpath(f), train_path + "/" + basename)
for f in test_files:
    basename = os.path.basename(f)
    os.symlink(os.path.realpath(f), test_path + "/" + basename)
