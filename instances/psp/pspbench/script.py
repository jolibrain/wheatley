import shutil
import glob

for x in ["30", "60", "90"]:
    for i in range(44, 49):
        for f in glob.glob("seen/j" + x + str(i) + "*"):
            shutil.move(f, "unknown")

    for i in range(1, 44):
        for f in glob.glob("seen/j" + x + str(i) + "_9.sm"):
            shutil.move(f, "unseen")
        for f in glob.glob("seen/j" + x + str(i) + "_10.sm"):
            shutil.move(f, "unseen")


x = "120"
for i in range(56, 61):
    for f in glob.glob("seen/j" + x + str(i) + "*"):
        shutil.move(f, "unknown")

for i in range(1, 56):
    for f in glob.glob("seen/j" + x + str(i) + "_9.sm"):
        shutil.move(f, "unseen")
    for f in glob.glob("seen/j" + x + str(i) + "_10.sm"):
        shutil.move(f, "unseen")
