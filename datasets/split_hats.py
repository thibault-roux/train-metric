
import random

# split hats_annotation.txt in train and test files

path = "./hats_annotation.txt"

with open(path, "r") as f:
    # skip first line
    f.readline()
    lines = f.readlines()

# shuffle lines
random.shuffle(lines)

# split in train and test
train = lines[:int(len(lines)*0.5)]
test = lines[int(len(lines)*0.5):]

# write train and test files
with open("./hats_annotation_train.txt", "w") as f:
    f.write("reference\thypA\thypB\tannotation\n")
    f.writelines(train)

with open("./hats_annotation_test.txt", "w") as f:
    f.write("reference\thypA\thypB\tannotation\n")
    f.writelines(test)
