

def read_new_dataset(namefile):
    with open(namefile, "r") as f:
        refs = []
        hypsA = []
        hypsB = []
        for line in f:
            ref, hypA, hypB = line.strip().split("\t")
            refs.append(ref)
            hypsA.append(hypA)
            hypsB.append(hypB)
    return refs, hypsA, hypsB


if __name__ == "__main__":
    refs, hypsA, hypsB = read_new_dataset("datasets/new_dataset.txt")
    print(len(refs))
    print(len(hypsA))
    print(len(hypsB))
