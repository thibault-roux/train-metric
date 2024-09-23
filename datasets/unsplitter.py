

def load(namefile):
    with open(namefile + ".txt", "r", encoding="utf8") as f:
        next(f)
        data = []
        for line in f:
            line = line.strip().split("\t")
            data.append(line)
    return data



def unsplitter(namefile, newname):
    og = load(namefile)

    with open(newname + ".txt", "w", encoding="utf8") as f:
        f.write("reference\thypA\thypB\n")
        for line in og:
            txt = line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + line[3]
            f.write(txt + "\n")

        for line in og:
            txt = line[0] + "\t" + line[2] + "\t" + line[1] + "\t" + line[3]
            f.write(txt + "\n")


if __name__ == "__main__":
    namefile = "hats_annotation_split_gpt"
    newname = "hats_annotation_gpt"

    unsplitter(namefile, newname)