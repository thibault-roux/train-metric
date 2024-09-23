

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

    with open(newname + ".txt", "w", encoding="utf8") as file:
        file.write("reference\thypA\thypB\tannotation\n")
        for line in og:
            txt = line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + line[3]
            file.write(txt + "\n")

        for line in og:
            if line[3] == "1":
                line[3] = "0"
            elif line[3] == "0":
                line[3] = "1"
            elif line[3] == "11":
                line[3] = "10"
            elif line[3] == "10":
                line[3] = "11"
            else:
                print("Weird annotation: " + line[3])
            txt = line[0] + "\t" + line[2] + "\t" + line[1] + "\t" + line[3]
            file.write(txt + "\n")


if __name__ == "__main__":
    namefile = "hats_annotation_split_gpt"
    newname = "hats_annotation_gpt"

    unsplitter(namefile, newname)