import json


# create the new dataset using the API return


def get_final_choice(answer):
    choice = None
    for i in range(len(answer)):
        if answer[-i] in ['A', 'B', 'a', 'b']:
            if answer[-i] == "A":
                choice = "A"
            elif answer[-i] == "B":
                choice = "B"
            elif answer[-i] == "a":
                choice = 'a'
            elif answer[-i] == "b":
                choice = 'b'
            break
    
    if choice is None:
        print("Weird output: '" + str(answer) + "'")
        change = input("Choose A or B or a or b: ")
        if change == "A":
            choice = "A"
        elif change == "B":
            choice = "B"
        elif change == "a":
            choice = 'a'
        elif change == "b":
            choice = 'b'
        else:
            raise ValueError("Invalid choice: " + str(change))
    if choice == "A":
        return "1"
    elif choice == "B":
        return "0"
    elif choice == "a":
        return "11"
    elif choice == "b":
        return "10"

if __name__ == "__main__":
    inputname = "hats-5k"

    # read json
    with open("batch/" + inputname + ".jsonl", "r", encoding="utf8") as file:
        id2triplets = dict()
        for line in file:
            jsonline = json.loads(line)
            triplet = jsonline["body"]["messages"][3]["content"].split("\n")
            triplet[0] = triplet[0].split("Référence : ")[1]
            triplet[1] = triplet[1].split("Hypothèse A : ")[1]
            triplet[2] = triplet[2].split("Hypothèse B : ")[1]
            id2triplets[jsonline["custom_id"]] = triplet

    # read results
    with open("batch/" + inputname + "_results.txt", "r", encoding="utf8") as file:
        id2choice = dict()
        for line in file:
            custom_id, result = line.strip().split("\t")
            id2choice[custom_id] = get_final_choice(result)
            # print(custom_id, id2triplets[custom_id], result, id2choice[custom_id])
            # input()


    txt = "reference\thypA\thypB\tannotation\n"
    for custom_id, choice in id2choice.items():
        triplet = id2triplets[custom_id]
        txt += triplet[0] + "\t" + triplet[1] + "\t" + triplet[2] + "\t" + choice + "\n"

    with open("../datasets/" + inputname + "_annotation.txt", "w", encoding="utf8") as file:
        file.write(txt)