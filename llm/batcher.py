from openai import OpenAI
import pickle
import json


# step 1 : save the file in a json
# step 2 : upload the batch to openai
# step 3 : create the batch
# step 4 : retrieve the results

"""
source ~/.zshrc
"""

if __name__ == "__main__":
    client = OpenAI()


    letsupload = False
    letscreate = False
    letsprintbatchobject = False
    letscheckstatus = False
    letsretrieve = False
    letsgoodformat = True

    inputname = "hats-5k" # "new_dataset" # new_dataset_subset # batchinput


    if letsupload:
        # step 2 : upload the batch to openai
        batch_input_file = client.files.create(
        file=open("batch/" + inputname + ".jsonl", "rb"),
        purpose="batch"
        )


    if letscreate:
        # step 3 : create the batch
        batch_input_file_id = batch_input_file.id

        batch_object = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "nightly eval job"
            }
        )

        # save batch object
        with open("pickle/" + inputname + "_batchobject.pkl", "wb") as f:
            pickle.dump(batch_object, f)

    if letsprintbatchobject:
        # print batch object
        with open("pickle/" + inputname + "_batchobject.pkl", "rb") as f:
            batch_object = pickle.load(f)
            # print(batch_object)
            print(batch_object.id)
            # input()

    if letscheckstatus:
        # check status
        # get batch object
        with open("pickle/" + inputname + "_batchobject.pkl", "rb") as f:
            batch_object = pickle.load(f)
            # print(batch_object)
        tmp = client.batches.retrieve(batch_object.id)
        print(tmp)
        print("The status of the batch is: " + tmp.status + "!")

    if letsretrieve:
        # retrieve the results
        with open("pickle/" + inputname + "_batchobject.pkl", "rb") as f:
            batch_object = pickle.load(f)
        tmp = client.batches.retrieve(batch_object.id)
        file_response = client.files.content(tmp.output_file_id)
        print(file_response.text)
        # save pickle
        with open("pickle/" + inputname + "_response.pkl", "wb") as f:
            pickle.dump(file_response, f)

    if letsgoodformat:
        # read pickle
        with open("pickle/" + inputname + "_response.pkl", "rb") as f:
            file_response = pickle.load(f)
            # print(file_response)
            
            txt = ""
            for line in file_response.text.split("\n"):
                # convert line to json
                try:
                    line = json.loads(line)
                    txt += line["custom_id"] + "\t" + line["response"]["body"]["choices"][0]["message"]["content"] + "\n"
                except:
                    print(line)
        with open("batch/" + inputname + "_results.txt", "w") as f:
            f.write(txt)