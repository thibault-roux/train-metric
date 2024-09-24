from openai import OpenAI
import pickle


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
    letsretrieve = True


    inputname = "new_dataset_subset" # batchinput


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
        with open("pickle/batchobject.pkl", "wb") as f:
            pickle.dump(batch_object, f)

    if letsprintbatchobject:
        # print batch object
        with open("pickle/batchobject.pkl", "rb") as f:
            batch_object = pickle.load(f)
            # print(batch_object)
            print(batch_object.id)
            # input()

    if letscheckstatus:
        # check status
        # get batch object
        with open("pickle/batchobject.pkl", "rb") as f:
            batch_object = pickle.load(f)
            # print(batch_object)
        tmp = client.batches.retrieve(batch_object.id)
        print(tmp)
        print(tmp.output_file_id)

    if letsretrieve:
        # retrieve the results
        with open("pickle/batchobject.pkl", "rb") as f:
            batch_object = pickle.load(f)
        tmp = client.batches.retrieve(batch_object.id)
        file_response = client.files.content(tmp.output_file_id)
        print(file_response.text)
