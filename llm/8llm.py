import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vigogne.preprocess import generate_instruct_prompt

model_name_or_path = "bofenghuang/vigogne-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")

user_query = "Expliquez la différence entre DoS et phishing."
prompt = generate_instruct_prompt(user_query)
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
input_length = input_ids.shape[1]

generated_outputs = model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.0,
        max_new_tokens=512,
    ),
    return_dict_in_generate=True,
)
generated_tokens = generated_outputs.sequences[0, input_length:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)
