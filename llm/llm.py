from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

model_name_or_path = "bigscience/bloom-1b1" # "bofenghuang/vigogne-2-70b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=False) # original padding_side='right'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")

streamer = TextStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)


def chat(
    query: str,
    history: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: float = 0,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 10,
    **kwargs,
):
    if history is None:
        history = []

    history.append({"role": "user", "content": query})

    input_ids = tokenizer.apply_chat_template(history, return_tensors="pt").to(model.device)
    input_length = input_ids.shape[1]

    generated_outputs = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        ),
        streamer=streamer,
        return_dict_in_generate=True,
    )

    generated_tokens = generated_outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    history.append({"role": "assistant", "content": generated_text})

    return generated_text, history


# 1st round
response, history = chat("Un escargot parcourt 100 mètres en 5 heures. Quelle est sa vitesse ?", history=None)

# # 2nd round
# response, history = chat("Quand il peut dépasser le lapin ?", history=history)

# # 3rd round
# response, history = chat("Écris une histoire imaginative qui met en scène une compétition de course entre un escargot et un lapin.", history=history)



print(history)
print(response)