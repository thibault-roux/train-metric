from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "dbddv01/gpt2-french-small"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function for text autocompletion
def autocomplete(prompt_text, max_length=50):
    # Encode the prompt text
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    
    # Generate text
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
prompt = "Once upon a time"
completed_text = autocomplete(prompt, max_length=50)
print(completed_text)
