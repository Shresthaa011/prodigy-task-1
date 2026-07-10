import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_text(model_path, prompt, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_p=0.92,
        temperature=0.85,
        do_sample=True,
        top_k=50,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    import os
    model_path = "./results"  # Path to the fine-tuned model
    if not os.path.exists(model_path):
        print("Fine-tuned model not found in ./results, falling back to base 'gpt2'...")
        model_path = "gpt2"
        
    prompt = "To be, or not to be, that is the question:"
    
    print(f"Prompt: {prompt}\n")
    print("Generated Text:")
    generated = generate_text(model_path, prompt)
    print(generated)
    
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n\nGenerated Text:\n{generated}\n")
    print("\n[The output has also been saved to output.txt]")
