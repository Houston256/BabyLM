import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from BabyLM.modeling_gptbert import GPTBertForCausalLM

def top_k_filter(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def main():
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_path = f"checkpoints/{sys.argv[1]}" if len(sys.argv) > 1 else "checkpoints/final"
    
    print(f"Loading model from {checkpoint_path}...")
    try:
        model = GPTBertForCausalLM.from_pretrained(checkpoint_path).to(device)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    print("\n--- BabyLM Stateless Completion (Manual Loop) ---")
    
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ["exit", "quit"]:
            break
            
        # 1. Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # 2. Manual Generation Loop
        # We do this manually to avoid modifying the model file
        generated = input_ids
        max_new_tokens = 40
        temperature = 0.8
        top_k = 50
        
        print("Completion: ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for the last token
                # is_causal=True ensures we use the triangular mask
                outputs = model(generated, is_causal=True)
                logits = outputs.logits[:, -1, :] / temperature
                
                # Filter and sample
                filtered_logits = top_k_filter(logits, top_k)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Decode and print immediately
                token_str = tokenizer.decode(next_token[0])
                print(token_str, end="", flush=True)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        print() # New line after completion

if __name__ == "__main__":
    main()
