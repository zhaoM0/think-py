import tiktoken 
import torch 
from transformer import GPT2Model

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, inputs, max_new_tokens, context_size,
                         temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        inputs_cond = inputs[:, -context_size:]
        with torch.no_grad():
            logits = model(inputs_cond)

        # Focus only on the last time step
        # (batch, seq_len, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Filer logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, 
                                 torch.tensor(float('-inf')).to(logits),
                                 logits)
            
        # Applay temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        
        # Append sampled index to the running sequence
        inputs = torch.cat((inputs, idx_next), dim=1)  # (batch, num_tokens+1)

    return inputs