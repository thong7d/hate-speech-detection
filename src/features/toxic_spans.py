import torch
import numpy as np

def extract_toxic_spans_gradcam(
    model,
    tokenizer,
    text: str,
    target_class: int,
    top_k: int = 5,
) -> list[dict]:
    """
    Extract top-k toxic spans using Grad-CAM on TextCNN conv layers.
    Fast inference pipeline technique (requires only 1 backward pass).
    """
    # Setup hooks to capture activations and gradients
    activations = {}
    gradients = {}
    
    def get_forward_hook(name):
        def hook(module, input_t, output_t):
            activations[name] = output_t
            # Register hook on activation tensor
            def tensor_hook(grad):
                gradients[name] = grad
                return grad
            output_t.register_hook(tensor_hook)
        return hook

    # Register hooks on all conv layers of XLMRobertaTextCNN
    handles = []
    for idx, conv in enumerate(model.convs):
        h = conv.register_forward_hook(get_forward_hook(f"conv_{idx}"))
        handles.append(h)

    # Prepare inputs and run backward pass to get gradients w.r.t target class
    device = next(model.parameters()).device
    inputs = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    
    with torch.enable_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logit_target = logits[0, target_class]
        
        # Zero gradients before backward
        model.zero_grad()
        logit_target.backward()

    # Remove hooks immediately
    for h in handles:
        h.remove()

    # Process Grad-CAM for each conv branch
    seq_len = inputs["input_ids"].size(1)
    token_scores = torch.zeros(seq_len, device=device)
    counts = torch.zeros(seq_len, device=device)

    for idx, conv in enumerate(model.convs):
        name = f"conv_{idx}"
        if name not in activations or name not in gradients:
            continue
        
        A = activations[name]  # [1, num_filters, L_conv]
        G = gradients[name]    # [1, num_filters, L_conv]
        
        # Channel-wise average of gradients (global average pooling)
        alpha = torch.mean(G, dim=2, keepdim=True)  # [1, num_filters, 1]
        
        # Weighted combination of forward activation maps
        gradcam = torch.relu(torch.sum(alpha * A, dim=1)).squeeze(0)  # [L_conv]
        
        # Map back to sequence length using the conv kernel size
        k = conv.kernel_size[0]
        for i in range(len(gradcam)):
            val = gradcam[i].item()
            for offset in range(k):
                if i + offset < seq_len:
                    token_scores[i + offset] += val
                    counts[i + offset] += 1

    # Average scores
    counts = torch.clamp(counts, min=1.0)
    token_scores = (token_scores / counts).cpu().numpy()

    # Extract tokens, exclude special tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).cpu().numpy())
    
    special_tokens = {
        tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.unk_token,
        "<s>", "</s>", "<pad>", "<unk>"
    }

    scored_tokens = []
    for i, (tok, score) in enumerate(zip(tokens, token_scores)):
        if tok in special_tokens:
            continue
        clean_tok = tok.replace(" ", " ")
        scored_tokens.append({
            "token": clean_tok,
            "raw_token": tok,
            "score": float(score),
            "position": i
        })

    # Sort descending by absolute score
    scored_tokens.sort(key=lambda x: abs(x["score"]), reverse=True)
    
    # Return top_k, sorted by their original position in the text
    top_tokens = scored_tokens[:top_k]
    top_tokens.sort(key=lambda x: x["position"])
    
    return top_tokens


def extract_toxic_spans_ig(
    model,
    tokenizer,
    text: str,
    target_class: int,
    top_k: int = 5,
) -> list[dict]:
    """
    Extract top-k toxic spans using Integrated Gradients via Captum on the word embedding layer.
    Deep analysis pipeline technique (slower, but mathematically grounded).
    """
    from captum.attr import LayerIntegratedGradients

    device = next(model.parameters()).device
    inputs = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Baseline ids: fill with pad token id
    baseline_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id, device=device)

    # Define forward_func receiving token ids
    def forward_func(ids):
        # Dynamically expand attention mask shape if Captum batches the inputs during attribution
        outputs = model(
            input_ids=ids,
            attention_mask=attention_mask.expand(ids.size(0), -1)
        )
        return outputs.logits[:, target_class]

    # Target the roberta.embeddings layer directly
    lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings)
    
    # Integrated Gradients requires gradient computation
    model.eval()
    with torch.enable_grad():
        attributions = lig.attribute(
            input_ids,
            baselines=baseline_ids,
            target=None,
            n_steps=50
        )

    # Aggregate: sum across embedding dimension to get token-level scores
    token_scores = attributions.sum(dim=-1).squeeze(0).cpu().numpy()

    # Convert tokens and extract top-k
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    special_tokens = {
        tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.unk_token,
        "<s>", "</s>", "<pad>", "<unk>"
    }

    scored_tokens = []
    for i, (tok, score) in enumerate(zip(tokens, token_scores)):
        if tok in special_tokens:
            continue
        clean_tok = tok.replace(" ", " ")
        scored_tokens.append({
            "token": clean_tok,
            "raw_token": tok,
            "score": float(score),
            "position": i
        })

    # Sort descending by absolute score
    scored_tokens.sort(key=lambda x: abs(x["score"]), reverse=True)
    
    # Return top_k, sorted by their original position in the text
    top_tokens = scored_tokens[:top_k]
    top_tokens.sort(key=lambda x: x["position"])
    
    return top_tokens
