from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

def load_model_and_tokenenizer(model_name_or_path):
    print("load_model")
 
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_relevant(question, model, tokenizer, index, CHUNK_SIZE = 512):
    tokenizer.pad_token = tokenizer.eos_token
    with torch.no_grad():
        tokenized = tokenizer([question,], padding=True, truncation=True, max_length = CHUNK_SIZE,return_tensors='pt')        
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        input_encoded = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        output = model(**input_encoded, output_hidden_states=True, return_dict=True)
        token_embedded = output['hidden_states'][-1].detach()
        question_embedding = mean_pooling(token_embedded, attention_mask)
        question_embedding = question_embedding.cpu().numpy()
        norm = np.linalg.norm(question_embedding)
        normalized_embeddings = (question_embedding / norm).astype(np.float32)
    _, idx = index.search(normalized_embeddings, 200)
    return idx