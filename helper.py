
def assign_gpu(tokenizer):
    tokens_tensor = tokenizer['input_ids'].to('cuda:0')
    token_type_ids = tokenizer['token_type_ids'].to('cuda:0')
    attention_mask = tokenizer['attention_mask'].to('cuda:0')

    output = {'input_ids' : tokens_tensor, 
            'token_type_ids' : token_type_ids, 
            'attention_mask' : attention_mask}

    return output