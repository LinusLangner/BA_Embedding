token_usage = {
    "gpt-4o-2024-08-06": {"input_tokens": 0, "output_tokens": 0},
    "gpt-4o-mini": {"input_tokens": 0, "output_tokens": 0}
}

def track_token_usage(usage, model_type):
    global token_usage

    token_usage[model_type]["input_tokens"] += usage.prompt_tokens
    token_usage[model_type]["output_tokens"] += usage.completion_tokens

def calculate_total_cost():
    total_cost = 0.0
    
    for model_type, usage in token_usage.items():
        
        if model_type == 'gpt-4o-2024-08-06':
            input_cost_per_million = 2.500  # $2.500 pro 1M Eingabetokens
            output_cost_per_million = 10.000  # $10.000 per 1M Ausgabetokens
        elif model_type == 'gpt-4o-mini':
            input_cost_per_million = 0.150  # $0.150 per 1M Eingabetokens
            output_cost_per_million = 0.600  # $0.600 per 1M Ausgabetokens

        input_cost = (usage["input_tokens"] / 1_000_000) * input_cost_per_million
        output_cost = (usage["output_tokens"] / 1_000_000) * output_cost_per_million
        
        total_cost += input_cost + output_cost

    return round(total_cost, 4)


