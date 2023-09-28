from datasets import Dataset, DatasetDict

def preprocessing(samples, tokenizer, max_input_length, max_output_length):
    samples["input_ids"] = tokenizer(samples["keywords"], max_length=max_input_length+1, truncation=True, padding="max_length")["input_ids"] + [tokenizer.pad_token_id]
    samples["input_ids"].remove(tokenizer.sep_token_id)
    samples["input_ids"].remove(tokenizer.cls_token_id)

    if isinstance(samples["descriptions"], list):  # If multi-reference (valid/test)
        tokenized_outputs = tokenizer(samples["descriptions"], max_length=max_output_length, truncation=True)["input_ids"]
        samples["labels"] = tokenized_outputs
    else:  # If single-reference (training)
        tokenized_outputs = tokenizer(samples["descriptions"], max_length=max_output_length, truncation=True, padding="max_length")["input_ids"]
        samples["labels"] = tokenized_outputs

    return samples


def load_and_preprocess_data(data_path, tokenizer, max_input_length, max_output_length, method, preprocessing_fct=preprocessing, test_set_shape="multi-ref"):

    # Build data dictionnary
    data = {
        "train": {"keywords": [], "descriptions": []},
        "valid": {"keywords": [], "descriptions": []},
        "test": {"keywords": [], "descriptions": []}
    }

    # Get data splits as lists
    train_keywords     = open(f'{data_path}/train_keywords.txt', encoding="utf-8").readlines()
    train_descriptions = open(f'{data_path}/train_descriptions.txt', encoding="utf-8").readlines()

    valid_keywords     = open(f'{data_path}/valid_keywords.txt', encoding="utf-8").readlines()
    valid_descriptions = open(f'{data_path}/valid_descriptions.txt', encoding="utf-8").readlines()

    test_keywords      = open(f'{data_path}/test_keywords.txt', encoding="utf-8").readlines()
    test_descriptions  = open(f'{data_path}/test_descriptions.txt', encoding="utf-8").readlines()


    # FULL TRAINING (Training = Train+Validation, Test = Test)
    if method == "Full Training":

        for i in range(len(train_keywords)): # Single-reference
            data["train"]['keywords'].append(train_keywords[i].replace('\n', ''))
            data["train"]['descriptions'].append(train_descriptions[i].replace('\n', ''))
        for i in range(len(valid_keywords)): # Single-reference
            data["train"]['keywords'].append(valid_keywords[i].replace('\n', ''))
            data["train"]['descriptions'].append(valid_descriptions[i].replace('\n', ''))

        if test_set_shape == "single-ref":
            for i in range(len(test_keywords)): # Single-reference
                data["test"]['keywords'].append(test_keywords[i].replace('\n', ''))
                data["test"]['descriptions'].append(test_descriptions[i].replace('\n', ''))
        elif test_set_shape == "multi-ref":
            test_src_to_refs = {}
            for i in range(len(test_keywords)): # Multi-reference
                src = test_keywords[i].replace('\n', '')
                desc = test_descriptions[i].replace('\n', '')
                if src not in test_src_to_refs:
                    test_src_to_refs[src] = []
                test_src_to_refs[src].append(desc)
            data['test']['keywords'] = list(test_src_to_refs.keys())
            data['test']['descriptions'] = list(test_src_to_refs.values())
        else:
            assert False, f"'{test_set_shape}' is not valid. Please use either 'single-ref' or 'multi-ref'."
    
    # SPLIT TRAINING (Training = Train, Test = Validation)
    elif method == "Split Training":
        for i in range(len(train_keywords)): # Single-reference
            data["train"]['keywords'].append(train_keywords[i].replace('\n', ''))
            data["train"]['descriptions'].append(train_descriptions[i].replace('\n', ''))
        
        if test_set_shape == "single-ref":
            for i in range(len(valid_keywords)): # Single-reference
                data["test"]['keywords'].append(valid_keywords[i].replace('\n', ''))
                data["test"]['descriptions'].append(valid_descriptions[i].replace('\n', ''))
        elif test_set_shape == "multi-ref":
            valid_src_to_refs = {}
            for i in range(len(valid_keywords)): # Multi-reference
                src = valid_keywords[i].replace('\n', '')
                desc = valid_descriptions[i].replace('\n', '')
                if src not in valid_src_to_refs:
                    valid_src_to_refs[src] = []
                valid_src_to_refs[src].append(desc)
            data['test']['keywords'] = list(valid_src_to_refs.keys())
            data['test']['descriptions'] = list(valid_src_to_refs.values())
        else:
            assert False, f"'{test_set_shape}' is not valid. Please use either 'single-ref' or 'multi-ref'."
    
    else:
        assert False, f"'{method}' is not a valid method. Please use either 'Full Training' or 'Split Training'."


    # Create Dataset objects
    train_dataset = Dataset.from_dict(data["train"])
    valid_dataset = Dataset.from_dict(data["test"])

    # Split data set
    dataset = DatasetDict(
        {"train": train_dataset,
        "validation": valid_dataset}
    )

    tokenized_dataset = dataset.map(lambda x: preprocessing_fct(x, tokenizer, max_input_length, max_output_length))

    return tokenized_dataset
