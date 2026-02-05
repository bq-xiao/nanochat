import functools
import json

import torch
from datasets import Dataset
# 可能需要特殊结构
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"  # 检测是否有GPU，如果有则使用，否则使用CPU


def prepare_for_dialogue(examples, tokenizer, block_size):
    """对话任务处理"""
    processed = []
    for doc in examples['text']:
        if "NaN" in doc:
            doc = doc.replace("NaN", "\"？\"")
        conversations = json.loads(doc)
        tokens = []
        for turn in conversations['conversations']:
            if turn['from'] == 'user':
                tokens.extend(tokenizer.encode(f"User: {turn['value']}"))
            else:
                tokens.extend(tokenizer.encode(f"Assistant: {turn['value']}"))
            tokens.append(tokenizer.eod)
        processed.extend(tokens)

    total = len(processed)
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = 0
    if total >= block_size:
        total_length = (total // block_size) * block_size
    if total - total_length > 0:
        total_length = total_length + 1
    batch = [processed[i: i + block_size] for i in range(0, total_length, block_size)]

    return {'input_ids': batch}


def get_dataloader(args, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    dataset = Dataset.from_text(args.train_data_file)
    if args.eval_data_file is not None:
        eval_dataset = Dataset.from_text(args.eval_data_file)
    print("train dataset processing ...")
    dataset = dataset.map(
        functools.partial(
            prepare_for_dialogue,
            tokenizer=tokenizer,
            block_size=args.block_size
        ),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        num_proc=args.num_workers
    )

    if eval_dataset is not None:
        print("using eval dataset, processing ...")
        eval_dataset = eval_dataset.map(
            functools.partial(
                prepare_for_dialogue,
                tokenizer=tokenizer,
                block_size=args.block_size
            ),
            batched=True,
            batch_size=args.batch_size,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=True,
            num_proc=args.num_workers
        )

    def data_collator(examples):
        ids = [torch.tensor(example['input_ids'], dtype=torch.long, device=device) for example in examples]
        input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        shifted = labels[:, 1:]
        shifted[shifted == pad_token_id] = -100
        return {'input_ids': input_ids, 'labels': shifted}

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )

    return train_loader, eval_loader
