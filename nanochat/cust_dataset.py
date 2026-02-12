import functools
import json
from collections.abc import Mapping, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq

device = "cuda" if torch.cuda.is_available() else "cpu"  # 检测是否有GPU，如果有则使用，否则使用CPU


# 可能需要特殊结构
def prepare_for_dialogue(examples):
    """对话任务处理"""
    processed = []
    for doc in examples['text']:
        if "NaN" in doc:
            doc = doc.replace("NaN", "\"？\"")
        conversations = json.loads(doc)
        processed.append(conversations['conversations'])
    return {'conversations': processed}


def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    for conv in batched_conv:
        input_ids, loss_masks = [tokenizer.get_command('[gMASK]'), tokenizer.get_command('sop')], \
                                [False, False]
        for message in conv:
            if message['from'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            new_input_ids = tokenizer.build_single_message(
                message['from'], '', message['value']
            )
            new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}


def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_conv = batch['conversations']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []

    for conv in batched_conv:
        input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]
        for message in conv:
            if len(input_ids) >= max_input_length:
                break
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['from'], '', message['value']
                )
                if message['from'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    output_ids.append(tokenizer.eos_token_id)
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]
                    )
                    batched_output_ids.append(output_ids[:max_output_length])
                input_ids += new_input_ids
    return {'input_ids': batched_input_ids, 'labels': batched_output_ids}


def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        src_len = batch['input_ids'].size(-1)
        tgt_len = batch['labels'].size(-1)
        max_len = max(src_len, tgt_len)
        batch['input_ids'] = self._pad_tensors_to_max_len(batch['input_ids'], max_len)
        batch['labels'] = self._pad_tensors_to_max_len(batch['labels'], max_len)
        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['labels'] = batch['labels'].to(device)
        return batch


def get_dataloader(args, tokenizer):
    data_files = {'train': args.train_data_file, 'test': args.eval_data_file}
    dataset = load_dataset('text', data_files=data_files)
    train_dataset = dataset['train']
    train_dataset = train_dataset.map(
        prepare_for_dialogue,
        batched=True,
        batch_size=args.device_batch_size,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=True,
        num_proc=args.num_workers
    )
    train_dataset = train_dataset.map(
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=args.max_seq_len // 2,
            max_output_length=args.max_seq_len // 2
        ),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=True,
    )
    print("train_dataset:")
    _sanity_check(train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer)
    print('=' * 100)
    eval_dataset = None
    if dataset['test'] is not None:
        eval_dataset = dataset['test']
        eval_dataset = eval_dataset.map(
            prepare_for_dialogue,
            batched=True,
            batch_size=args.device_batch_size,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=True,
            num_proc=args.num_workers
        )
        eval_dataset = eval_dataset.map(
            functools.partial(
                process_batch_eval,
                tokenizer=tokenizer,
                max_input_length=args.max_seq_len // 2,
                max_output_length=args.max_seq_len // 2
            ),
            batched=True,
            num_proc=args.num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=True,
        )
        print("eval_dataset:")
        _sanity_check(eval_dataset[0]["input_ids"], eval_dataset[0]["labels"], tokenizer)
        print('=' * 100)

    it = iter(train_dataset)
    for i in range(5):
        data = next(it)
        print(tokenizer.decode(data['input_ids']))
        print(''.join([tokenizer.decode([i]) if i != -100 else str(i) for i in data['labels']]))
        print('-' * 100)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding='longest',
        max_length=args.max_seq_len,
        return_tensors='pt',
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.device_batch_size,
        collate_fn=data_collator
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.device_batch_size,
        collate_fn=data_collator
    )

    return train_loader, eval_loader
