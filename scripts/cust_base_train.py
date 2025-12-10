# Model architecture
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_scheduler

from nanochat.gpt import GPTConfig, GPT

depth = 12 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 128 # max context length
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
batch_size = 8
print(f"num_layers: {num_layers}")
print(f"model_dim: {model_dim}")
print(f"num_heads: {num_heads}")
print(f"num_kv_heads: {num_kv_heads}")

model_name_or_path = "../../ml-demo/nlp/download_model/gpt2-chinese-cluecorpussmall"
device = "cuda" if torch.cuda.is_available() else "cpu"  # 检测是否有GPU，如果有则使用，否则使用CPU

tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
special_tokens = {
    'additional_special_tokens': [
        # tokens below are only used during finetuning to render Conversations into token ids
        "<|user_start|>",  # user messages
        "<|user_end|>",
        "<|assistant_start|>",  # assistant messages
        "<|assistant_end|>"
    ]
}
tokenizer.add_special_tokens(special_tokens)
vocab_size = len(tokenizer.vocab)
# Initialize the Model
# Create a new model with random weights
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)

model_config = GPTConfig(**model_config_kwargs)
model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
print(f"origin model:\n {model}")

class CustDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self):
        return len(self.data) - self.max_seq_len

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.max_seq_len + 1]
        return chunk

text = open('news-commentary-v13-zh-en.txt', 'r', encoding='utf-8').read() # don't worry we won't run out of file handles
train_dataset = CustDataset(text, max_seq_len = max_seq_len)
print("\n\ndataset:")
chunk = next(iter(train_dataset))
print(chunk)

pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

def collate_fn(batch):
    batch_ids = []
    for i, chunk in enumerate(batch):
        encode = tokenizer.encode(chunk)
        batch_ids.append(encode)
    nrows = len(batch)
    ncols = max(len(ids) for ids in batch_ids) - 1  # seq of n creates inputs/targets of n-1
    inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
    targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
    for i, ids in enumerate(batch_ids):
        n = len(ids)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        inputs[i, :n - 1] = ids_tensor[:-1]
        # recall -1 is the ignore index, so mask out targets where mask is 0
        row_targets = ids_tensor[1:]
        # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
        row_targets[row_targets == 0] = -1  # mask out targets where mask is 0
        targets[i, :n - 1] = row_targets
    inputs = inputs.to(device)  # move to device
    targets = targets.to(device)
    return {'input_ids': inputs, 'labels': targets}


train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn
)
print("\n\ndataload:")
data = next(iter(train_dataloader))
ids = data['input_ids'][0]
ids_str = tokenizer.decode(ids)
print(ids_str)
l = data['labels'][0].tolist()
labels_str = ''.join([tokenizer.convert_ids_to_tokens(int(i)) if i != -100 else str(i) for i in l])
print(labels_str)

print("\n\nstart training ...")
# 分割训练集和验证集
print(f"训练样本数: {len(train_dataset)}")
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0006, betas=(0.9, 0.98), eps=1e-06
)
# 定义学习率调度器
scheduler = get_scheduler(name="linear",  # 线性调度器
                          num_warmup_steps=100,  # 预热步数
                          num_training_steps=len(train_dataloader),  # 总训练步数
                          optimizer=optimizer)
model.train()
for epoch in range(10):
    for i, batch in enumerate(train_dataloader):
        idx, targets = batch['input_ids'], batch['labels']
        idx = idx.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()  # 清空梯度
        loss = model(idx, targets)
        train_loss = loss.detach()  # for logging
        loss.backward()
        optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新学习率

        if i % 50 == 0:  # 每隔50个批次打印一次信息
            lr = optimizer.state_dict()["param_groups"][0]['lr']  # 获取当前学习率
            # 打印训练信息
            print(f"epoch:{epoch}\t batch:{i}\t loss:{train_loss.item()}\t lr:{lr}\t")