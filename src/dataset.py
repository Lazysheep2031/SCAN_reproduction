import torch
from torch.utils.data import Dataset

class SCANDataset(Dataset):
    def __init__(self, file_path, word2idx=None, action2idx=None):
        self.commands = [] # 输入：jump twice
        self.actions = []  # 输出：JUMP JUMP
        
        # 1. 读取文件
        with open(file_path, 'r') as f:
            for line in f:
                # 文件格式通常是: IN: jump twice OUT: JUMP JUMP
                part_in, part_out = line.strip().split(' OUT: ')
                input_cmd = part_in.replace('IN: ', '').split()
                output_act = part_out.split()
                
                self.commands.append(input_cmd)
                self.actions.append(output_act)

        # 2. 构建词表
        if word2idx is None:
            self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2} # 特殊符号
            self.action2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
            
            for cmd in self.commands:
                for word in cmd:
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
            
            for act in self.actions:
                for action in act:
                    if action not in self.action2idx:
                        self.action2idx[action] = len(self.action2idx)
        else:
            self.word2idx = word2idx
            self.action2idx = action2idx

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        # 3. 将文本转换为数字索引
        cmd_indices = [self.word2idx[w] for w in self.commands[idx]]
        act_indices = [self.action2idx[a] for a in self.actions[idx]]
        
        # 加上结束符
        act_indices.append(self.action2idx["<EOS>"])
        
        return torch.tensor(cmd_indices), torch.tensor(act_indices)

# 用于 DataLoader 的 collate_fn，处理变长序列
def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    src_batch, tgt_batch = zip(*batch)
    # 填充到相同长度
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_batch, tgt_batch