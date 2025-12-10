import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math

# 导入模块
from src.dataset import SCANDataset, collate_fn
from src.model import Encoder, Decoder, Seq2Seq
from src.train import train, evaluate  # 导入刚才写的训练函数

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 1. 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple MPS (Metal Performance Shaders) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU!")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (Slow).")

    # 2. 准备数据路径 
    train_path = 'data/tasks_train_addprim_jump.txt'
    test_path = 'data/tasks_test_addprim_jump.txt'

    print("Loading datasets...")
    # 读取训练集
    train_data = SCANDataset(train_path)
    # 读取测试集 (要共享训练集的词表 word2idx 和 action2idx)
    test_data = SCANDataset(test_path, word2idx=train_data.word2idx, action2idx=train_data.action2idx)

    # 3. 创建 DataLoader
    BATCH_SIZE = 32
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. 初始化模型参数
    INPUT_DIM = len(train_data.word2idx)
    OUTPUT_DIM = len(train_data.action2idx)
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    HID_DIM = 64
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    # 忽略 <PAD> 的 Loss
    criterion = nn.CrossEntropyLoss(ignore_index=train_data.action2idx["<PAD>"])

    # 5. 开始训练循环
    N_EPOCHS = 10
    CLIP = 1

    print(f"Start training for {N_EPOCHS} epochs...")

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        # 调用 src/train.py 里的函数
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, device)
        valid_loss = evaluate(model, test_loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print("-" * 60)

if __name__ == "__main__":
    main()