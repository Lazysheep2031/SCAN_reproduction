import torch

def train(model, iterator, optimizer, criterion, clip, device):
    """
    执行一个 Epoch 的训练
    """
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        # 将数据移动到设备 (MPS/CUDA/CPU)
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        
        # model forward
        output = model(src, trg)
        # output: [batch size, trg len, output dim]
        # trg: [batch size, trg len]
        
        output_dim = output.shape[-1]
        
        # 展平数据以计算 Loss (跳过 <SOS> token)
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    """
    执行评估（测试）
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            # 测试时关闭 Teacher Forcing (使用 0)
            output = model(src, trg, 0) 

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)