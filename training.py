import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train(net, criterion, optimizer, trainloader, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}")
        for i, data in pbar:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(Loss=running_loss / (i + 1))

        
        #print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

    print("Finished Training")
    


def train_kd(teacher_net, 
             student_net,
             optimizer, 
             trainloader, 
             epochs,
             temperature=2.0,
             alpha=0.5):

    criterion = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(log_target=True, reduction="batchmean")
    
    for epoch in range(epochs):
        running_total_loss = 0.0
        running_kd_loss = 0.0
        running_label_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}")
        for i, data in pbar:
            inputs, hard_labels = data
            optimizer.zero_grad()
            
            with torch.no_grad():
                logits_teacher = teacher_net(inputs)
            logits_student = student_net(inputs)
            
            soft_labels = F.log_softmax(logits_teacher / temperature, dim=-1)
            soft_pred = F.log_softmax(logits_student / temperature, dim=-1)
            
            kd_loss = criterion_kd(soft_pred, soft_labels) * temperature ** 2
            label_loss = criterion(logits_student, hard_labels)
            
            total_loss = alpha * kd_loss + \
                        (1 - alpha) * label_loss 
            
            total_loss.backward()
            optimizer.step()
            
            running_kd_loss += kd_loss.item()
            running_label_loss += label_loss.item()
            running_total_loss += total_loss.item()

            pbar.set_postfix(kd_loss=running_kd_loss / (i + 1),
                             label_loss=running_label_loss / (i + 1),
                             total_loss=running_total_loss / (i + 1))

    print("Finished knowledge distillation")