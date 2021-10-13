from config import AUDIO_DIM, TEXT_DIM, IMAGE_DIM
from config import USE_TEXT, USE_AUDIO, USE_IMAGE, LR, EPOCHS
from config import unimodal_folder, GRU_DIM
import sys
import os
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.optim import Adam
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, imgs, aud, text, target, length):
        self.imgs = imgs
        self.aud = aud
        self.text = text
        self.target = target
        self.length = length
        
    def __getitem__(self, index):
        
        return {'img': self.imgs[index], 'aud': self.aud[index],
                'text': self.text[index], 'target': self.target[index],
                'length':self.length[index]}
    
    def __len__(self):
        return len(self.imgs)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, layers, bidirectional_flag):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        if bidirectional_flag == True:
          self.fc = nn.Linear(2*hidden_dim, output_dim)
        else:
          self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.bidirectional_used = bidirectional_flag
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        output, _ = self.rnn(x)
        #output = self.fc(output)
        out = self.fc(output)
        return output, out

def compute_accuracy(output, train_dic):
    batch_correct = 0.0
    batch_total = 0.0
    for i in range(output.shape[0]):
        req_len = torch.sum(train_dic['length'][i]).int()
        out_required = output[i][:req_len, :]
        target_required = train_dic['target'][i][:req_len].long()
        pred = torch.argmax(out_required, dim = 1)
        correct_pred = (pred == target_required).float()
        tot_correct = correct_pred.sum()
        batch_correct += tot_correct
        batch_total += req_len
    return batch_correct/batch_total

def compute_loss(output, train_dic):
    batch_loss = 0.0
    for i in range(output.shape[0]):
        req_len = torch.sum(train_dic['length'][i]).int()
        loss = nn.CrossEntropyLoss(ignore_index = 4)(output[i][:req_len, :],
                                                     train_dic['target'][i][:req_len].long().to(device))
        batch_loss += loss
    return batch_loss/output.shape[0]

def count_parameters(model):
    print("NUM PARAMETERS", sum(p.numel() for p in model.parameters() if p.requires_grad))

def train(train_data, val_data, max_length):
    base_lr = LR
    n_gpu = torch.cuda.device_count()
    for fold in range(1):
        final_val_loss = 999999
        logger.info(f"Running fold {fold}")
        train_loader = train_data[fold]
        val_loader = val_data[fold]
        model = GRUModel(GRU_DIM*4, GRU_DIM, 4, 0.2, 2, True).double()
        model.to(device)
        count_parameters(model)
        if USE_TEXT:
            text_model = GRUModel(TEXT_DIM, GRU_DIM, 4, 0.2, 2, True).double()
            text_model.to(device)
            checkpoint_text = torch.load(os.path.join(unimodal_folder, 'best_model_text'+str(fold)+'.tar'))
            text_model.load_state_dict(checkpoint_text['model_state_dict'])
            text_model.eval()
        if USE_AUDIO:
            aud_model = GRUModel(AUDIO_DIM, GRU_DIM, 4, 0.2, 2, True).double()
            aud_model.to(device)
            checkpoint_aud = torch.load(os.path.join(unimodal_folder, 'best_model_aud'+str(fold)+'.tar'))
            aud_model.load_state_dict(checkpoint_aud['model_state_dict'])
            aud_model.eval()

        optimizer = Adam(model.parameters(), lr=base_lr)
        for e in range(EPOCHS):
            tot_loss, tot_acc = 0.0, 0.0
            model.train()
            for ind, train_dic in enumerate(train_loader):
                model.zero_grad()
                if USE_AUDIO:
                    inp = train_dic['aud'].permute(0, 2, 1).double()
                    out_aud, _ = aud_model.forward(inp.to(device))
                if USE_TEXT:
                    inp = train_dic['text'].permute(0, 2, 1).double()
                    out_text, _ = text_model.forward(inp.to(device))

                inp = torch.cat((out_aud, out_text), axis = 2)
                _, out = model.forward(inp.to(device))
                #torch.save(out1, 'gpu_out'+str(e)+str(ind)+".pt")
                train_dic['target'][train_dic['target'] == -1] = 4
                acc = compute_accuracy(out.cpu(), train_dic)
                loss = compute_loss(out.to(device), train_dic)
                tot_loss += loss.item()
                tot_acc += acc.item()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss, val_acc = 0.0, 0.0
                for ind, val_dic in enumerate(val_loader):
                    if USE_AUDIO:
                        inp = val_dic['aud'].permute(0, 2, 1).double()
                        out_aud, _ = aud_model.forward(inp.to(device))
                    if USE_TEXT:
                        inp = val_dic['text'].permute(0, 2, 1).double()
                        out_text, _ = text_model.forward(inp.to(device))

                    inp = torch.cat((out_aud, out_text), axis = 2)
                    _, val_out = model.forward(inp.to(device))
                    val_dic['target'][val_dic['target'] == -1] = 4
                    val_acc += compute_accuracy(val_out.cpu(), val_dic).item()
                    val_loss += compute_loss(val_out.to(device), val_dic).item()
                if val_loss < final_val_loss:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                'lstm_best_model_' + str(fold) + '.tar')
                    final_val_loss = val_loss
                e_log = e + 1
                train_loss = tot_loss/len(train_loader)
                train_acc = tot_acc/len(train_loader)
                val_loss_log = val_loss/len(val_loader)
                val_acc_log = val_acc/len(val_loader)
                logger.info(f"Epoch {e_log}, \
                            Training Loss {train_loss},\
                            Training Accuracy {train_acc}")
                logger.info(f"Epoch {e_log}, \
                            Validation Loss {val_loss_log},\
                            Validation Accuracy {val_acc_log}")
