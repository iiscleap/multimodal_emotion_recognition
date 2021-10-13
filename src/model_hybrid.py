from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM, IMAGE_DIM, HIDDEN_SIZE_TRANS
from config import USE_TEXT, USE_AUDIO, USE_IMAGE, LR, EPOCHS
from config import unimodal_folder, GRU_DIM
import sys
import os
import logging
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from pytorch_transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                                BertEncoder, BertLayerNorm,
                                                BertPreTrainedModel)
from src.model import MultiModalTransformer as MMT

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
#device = torch.device("cpu")
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

class MultiModalTransformer(BertPreTrainedModel):

    def __init__(self, config, aud_ft_len, text_ft_len, img_ft_len, max_utt_len, req_aud, req_text):
        super().__init__(config)
        self.audio_feature_length = aud_ft_len
        self.text_feature_length = text_ft_len
        self.img_feature_length = img_ft_len
        self.max_utt_length = max_utt_len
        self.encoder = BertEncoder(config)
        self.hidden_size = config.hidden_size
        if req_text:
            self.fc_text = nn.Linear(self.text_feature_length, self.hidden_size)
        if req_aud:
            self.fc_audio = nn.Linear(self.audio_feature_length, self.hidden_size)
        if USE_IMAGE:
            self.fc_img = nn.Linear(self.img_feature_length, self.hidden_size)
        self.total_mod = 0
        if req_text:
            self.total_mod += 1
        if req_aud:
            self.total_mod += 1
        if USE_IMAGE:
            self.total_mod += 1
        self.concat1 = nn.Linear(self.hidden_size*self.total_mod, 4)
        self.dropout = nn.Dropout(0.2)
        #self.bn = nn.BatchNorm1d(self.max_utt_length*self.total_mod)

    def forward(self, batch_dict, text_flag, audio_flag, image_flag):
        if text_flag:
            batch_dict['text'] = batch_dict['text'].to(device)
            text_trans = batch_dict['text'].transpose(1, 2)
            text_final = self.fc_text(text_trans.float())
            text_final = self.dropout(text_final)
            num_examples = text_final.shape[0]
        if audio_flag:
            batch_dict['aud'] = batch_dict['aud'].to(device)  
            audio_trans = batch_dict['aud'].transpose(1, 2).float()
            audio_final = self.fc_audio(audio_trans)
            audio_final = self.dropout(audio_final)
            num_examples = audio_final.shape[0]
        if image_flag:
            batch_dict['img'] = batch_dict['img'].to(device)
            img_trans = batch_dict['img'].transpose(1, 2)
            img_final = self.fc_img(img_trans.float())
            img_final = self.dropout(img_final)
            num_examples = img_final.shape[0]
        encoder_inputs = torch.zeros((num_examples, self.max_utt_length*self.total_mod, self.hidden_size))
        atten = batch_dict["length"].float().to(device)
        if self.total_mod == 1:
            if text_flag:
                encoder_inputs = text_final
            if USE_IMAGE:
                encoder_inputs = img_final
            if audio_flag:
                encoder_inputs = audio_final
            attention_mask = atten
        elif self.total_mod == 2:
            if text_flag and USE_IMAGE:
                encoder_inputs = torch.cat([text_final, img_final], dim = 1)
            if text_flag and audio_flag:
                encoder_inputs = torch.cat([text_final, audio_final], dim = 1)
            if USE_AUDIO and USE_IMAGE:
                encoder_inputs = torch.cat([audio_final, img_final], dim = 1)
            attention_mask = torch.cat([atten, atten], dim=1)
        else:
            encoder_inputs = torch.cat(
                [
                    text_final,#20*768
                    audio_final,#50*768
                    img_final
                ],
                dim=1,
            )
            attention_mask = torch.cat([atten, atten, atten], dim=1)
        to_seq_length = attention_mask.size(1)
        from_seq_length = int(to_seq_length)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        encoder_out = self.encoder(encoder_inputs, extended_attention_mask,
                                       head_mask = head_mask)[0]
        #encoder_out = encoder_out_all[0]
        num_examples = encoder_out.size(0)
        total_mod = self.total_mod
        l = self.max_utt_length
        encoded_outputs = torch.zeros((num_examples, l,
                                self.hidden_size*total_mod)).to(device)
        ind = self.hidden_size
        j = 0
        for i in range(num_examples):
            if total_mod == 1:
                encoded_outputs[i, :, 0:ind] = encoder_out[i, :l, :]
            elif total_mod == 2:
                encoded_outputs[i, :, 0:ind] = encoder_out[i, :l, :]
                encoded_outputs[i, :, ind:2*ind] = encoder_out[i, l:2*l, :]
            else:
                encoded_outputs[i, :, 0:ind] = encoder_out[i, :l, :]
                encoded_outputs[i, :, ind:2*ind] = encoder_out[i, l:2*l, :]
                encoded_outputs[i, :, 2*ind:] = encoder_out[i, 2*l:, :]
        encoded_outputs = self.dropout(encoded_outputs)
        concat_out_1 = self.concat1(encoded_outputs)
        #final_output = self.concat2(concat_out_1)

        return encoded_outputs, concat_out_1

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, layers, bidirectional_flag):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        if bidirectional_flag == True:
          self.fc = nn.Linear(2*hidden_dim, output_dim)
        else:
          self.fc = nn.Linear(hidden_dim, output_dim)
        #self.fc1 = nn.Linear(200, output_dim)
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
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    config.num_hidden_layers = NUM_HIDDEN_LAYERS
    config.num_attention_heads = NUM_ATTENTION_HEADS
    config.hidden_size = HIDDEN_SIZE_TRANS
    for fold in range(1):
        final_val_loss = 999999
        logger.info(f"Running fold {fold}")
        train_loader = train_data[fold]
        val_loader = val_data[fold]
        model = MMT(config, HIDDEN_SIZE, GRU_DIM*2, IMAGE_DIM, max_length)
        model.to(device)
        count_parameters(model)
        if USE_TEXT:
            text_model = GRUModel(TEXT_DIM, GRU_DIM, 4, 0.2, 2, True).double()
            text_model.to(device)
            checkpoint_text = torch.load(os.path.join(unimodal_folder, 'best_model_text'+str(fold)+'.tar'))
            text_model.load_state_dict(checkpoint_text['model_state_dict'])
            text_model.eval()
        if USE_AUDIO:
            config_audio = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
            config_audio.num_hidden_layers = 3
            config_audio.num_attention_heads = 12
            config_audio.hidden_size = 60
            aud_model = MultiModalTransformer(config_audio, AUDIO_DIM, TEXT_DIM, IMAGE_DIM, max_length, True, False)
            aud_model.to(device)
            checkpoint_aud = torch.load('unimodal_trans_audio'+str(fold)+'.tar')
            aud_model.load_state_dict(checkpoint_aud['model_state_dict'])
            aud_model.eval()
        if USE_IMAGE:
            img_model = GRUModel(IMAGE_DIM, GRU_DIM, 4, 0.2, 2, True).double()
            img_model.to(device)
            checkpoint_img = torch.load(os.path.join(unimodal_folder, 'best_model_img'+str(fold)+'.tar'))
            img_model.load_state_dict(checkpoint_img['model_state_dict'])
            img_model.eval()

        optimizer = Adam(model.parameters(), lr=base_lr)
        for e in range(EPOCHS):
            tot_loss, tot_acc = 0.0, 0.0
            model.train()
            for ind, train_dic in enumerate(train_loader):
                model.zero_grad()
                if USE_IMAGE:
                    inp = train_dic['img'].permute(0, 2, 1).double()
                    train_dic['img'], _ = img_model.forward(inp.to(device))
                if USE_AUDIO:
                    train_dic['aud'], _ = aud_model(train_dic, False, True, False)
                    train_dic['aud'] = train_dic['aud'].transpose(1, 2)
                if USE_TEXT:
                    inp = train_dic['text'].permute(0, 2, 1).double()
                    train_dic['text'], _ = text_model.forward(inp.to(device))
                    train_dic['text'] = train_dic['text'].transpose(1,2)

                _, out = model.forward(train_dic, USE_TEXT, USE_AUDIO, USE_IMAGE)
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
                    if USE_IMAGE:
                        inp = val_dic['img'].permute(0, 2, 1).double()
                        val_dic['img'], _ = img_model.forward(inp.to(device))
                    if USE_AUDIO:
                        val_dic['aud'], _ = aud_model(val_dic, False, True, False)
                        val_dic['aud'] = val_dic['aud'].transpose(1,2)
                    if USE_TEXT:
                        inp = val_dic['text'].permute(0, 2, 1).double()
                        val_dic['text'], _ = text_model.forward(inp.to(device))
                        val_dic['text'] = val_dic['text'].transpose(1,2)

                    _, val_out = model.forward(val_dic, USE_TEXT, USE_AUDIO, USE_IMAGE)
                    val_dic['target'][val_dic['target'] == -1] = 4
                    val_acc += compute_accuracy(val_out.cpu(), val_dic).item()
                    val_loss += compute_loss(val_out.to(device), val_dic).item()
                if val_loss < final_val_loss:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                'hybrid_best_model_' + str(fold) + '.tar')
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
