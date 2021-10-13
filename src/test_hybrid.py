import logging
from src.model import MyDataset
from src.model import MultiModalTransformer as MMT
from src.model_hybrid import GRUModel, compute_accuracy
from src.model_hybrid import MultiModalTransformer
from config import NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS, HIDDEN_SIZE
from config import AUDIO_DIM, TEXT_DIM, IMAGE_DIM, GRU_DIM, HIDDEN_SIZE_TRANS
from config import USE_TEXT, USE_AUDIO, USE_IMAGE, unimodal_folder
from pytorch_transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                                BertEncoder, BertLayerNorm,
                                                BertPreTrainedModel)
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import os
torch.backends.cudnn.enabled = False
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def test_model(PATH, test_imgs, test_aud, test_text, test_labels, test_lengths, max_length):
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    config.num_hidden_layers = NUM_HIDDEN_LAYERS
    config.num_attention_heads = NUM_ATTENTION_HEADS
    config.hidden_size = HIDDEN_SIZE_TRANS
    test_dataset = MyDataset(test_imgs, test_aud, test_text, test_labels,
                            test_lengths)
    test_size = test_imgs.shape[0]
    indices = list(range(test_size))
    test_sampler = SubsetRandomSampler(indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    test_loader = DataLoader(test_dataset,
                            sampler = test_sampler,
                            batch_size=64,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False,
                            )
    test_results = dict()
    with torch.no_grad():
        for fold in range(1):
            test_model = MMT(config, HIDDEN_SIZE, GRU_DIM*2, IMAGE_DIM, max_length)
            model_path = PATH.replace("model", "model_"+str(fold))
            checkpoint = torch.load(model_path)
            test_model.load_state_dict(checkpoint['model_state_dict'])
            test_model.to(device)
            test_model.eval()
            
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
                img_model.eval()
                checkpoint_img = torch.load(os.path.join(unimodal_folder, 'best_model_img'+str(fold)+'.tar'))
                img_model.load_state_dict(checkpoint_img['model_state_dict'])
            for ind, test_dic in enumerate(test_loader):
                if USE_IMAGE:
                    inp = train_dic['img'].permute(0, 2, 1).double()
                    train_dic['img'], _ = img_model.forward(inp.to(device))
                if USE_AUDIO:
                    test_dic['aud'], _ = aud_model(test_dic, False, True, False)
                    test_dic['aud'] = test_dic['aud'].transpose(1, 2)
                if USE_TEXT:
                    inp = test_dic['text'].permute(0, 2, 1).double()
                    test_dic['text'], _ = text_model.forward(inp.to(device))
                    test_dic['text'] = test_dic['text'].transpose(1,2)


                _, test_out = test_model.forward(test_dic, USE_TEXT, USE_AUDIO, USE_IMAGE)
                test_dic['target'][test_dic['target'] == -1] = 4
                test_results[fold] = test_out
            acc_fold = compute_accuracy(test_out.cpu(), test_dic).item()
            logger.info(f"Accuracy of fold-  {fold}- {acc_fold}")
