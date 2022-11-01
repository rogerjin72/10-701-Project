import torch
from torch import nn
import hyperparams as hp
from caption_model import CaptionModel, Predictor
from torchvision.models import resnet50, ResNet50_Weights

class ImageCaptionModel(nn.Module):
    def __init__(self, model_path):
        '''
        model_path: str
            Path to the caption model to load
        '''
        super().__init__()

        # Load image encoder (on cpu)
        resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        resnet50_layers = list(resnet.children())
        self.img_encoder = torch.nn.Sequential(*resnet50_layers[:-1])
        self.img_encoder.eval()

        # Text decoder
        self.txt_decoder = CaptionModel()
        self.predictor = Predictor(self.txt_decoder)
        checkpoint = torch.load(model_path)
        self.txt_decoder.load_state_dict(checkpoint['model_state_dict'])
    
    def forward(self, img):
        '''
        img: torch.Tensor
            224x224x3 RBG image for which caption is generated
        return: str
            Caption of the image
        '''
        img_encoding = self.img_encoder(img)[:, :, 0, 0]
        img_encoding = img_encoding.to(hp.DEVICE)
        text = self.predictor.top_k_predict(img_encoding, limit = 20, k = 3)
        return text
        