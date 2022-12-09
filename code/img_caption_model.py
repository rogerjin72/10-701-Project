import torch
from torch import nn
import hyperparams as hp
from caption_model import CaptionModel, Predictor
from transformers import ViTFeatureExtractor, ViTModel

class ImageCaptionModel(nn.Module):
    def __init__(self, model_path):
        '''
        model_path: str
            Path to the caption model to load
        '''
        super().__init__()

        # Load image encoder (on cpu)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit.eval()

        # Text decoder
        self.txt_decoder = CaptionModel()
        self.predictor = Predictor(self.txt_decoder)
        checkpoint = torch.load(model_path)
        self.txt_decoder.load_state_dict(checkpoint['model_state_dict'])

        # Forward pass prefix embeddings (saved during forward pass)
        self.prefix_embed = None
    
    def forward(self, img):
        '''
        img: torch.Tensor
            224x224x3 RBG image for which caption is generated
        return: str
            Caption of the image
        '''
        img_features = self.feature_extractor(img.long(), return_tensors="pt")
        img_encoding = self.vit(**img_features)
        img_encoding = img_encoding.last_hidden_state
        text = self.predictor.top_k_predict(img_encoding, limit = 20, k = 3)
        return text

    def attach_hook(self):
        '''
        Register a forward hook to save intermediate activations
        '''
        self.hook = self.txt_decoder.align.register_forward_hook(self.hook)
    
    def hook(self, model, input, output):
        self.prefix_embed = output.detach()

        