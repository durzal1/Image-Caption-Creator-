import torch
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import *
import torch.nn as nn

# Directory for images and captions (From COCO dataset)
images_dir = "flickr30k_images/"
captions_file = "results.csv"

# Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
CAPTION_LENGTH = 25

# Image preprocessing and normalization
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# Load the Flickr30k dataset
dataset = ImageCaptionDataset(images_dir, captions_file, CAPTION_LENGTH, transform=image_transform)

# Prepare the DataLoader
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

dataset.__getitem__(1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        # features: Image features (output of the encoder)
        # captions: Ground truth captions for teacher forcing (batch_size, caption_length)

        # Remove the last token from captions (input to LSTM) to get the input sequence
        input_captions = captions[:, :-1]

        # Embed the input captions
        embedded_captions = self.embedding(input_captions)

        # Concatenate image features with embedded captions
        inputs = torch.cat((features.unsqueeze(1), embedded_captions), dim=1)

        # Pass the inputs through the LSTM
        lstm_outputs, _ = self.lstm(inputs)

        # Final output prediction at each time step
        outputs = self.fc(lstm_outputs)

        return outputs

class Encoder(nn.Module):
    def __init__(self):
        pass

