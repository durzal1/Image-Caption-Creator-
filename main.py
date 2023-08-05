from torchvision import transforms
import torch.optim as optim
from dataset import *
import torch.nn as nn
import torchvision.models as models

# Directory for images and captions (From COCO dataset)
images_dir = "flickr30k_images/"
captions_file = "results.csv"

# Constants
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_LAYERS = 2
CLIP = 5
HIDDEN_SIZE = 64
EMBEDDING_SIZE = 64
BATCH_SIZE = 32
CAPTION_LENGTH = 10
NUM_EPOCHS = 20
LEARNING_RATE = 0.004

# Image preprocessing and normalization
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# Load the Flickr30k dataset
dataset = ImageCaptionDataset(images_dir, captions_file, CAPTION_LENGTH, transform=image_transform)
vocab_size = len(dataset.vocab)

# Prepare the DataLoader
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

dataset.__getitem__(1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features: Image features (batch_size, embedding_size)
        # captions: Ground truth captions for teacher forcing (batch_size, number_captions, caption_length)

        input_captions = captions

        # Embed the input captions
        embedded_captions = self.dropout(self.embedding(input_captions))

        # Initialize the hidden state and cell state of the LSTM
        hidden = torch.zeros(self.num_layers, BATCH_SIZE * self.num_captions, self.hidden_dim).to(features.device)
        cell = torch.zeros(self.num_layers, BATCH_SIZE * self.num_captions, self.hidden_dim).to(features.device)

        # Combine features with each embedded caption and pass through LSTM
        lstm_outputs_list = []
        for i in range(num_captions):
            lstm_input = torch.cat((features[:, i, :], embedded_captions[:, i, :]), dim=1)
            lstm_input = lstm_input.unsqueeze(1)  # Add time step dimension (sequence length = 1)
            lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            lstm_outputs_list.append(lstm_output)

        # Combine LSTM outputs for all captions
        lstm_outputs = torch.cat(lstm_outputs_list, dim=1)

        # Pass the inputs through the LSTM
        lstm_outputs, _ = self.lstm(inputs)

        # Final output prediction at each time step
        outputs = self.fc(lstm_outputs)

        # Output shape -> (Batch, 5, caption_length, vocab_size)
        return outputs

class Encoder(nn.Module):
    def __init__(self, output_size):
        super(Encoder, self).__init__()

        resnet = models.resnet50(pretrained=True)

        # Remove the last two layers (avgpool and fc)
        # Since we're only using the resnet for the features
        modules = list(resnet.children())[:-2]

        # Create the encoder layers using the res net
        self.resnet = nn.Sequential(*modules)

        # Add a global average pooling layer to convert the feature map into a fixed-size vector
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a linear layer to project the output to the desired output size (embedding dimension)
        self.fc = nn.Linear(resnet.fc.in_features, output_size)

    def forward(self, x):
        # x shape (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH) -> image

        # Extract image features using ResNet
        features = self.resnet(x)

        # Global average pooling to obtain a fixed-size vector
        features = self.global_avg_pool(features)

        # Flatten the features
        features = features.view(features.size(0), -1)

        # Project to the desired output size (embedding dimension)
        features = self.fc(features)

        # Features -> (batch, caption_length, embedding) dim
        features = features.unsqueeze(1).expand(-1, 5, -1)

        return features

class Image2Words(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers ):
        super(Image2Words, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder(self.vocab_size, self.embedding_size,
                               self.hidden_size, self.num_layers)

    def forward(self, images, captions):
        # Now given the images we will encode it then decode it
        # images - > (Batch, 3, WIDTH, HEIGHT)

        # x -> (Batch, embedding_dim)
        x = self.encoder(images)

        x = self.decoder(x, captions)

        return x


# training

# model being used
model = Image2Words(EMBEDDING_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS)
model.to(DEVICE)

# criterion and optimizer for training
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# training
for epoch in range(NUM_EPOCHS):
    total_loss = 0

    for batch, (x, y) in enumerate(data_loader):

        x, y = x.to(DEVICE), y.to(DEVICE)

        # forward prop
        out = model(x,y)

        # loss function
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()

        # Help prevent exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        # update gradients
        optimizer.step()

        # accumulate loss
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {average_loss:.4f}")