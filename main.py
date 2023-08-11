import heapq

from matplotlib import pyplot as plt
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from dataset import *
import torch.nn as nn
import torchvision.models as models

# Directory for images and captions (From flickr30k dataset)
images_dir = "images/"
test_dir = "flickr30k_images/"
captions_file = "captions.txt"
captions_test = "test2.csv"

# Constants
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_LAYERS = 2
CLIP = 5
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 256
BATCH_SIZE = 32
CAPTION_LENGTH = 15
NUM_EPOCHS = 70
LEARNING_RATE = 0.0003

# Image preprocessing and normalization
image_transform = transforms.Compose([
    transforms.Resize((356, 356)),
    transforms.RandomCrop((IMAGE_HEIGHT, IMAGE_HEIGHT)),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# Load the Flickr30k dataset
dataset_train = ImageCaptionDataset(images_dir, captions_file, CAPTION_LENGTH, transform=image_transform)
# dataset_test = ImageCaptionDataset(images_dir, captions_test, CAPTION_LENGTH, transform=image_transform)
vocab_size = len(dataset_train.vocab)
# dataset_test.vocab = dataset_train.vocab

# Prepare the DataLoaders
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
# validation_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.num_captions = 5

    def forward(self, features, captions):
        # features: Image features (batch_size, embedding_size)
        # captions: Ground truth captions for teacher forcing (batch_size, number_captions, caption_length)

        # Remove EOS
        input_captions = captions[:, :-1]

        # Embed the input captions
        embedded_captions = self.dropout(self.embedding(input_captions))

        features = features.unsqueeze(1)

        embeddings = torch.cat((features, embedded_captions), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)

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

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # x shape (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH) -> image

        # Extract image features using ResNet
        features = self.resnet(x)

        # Global average pooling to obtain a fixed-size vector
        features = self.global_avg_pool(features)

        # Flatten the features
        features = features.view(features.size(0), -1)

        # Project to the desired output size (embedding dimension)
        features = self.dropout(self.fc(features))

        return features

class Image2Words(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers):
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

    # Generate captions for test data
    def generate_captions(self, images):
        model.eval()


        with torch.no_grad():
            captions = []

            # Initialize the hidden state and cell state of the LSTM
            hidden = torch.zeros(model.decoder.num_layers, BATCH_SIZE, model.decoder.hidden_dim).to(DEVICE)
            cell = torch.zeros(model.decoder.num_layers, BATCH_SIZE, model.decoder.hidden_dim).to(DEVICE)

            features = model.encoder(images)

            states = None
            # states = None
            for i in range(CAPTION_LENGTH):
                outputs, states = self.decoder.lstm(features,states)

                output = model.decoder.fc(outputs)

                _, predicted_word = torch.max(output, dim=1)

                predicted_word = predicted_word.unsqueeze(1)
                real_words = self.convertWords(predicted_word)
                captions.append(real_words)
        captions = list(zip(*captions))

        return captions

    # Beam Search
    def beam_search(self, images, beam_width=10, max_length=CAPTION_LENGTH):
        model.eval()
        id_to_word = {idx: word for idx, word in enumerate(dataset_train.vocab.get_itos())}

        ## TODO i dont' think this works with more than 1 batch so fix that
        with torch.no_grad():

            # Compute the features
            features = model.encoder(images)

            # Set hidden states to None since they don't exist at the beginning
            states = None

            # Start with a sequence containing only the SOS token
            start_token = torch.tensor([dataset_train.vocab["<SOS>"]], dtype=torch.long).to(DEVICE)
            sequences = [(0.0, [start_token], states)]

            # Perform beam search
            for _ in range(max_length):
                candidates = []
                for score, sequence, states in sequences:
                    # Perform a forward pass through the LSTM
                    output, new_states = self.decoder.lstm(features, states)

                    # Get the output prediction from the fully connected layer
                    output = self.decoder.fc(output)

                    # Apply log-softmax to convert to log-probabilities
                    log_probs = nn.functional.log_softmax(output, dim=-1)

                    # Get the top K words and their corresponding log-probabilities
                    top_log_probs, top_words = log_probs.squeeze(1).topk(beam_width)

                    for i in range(beam_width):
                        word = top_words[0, i]
                        word2 = word.item()

                        val = id_to_word[word2]

                        if (val == "<SOS>"):
                            continue

                        new_score = score - top_log_probs[0, i].item()

                        # Check if we're at the end
                        if word == dataset_train.vocab["<EOS>"]:
                            candidates.append((new_score, sequence, states))
                        else:
                            # Otherwise we keep adding onto it
                            new_sequence = sequence + [word]
                            candidates.append((new_score, new_sequence, new_states))

                # Select the top beam_width sequences
                sequences = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])


            # Get the best sequence
            best_sequence = sequences[0][1]

            # Convert the word indices to words
            predicted_words = self.convertWords(torch.tensor(best_sequence, dtype=torch.long).unsqueeze(0))

            model.train()

            return predicted_words

    # Converts the integer words into string words
    def convertWords(self, words):
        # words is tensor shape (batch_size, word_size)
        id_to_word = {idx: word for idx, word in enumerate(dataset_train.vocab.get_itos())}

        words = [[id_to_word[word_idx.item()] for word_idx in test] for test in words]
        return words

    # Specifically for y_test
    def convertWordsY(self, words):
        # words is tensor shape (batch_size, word_size)
        id_to_word = {idx: word for idx, word in enumerate(dataset_train.vocab.get_itos())}

        # Convert word indices to words for each caption in the batch
        batch_words = []
        for batch_idx in range(words.size(0)):
            caption = [id_to_word[word_idx.item()] for word_idx in words[batch_idx]]

            batch_words.append(caption)

        return batch_words
# Makes a visual representation of everything
def imagine(images, captions_predicted, captions_correct):
    images = images.squeeze(0)

    # Fixing the format of everything
    captions_predicted = list(captions_predicted[0])

    # Check if the input is in the correct format
    assert images.shape == (3, 224, 224), "Invalid shape for images"
    assert isinstance(captions_predicted[0], str), "captions_predicted should be a list of strings"
    assert len(captions_correct) == 1, "Invalid size for captions_correct"

    # Create a new figure
    plt.figure(figsize=(8, 8))

    images_cpu = images.cpu()

    # Plot the RGB image
    plt.subplot(1, 1, 1)
    plt.imshow(images_cpu.permute(1, 2, 0))
    plt.axis('off')

    # Create a sentence from the predicted captions and plot it
    predicted_sentence = " ".join(captions_predicted)
    plt.title("Predicted Caption:\n" + predicted_sentence, fontsize=12)

    # Create a sentence from the correct captions and plot it
    caption_correct = captions_correct[0]
    correct_caption_str = " ".join(caption_correct)

    plt.text(0, 250, "Correct Captions:\n" + correct_caption_str, fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()

    print('f')


# training

# model being used
model = Image2Words(EMBEDDING_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS)
model.to(DEVICE)

model.load_state_dict(torch.load("updated4.pth"))

# criterion and optimizer for training
criterion = nn.CrossEntropyLoss(ignore_index=dataset_train.vocab["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')



# training
# for epoch in range(NUM_EPOCHS):
#     total_loss = 0
#     model.train()
#
#     loop = tqdm(train_loader, leave=True)
#
#     for batch, (x, y) in enumerate(loop):
#         x, y = x.to(DEVICE), y.to(DEVICE)
#
#         # forward prop
#         out = model(x,y)
#
#         # Flatten both out and y in order to work
#         out = out.view(-1, vocab_size)
#         y = y.view(-1)
#
#         y = y.long()
#
#         # loss function
#         loss = criterion(out, y)
#
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Help prevent exploding gradient problem
#         nn.utils.clip_grad_norm_(model.parameters(), CLIP)
#
#         # update gradients
#         optimizer.step()
#
#         # accumulate loss
#         total_loss += loss.item()
#
#         if batch == 499:
#
#             # use only 1 image in the batch for ease
#             x_test = x[1, :, :].unsqueeze(0)
#
#             # Generate captions for the images
#             generated_captions = model.beam_search(x_test)
#
#             print(generated_captions)
#
#             model.train()
#
#
#     average_loss = total_loss / len(train_loader)
#     print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {average_loss:.4f}")

#
#     # Validation
#
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for batch, (x_val, y_val) in enumerate(validation_loader):
#             x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
#
#             # Forward prop
#             out_val = model(x_val, y_val)
#
#             # Flatten both out_val and y_val in order to work
#             out_val = out_val.view(-1, vocab_size)
#             y_val = y_val.view(-1)
#
#             y_val = y_val.long()
#
#             # Calculate validation loss
#             val_loss += criterion(out_val, y_val).item()
#
#             if batch == 0:
#                 # use only 1 image in the batch for ease
#                 x_test = x_val[1, :, :].unsqueeze(0)
#
#                 # Generate captions for the images
#                 generated_captions = model.beam_search(x_test)
#
#                 model.eval()
#
#                 print(generated_captions)
#
#
#     average_val_loss = val_loss / len(validation_loader)
#
#     print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation Loss: {average_val_loss:.4f}")

# Testing
with torch.no_grad():
    for batch, (x_test, y_test) in enumerate(train_loader):
        x_test = x_test.to(DEVICE)

        # use only 1 image in the batch for ease
        x_test = x_test[1, :, :].unsqueeze(0)
        y_test = y_test[1, :].unsqueeze(0)


        # Generate captions for the images
        generated_captions = model.beam_search(x_test)

        # Correct captions for the images
        correct_captions = model.convertWordsY(y_test)
        print(generated_captions)
        print(correct_captions)

        # Plot everything
        imagine(x_test, generated_captions, correct_captions)

# Save the trained model if desired
torch.save(model.state_dict(), "updated4.pth")
