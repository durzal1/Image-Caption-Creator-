import torch
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
from transformers import BertTokenizer


# Directory for images and captions (From COCO dataset)
images_dir = "train2014/"
captions_file = "captions_train2014.json"

# Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
CAPTION_LENGTH = 10

# Image preprocessing and normalization
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load the MS COCO dataset
dataset = CocoCaptions(root=images_dir, annFile=captions_file, transform=image_transform)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare the DataLoader with collate_fn for padding
def collate_fn(batch):
    images, captions = zip(*batch)

    # Not sure why I need this, but it breaks without it.
    # All images should already be the same size
    images_padded = torch.nn.utils.rnn.pad_sequence(images, batch_first=True)

    ##TODO FIGURE OUT
    # Convert each batch of captions to tensors
    captions_tensors = []
    for caption_batch in captions:
        caption_tensors = []
        for caption in caption_batch:
            # Assuming you have a tokenizer function to tokenize the sentences (e.g., spaCy or NLTK)
            # You should replace this tokenizer function with the one you use in your specific case.
            tokens = tokenizer(caption)
            caption_tensors.append(torch.tensor(tokens))
        captions_tensors.append(torch.stack(caption_tensors))

    # Make them all the same size
    captions_padded = torch.nn.utils.rnn.pad_sequence(captions_tensors, batch_first=True)

    print('f')
    # return images_padded, captions_tokenized

# Prepare the DataLoader
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


for batch_idx, (x, y) in enumerate(data_loader):
    print("Images shape:", x.shape)
    print("Captions:", y)