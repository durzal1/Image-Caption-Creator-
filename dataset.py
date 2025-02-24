import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, caption_file, caption_length, transform=None):
        self.image_path = image_paths
        self.caption_file = caption_file
        self.transform = transform
        self.caption_length = caption_length

        self.tokenizer = get_tokenizer('basic_english')


        self.SetUpCaptions()

    def SetUpCaptions(self):
        captions = pd.read_csv(self.caption_file)
        self.image_names = captions["image"]
        self.comments = captions["caption"]

        num_rows, num_columns = captions.shape
        #
        # # All the data we need
        # self.image_names = []
        # self.comment_numbers = []
        # self.comments = []
        #
        # For the vocab
        data = []

        # Get Data
        for ind in range(0,num_rows):

            # In order to train faster we're only using the first 80000 values
            # if ind == 80000:
            #     break
            #
            # # only get every 5th image
            # if ind % 5 == 0:
            #     image_name = captions.loc[ind,"image_name"]
            #     self.image_names.append(image_name)
            #
            # comment_number = captions.loc[ind," comment_number"]
            # if (self.caption_file == "test2.csv"):
            #     comment = captions.loc[ind," comment,,,"]
            # else:
            #     comment = captions.loc[ind," comment"]

            # Broken indices for some reason
            # if ind == 19999:
            #     comment = "a dog runs across the grass"
            #
            # if ind == 402 and num_rows == 420:
            #     comment_number = 2
            #     comment = "People outside"

            data.append(self.tokenizer(self.comments[ind]))

        self.vocab = build_vocab_from_iterator(data, specials= ["<PAD>", "<UNK>", "<SOS>", "<EOS>"], min_freq=3)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image = self.image_names[index]


        caption = self.comments[index]

        # Tokenize the text
        tokens = self.tokenizer(caption)

        # Create the numerical representation
        numerical_representation = [self.vocab["<SOS>"]]

        for token in tokens:
            if token not in self.vocab:
                numerical_representation.append(self.vocab["<UNK>"])
            else:
                numerical_representation.append(self.vocab[token])

        numerical_representation.append(self.vocab["<EOS>"])
        numerical_tensor = torch.tensor(numerical_representation)

        numerical_representation = numerical_tensor.squeeze(0)

        # Compute the padding amount
        padding_amount = max(0, self.caption_length - numerical_representation.size(0))

        # Define the padding tuple for the specified dimension
        padding = [0] * (2 * len(numerical_representation.size()))
        padding[0 * 2 + 1] = padding_amount

        # Apply padding to the tensor
        padded_representation = F.pad(numerical_representation, padding, value=0)

        padded_representation = padded_representation[:self.caption_length]

        specific_image_path = os.path.join(self.image_path, image)
        image = Image.open(specific_image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, padded_representation