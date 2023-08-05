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
        captions = pd.read_csv("results.csv", sep="|")

        num_rows, num_columns = captions.shape

        # All the data we need
        self.image_names = []
        self.comment_numbers = []
        self.comments = []

        # For the vocab
        data = []

        # Get Data
        for ind in range(0,num_rows):

            # In order to train faster we're only using the first 20000 values
            if ind == 20000:
                break

            # only get every 5th image
            if ind % 5 == 0:
                image_name = captions.loc[ind,"image_name"]
                self.image_names.append(image_name)

            comment_number = captions.loc[ind," comment_number"]
            comment = captions.loc[ind," comment"]

            # Broken index for some reason
            if ind == 19999:
                comment = "a dog runs across the grass"

            self.comment_numbers.append(comment_number)
            self.comments.append(comment)

            data.append(self.tokenizer(comment))

        self.vocab = build_vocab_from_iterator(data, specials= ["<PAD>", "<UNK>", "<SOS>", "<EOS>"], min_freq=1)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image = self.image_names[index]

        caption_tensor = torch.zeros((5, self.caption_length), dtype=torch.int32)


        for i in range(5):
            caption_index = index * 5 + i
            caption = self.comments[caption_index]

            # Tokenize the text
            tokens = self.tokenizer(caption)

            # Create the numerical representation
            numerical_representation = []

            for token in tokens:
                if token not in self.vocab:
                    numerical_representation.append(self.vocab["<UNK>"])
                else:
                    numerical_representation.append(self.vocab[token])
            numerical_tensor = torch.tensor(numerical_representation)

            numerical_representation = numerical_tensor.squeeze(0)

            # Compute the padding amount
            padding_amount = max(0, self.caption_length - numerical_representation.size(0))

            # Define the padding tuple for the specified dimension
            padding = [0] * (2 * len(numerical_representation.size()))
            padding[0 * 2 + 1] = padding_amount

            # Apply padding to the tensor
            padded_representation = F.pad(numerical_representation, padding, value=0)


            caption_tensor[i] = padded_representation[:self.caption_length]

        specific_image_path = os.path.join(self.image_path, image)
        image = Image.open(specific_image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, caption_tensor