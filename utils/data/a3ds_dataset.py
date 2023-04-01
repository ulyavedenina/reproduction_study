import os
import torch
import json
import h5py
import spacy
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer


nlp = spacy.load('en_core_web_sm')

class A3DSDataset(data.Dataset):
    dataset_prefix = '3d'
    # Some variables to initialize during the creation of the constructor. Here the h5py file is opened and loaded.
    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), 'data/A3DS')
        self.start_token="START"
        self.end_token="END"
        self.unk_token="UNK"
        self.unk_index=1
        self.pad_index=0
        
        # Load image features and labels from h5 file
        with h5py.File(os.path.join(self.data_dir, '3dshapes.h5'), 'r') as f:
            self.images = f['images'][...]
            self.labels = f['labels'][...]
            
        # Load captions from json file
        with open(os.path.join(self.data_dir, '3dshapes_captions_short.json'), 'r') as f:
            self.captions = json.load(f)

        # Determine the number of unique labels in the dataset
        self.num_classes = len(set(map(tuple, self.labels)))
        # Calculate the vocabulary
        self.vocab = set()
        for caption in self.captions:
            self.vocab.update(set(caption))
        self.vocab = list(self.vocab)
        self.vocab_size = len(self.vocab)

        self.tokenizer = get_tokenizer("spacy")

        # This is needed for the training the pragmatic isic. We didnt use it during the training the sentence classifier
        # features = torch.load(os.path.join(self.data_dir, '3dshapes_all_ResNet_features.pt'))
        # self.input_size = features.shape[1]


    def __len__(self):
        return len(self.images)    
    

    """ This function is defining the behavior of the object when an item is accessed by index."""  
    def __getitem__(self, index):
        # Retrieve the image data associated with the specified index from the object's images attribute 
        # and convert it to a PyTorch tensor using the torch.from_numpy() function. This allows the image
        # data to be used with PyTorch's deep learning functionality.
        image = torch.from_numpy(self.images[index])
        # Retrieve the label associated with the specified index from the object's labels attribute and convert
        # it to a PyTorch tensor using the torch.tensor() function. This also allows the label to be used with PyTorch's deep
        # learning functionality.
        label = torch.tensor(self.labels[index])
        # Check if caption exists for given index
        caption = self.captions.get(str(index), '')  
        return image, label, caption

    
    def __len__(self):
        return len(self.images)
    

    nlp = spacy.load('en_core_web_sm')

    """This code defines a collate function for a PyTorch DataLoader. The purpose of this function
    is to take a batch of data and organize it into a format that can be easily fed into a neural network."""
    def collate_fn(self, data):
        """Collate function to be used with DataLoader."""
        # Sort the data by caption length in descending order, so that captions with similar lengths are grouped together.
        data.sort(key=lambda x: len(x[2]), reverse=True)
        # Unzip the list of tuples into three separate lists: images, labels, and captions.
        images, labels, captions = zip(*data)


        # Stack images and labels into a tensor
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        # Convert captions to lists of tokens and add start and end tokens
        captions = [
            [self.start_token] + self.tokenizer(" ".join(caption)) + [self.end_token]
            for caption in captions
        ]
        # Check if token is instance of Token, convert to text and add to list
        captions = [[token.text if isinstance(token, spacy.tokens.token.Token) else token for token in caption] for caption in captions]

        # Convert tokens to tensors using vocabulary
        captions = [
            [
                self.vocab.index(token) if token in self.vocab else self.unk_index
                for token in caption
            ]
            for caption in captions
        ]
        captions = [torch.tensor(caption[:-1]).clone().detach() for caption in captions]
        

        # Pad the sequences of caption indices with the padding index so that they are all the same length.
        word_targets = pad_sequence(
            [torch.tensor(caption[:-1]) for caption in captions],
            batch_first=True,
            padding_value=self.pad_index
        )
        # Sort the captions and lengths by length in descending order, so that they can be packed into 
        # a PackedSequence object. This is an optimization that allows the model to skip over padded elements during training.
        lengths = torch.tensor([len(caption) - 1 for caption in captions])
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_captions = [captions[i] for i in sorted_indices]
        sorted_word_targets = word_targets[sorted_indices]

        
        # Convert the labels into a tensor.
        labels = torch.tensor(labels)
        
        # Return the sorted_word_targets, sorted_lengths, and labels as a tuple. sorted_word_targets is a tensor of padded caption indices,
        # sorted_lengths is a tensor of caption lengths in descending order, and labels is a tensor of image labels.
        return sorted_word_targets, sorted_lengths, labels
