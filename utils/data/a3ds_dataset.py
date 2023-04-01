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
    
      
    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        label = torch.tensor(self.labels[index])
        # Check if caption exists for given index
        caption = self.captions.get(str(index), '')  
        return image, label, caption

    
    def __len__(self):
        return len(self.images)
    

    nlp = spacy.load('en_core_web_sm')

    def collate_fn(self, data):
        """Collate function to be used with DataLoader."""
        # Sort data by caption length
        data.sort(key=lambda x: len(x[2]), reverse=True)
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
        

        # Pad sequences
        word_targets = pad_sequence(
            [torch.tensor(caption[:-1]) for caption in captions],
            batch_first=True,
            padding_value=self.pad_index
        )
        # Sort lengths in descending order
        lengths = torch.tensor([len(caption) - 1 for caption in captions])
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_captions = [captions[i] for i in sorted_indices]
        sorted_word_targets = word_targets[sorted_indices]

        

        labels = torch.tensor(labels)
        
        return sorted_word_targets, sorted_lengths, labels
