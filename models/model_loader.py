import torch

from .lrcn import LRCN
from .gve import GVE
from .sentence_classifier import SentenceClassifier

class ModelLoader:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def lrcn(self):
        # LRCN arguments
        pretrained_model = self.args.pretrained_model
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)

        layers_to_truncate = self.args.layers_to_truncate

        lrcn = LRCN(pretrained_model, embedding_size, hidden_size, vocab_size,
                layers_to_truncate)

        return lrcn

    def gve(self):
        # Make sure dataset returns labels
        #self.dataset.set_label_usage(True)
        # GVE arguments
        # Initialize some variables and then call the GVE class which contains the forward function, and the number of layers etc.
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        input_size = self.dataset.input_size
        num_classes = self.dataset.num_classes

        sc = self.sc()
        if torch.cuda.is_available():
            sc.load_state_dict(torch.load(self.args.sc_ckpt))
        else:
            sc.load_state_dict(torch.load(self.args.sc_ckpt, map_location='cpu'))

        for param in sc.parameters():
            param.requires_grad = False
        sc.eval()

        gve = GVE(input_size, embedding_size, hidden_size, vocab_size, sc,
                num_classes)

        if self.args.weights_ckpt:
            gve.load_state_dict(torch.load(self.args.weights_ckpt))

        return gve



    def sc(self):
        # comment line 59 because we didnt implement the method.
        # Make sure dataset returns labels
        # self.dataset.set_label_usage(True)
        # Sentence classifier arguments
        # Initialize some variables and then call the sentence_classifier class which contains the forward function, and the number of layers etc.
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        # also try this 
        #num_classes = self.dataset.len tou ids
        num_classes = self.dataset.num_classes

        sc = SentenceClassifier(embedding_size, hidden_size, vocab_size,
                num_classes)

        return sc
