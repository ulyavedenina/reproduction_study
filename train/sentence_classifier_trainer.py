import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pdb

class SCTrainer:

    REQ_EVAL = False

    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.train = args.train
        self.logger = logger
        self.device = device

        model.to(self.device)

        # TODO: Implement checkpoint recovery
        if checkpoint is None:
            self.criterion = nn.CrossEntropyLoss()
            self.params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
            self.total_steps = len(data_loader)
            self.num_epochs = args.num_epochs
            self.log_step = args.log_step
            self.curr_epoch = 0

    def train_epoch(self):
        # Result is list of losses during training
        # and generated captions during evaluation
        result = []
        # A loop is executed where self.data_loader is iterated over, where each iteration generates a mini-batch of data consisting 
        # of word_targets, lengths, and labels.
        for i, (word_targets, lengths, labels) in enumerate(self.data_loader):
            # word_targets and labels are moved to a device (i.e., CPU or GPU) specified by self.device.
            word_targets = word_targets.to(self.device)
            labels = labels.to(self.device)
            # Prepare mini-batch dataset
            if self.train:
                # the train_step function is called with word_targets, labels, and lengths as arguments.
                loss = self.train_step(word_targets, labels, lengths)
                result.append(loss.data.item())

                step = self.curr_epoch * self.total_steps + i + 1
                # A scalar summary of the current batch loss value is logged to some logger object with the tag 'batch_loss'.
                self.logger.scalar_summary('batch_loss', loss.data.item(), step)

            else:
                # eval_step is called with word_targets, labels, and lengths as arguments.
                score = self.eval_step(word_targets, labels, lengths)
                result.append(score)

            # TODO: Add proper logging
            # Print log info
            # For every self.log_step number of iterations (i.e., every self.log_step-th iteration), 
            # some logging information is printed to the console.
            if i % self.log_step == 0:
                # The logging information consists of the current epoch and step number,
                # and optionally, the current loss value (if self.train is True) and its corresponding perplexity value
                print("Epoch [{}/{}], Step [{}/{}]".format(self.curr_epoch,
                    self.num_epochs, i, self.total_steps), end='')
                if self.train:
                    print(", Loss: {:.4f}, Perplexity: {:5.4f}".format(loss.data.item(),
                                np.exp(loss.data.item())), end='')
                print()

        
        self.curr_epoch += 1

        # If self.train is True, then a scalar summary of the average loss value over 
        # the current epoch is logged with the tag 'epoch_loss'.
        if self.train:
            self.logger.scalar_summary('epoch_loss', np.mean(result), self.curr_epoch)
        # Otherwise, the result list is summed along the 0-th axis, and then the ratio of the second 
        # and first elements of the summed result is calculated and stored in result.
        else:
            result = np.sum(result, axis=0)
            result = result[1] / result[0]
            print("Evaluation Accuracy: {}".format(result))


        return result


    def train_step(self, word_targets, class_labels, lengths):
        # Forward, Backward and Optimize
        self.model.zero_grad()
        outputs = self.model(word_targets, lengths)
        # Convert class_labels to tensor of size (batch_size,) because we want it to 
        # contain class labels for each example in the batch. 
        class_labels = torch.argmax(class_labels, dim=1)
        loss = self.criterion(outputs, class_labels.long())
        loss.backward()
        self.optimizer.step()

        return loss


    def eval_step(self, word_targets, class_labels, lengths):
        outputs = self.model(word_targets, lengths)
        _, predicted = torch.max(outputs.data, 1)
        # convert again because it changed again
        class_labels = torch.argmax(class_labels, dim=1)

        return [class_labels.size(0), (predicted == class_labels).sum().item()]


