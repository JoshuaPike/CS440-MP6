# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output




        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate

        # Make first layer convolutional
        # print("in_size: ",in_size)
        # self.conv1 = torch.nn.Conv2d(3, 100, 32)
        # self.conv2 = torch.nn.Conv2d(3, 100, 32)
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 6, 5)
        # self.pool = torch.nn.MaxPool2d(1, 1) # Dont understand parameters here, but using these as a test
        self.pool = torch.nn.MaxPool2d(2, 2)
        # self.fc1 = torch.nn.Linear(6*14*14, 32)
        self.fc1 = torch.nn.Linear(6*5*5, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fcmid = torch.nn.Linear(64, 64)
        # self.fcmid2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(64, out_size)
        


        self.optim = torch.optim.SGD(params=self.parameters(), lr=lrate, momentum=0.9)



    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        # print("Original x shape \n", x.shape)
        # Reshape x
        x = x.view([-1, 3, 32, 32])
        # print("Rehaped x\n", x.shape)
        # print("I dont fucking know \n", torch.nn.functional.relu(self.conv1(x)).shape)
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        m = torch.nn.Dropout(p=0.2)
        # x = m(x)
        # print("After conv1 shape\n", x.shape)
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        # m = torch.nn.Dropout(p=0.2)
        # x = m(x)
        # print("After conv2 shape\n", x.shape)
        x = x.view(-1, 6 * 5 * 5)
        # x = x.view(-1, 8)
        # print("weird view line\n", x.shape)
        x = torch.nn.functional.relu(self.fc1(x))
        x = m(x)
        # print(x.shape)
        x = torch.nn.functional.relu(self.fc2(x))
        
        x = torch.nn.functional.relu(self.fcmid(x))
        # x = torch.nn.functional.relu(self.fcmid2(x))
        # print(x.shape)
        x = self.fc3(x)
        
        return x

        # return torch.ones(x.shape[0], 1)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        # Find mean and standard deviation to increase accuracy
        # mu = torch.mean(x)
        # sig = torch.std(x)
        # x = (x-mu)/sig

        yhat = self.forward(x)
        # print(yhat.shape)
        # print(y.shape)
        self.optim.zero_grad()

        # yhat = yhat.view([100, -1])

        lossq = self.loss_fn(yhat, y)

        lossq.backward()

        self.optim.step()

        return lossq.item()

        # return 0.0



def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N

    model's performance could be sensitive to the choice of learning_rate. We recommend trying different values in case
    your first choice does not seem to work well.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    numPix = len(train_set[0])

    lossArr = []

    # 0.007 best 0.8584 no dropout
    net = NeuralNet(lrate = 0.01, loss_fn=loss_fn, in_size = numPix, out_size = 2)
    train_set = (train_set - torch.mean(train_set)) / torch.std(train_set)
    dev_set = (dev_set - torch.mean(dev_set)) / torch.std(dev_set)

    # Split up batches. Batches has (images, label)
    batches = []
    i = 0
    while i < len(train_set):
        try:
            batches.append((train_set[i:i+100], train_labels[i:i+100]))
        except:
            batches.append((train_set[i:len(train_set)], train_labels[i:len(train_labels)]))
        i += 100

    it = 0
    i = 0
    while it < n_iter:
        # print('Current Training Iteration: %i' % it, end='\r')
        batchLoss = 0
        # for batch in batches:
        #     batchLoss = net.step(batch[0], batch[1])
        #     return

        if i < len(batches):
            batchLoss = net.step(batches[i][0], batches[i][1])
        else:
            i = 0
            batchLoss = net.step(batches[i][0], batches[i][1])
        
        lossArr.append(batchLoss)

        i+= 1
        it += 1

    # At this point training is complete
    # print("\nTraining Complete\n")
    netGuesses = []
    for image in dev_set:
        output = net.forward(image)
        # print("output: ", output)
        # print("output[0]: ", output[0])
        netGuesses.append(torch.argmax(output[0]))
        # if output[0][0].item() >= output[1].item():
        #     netGuesses.append(0)
        # else:
        #     netGuesses.append(1)

    return lossArr, netGuesses, net
