#coding:utf8

from model import *
from dataset import *
from datetime import timedelta, datetime

import paddle
num_epochs = 2
batch_size = 256
window_size = 50
starter_learning_rate = 0.001
learning_rate_decay = 1.0

today = datetime.today() + timedelta(0)


def train(model):
    dataset = DataIterator('alimama_sampled.txt', batch_size)
    train_loader = paddle.io.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = paddle.optimizer.Adam(learning_rate=0.1,parameters=model.parameters())
    for epoch_id in range(num_epochs):
        train_losses = []
        for batch_id, data in enumerate(train_loader()):
            print(batch_id)
            feature = data[0]
            target = data[1]
            feature = paddle.to_tensor(feature)
            target = paddle.to_tensor(target)
            preds = model(feature)
            loss = paddle.nn.functional.binary_cross_entropy(preds,target)
            loss.backward(retain_graph=True)
            train_loss = loss.numpy()[0]
            optimizer.step()
            optimizer.clear_grad()
            train_losses.append(train_loss)
        print("epoch: {} train_loss: {}".format(epoch_id,sum(train_losses)*1.0/len(train_losses)))


if __name__ == '__main__':
    model = Model_DMR(0.001, 0)
    model.train()
    train(model)
