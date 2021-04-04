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
    model.train()
    dataset = DataIterator('alimama_sampled.txt', batch_size)
    train_loader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters())
    for epoch_id in range(num_epochs):
        train_losses = []
        for batch_id, data in enumerate(train_loader()):
            # print(batch_id)
            feature = data[0]
            target = data[1]
            feature = paddle.to_tensor(feature)
            target = paddle.to_tensor(target)
            preds = model(feature)
            loss = paddle.nn.functional.binary_cross_entropy(preds,target)
            if batch_id % 10 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch_id, batch_id, loss.numpy()))
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        # print("epoch: {} train_loss: {}".format(epoch_id,sum(train_losses)*1.0/len(train_losses)))


if __name__ == '__main__':
    model = Model_DMR()
    train(model)
