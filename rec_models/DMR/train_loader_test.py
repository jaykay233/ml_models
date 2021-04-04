#coding:utf8

from model import *
from dataset import *
from datetime import timedelta, datetime

import paddle
num_epochs = 1
batch_size = 256
window_size = 50
starter_learning_rate = 0.001
learning_rate_decay = 1.0

today = datetime.today() + timedelta(0)

dataset = DataIterator('alimama_sampled.txt',batch_size)
train_loader = paddle.io.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
for batch_id, data in enumerate(train_loader()):
    feature = data[0]
    target = data[1]
    print(feature,target)
