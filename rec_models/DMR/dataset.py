import paddle
from paddle.io import Dataset
import numpy as np
BATCH_SIZE = 64
BATCH_NUM = 20


class DataIterator(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, source, batch_size=256):
        super().__init__()
        self.file_source = source
        self.batch_size = batch_size
        self.source, self.target = self.readfile()
        self.num_samples = self.source.shape[0]

    def readfile(self):
        """
        readfile
        :return:
        """
        source = []
        target = []
        with open(self.file_source, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                ss = line.split(',')
                source.append(ss[:-1])
                target.append(ss[-1])

        return np.array(source,np.float32), np.array(target,np.float32).reshape([-1,1])

    def __getitem__(self, index):
        return self.source[index], self.target[index]

    def __len__(self):
        return self.num_samples
