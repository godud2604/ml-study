# %%
import math
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras

# %%
class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, csv_path,
                fold, image_size, mode="train",
                shuffle=True):
        self.batch_size = batch_size
        self.fold = fold
        self.shuffle = shuffle
        self.image_size = image_size
        self.mode = mode

        self.df = pd.read_csv(csv_path)

        if self.mode == 'train':
            # fold가 아닌 것들만 선택하여 학습 데이터로 사용
            self.df = self.df[self.df['fold'] != self.fold]
        elif self.mode == 'val':
            # fold에 해당하는 것들만 선택하여 검증 데이터로 사용합니다.
            self.df = self.df[self.df['fold'] == self.fold]
        
        self.on_epoch_end()
        
    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    """
    fin을 (idx + 1) * self.batch_size로 계산하는 이유는,
    인덱스 idx가 현재 몇 번째 배치인지를 나타내는데, 
    (idx + 1) * self.batch_size는 다음 배치의 시작 인덱스를 나타냅니다. 
    따라서 현재 배치의 끝 인덱스가 된다.
    이렇게 함으로써 각 배치의 데이터를 올바르게 가져올 수 있다.
    """
    def __getitem__(self, idx):
        strt = idx * self.batch_size
        fin = (idx + 1) * self.batch_size 
        data = self.df.iloc[strt:fin]

        batch_x, batch_y = self.get_data(data)

        return np.array(batch_x), np.array(batch_y)

    def get_data(self, data):
        batch_x = []
        batch_y = []

        for _, r in data.iterrows():
            file_name = r['file_name']
            image = cv2.imread(f'data/images/{file_name}.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image / 255.

            label = int(r['species']) - 1

            batch_x.append(image)
            batch_y.append(label)

        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            # 데이터를 섞어서 학습 시 다양한 데이터에 노출
            self.df = self.df.sample(frac=1).reset_index(drop=True) # self.df.sample(frac=1) 이 뭐지?


# %%
csv_path = 'data/kfolds.csv'
train_generator = DataGenerator(
    batch_size=9, 
    csv_path=csv_path,
    fold=1, 
    image_size=256,
    mode="train",
    shuffle=True
)

# %%
print(len(train_generator))
print(len(csv_path))
# %%
class_name = ['Cat', 'Dog']

for batch in train_generator:
    X, y = batch
    plt.figure(figsize=(15, 15))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1) 
        plt.imshow(X[i])
        plt.title(class_name[y[i]])
        plt.axis('off')
    break
# %%
