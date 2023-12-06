"""
- 문제점
  => train, validation, test로 data를 분리하여 학습을 시켰으나, 이미지가 학습할 수 없음 (포켓몬 진화 전, 후 형태가 너무 달라 학습 불가
  => 과소 적합 (같은 이미지가 없으므로 학습 불가.)

- 해결 방법
  => final_images, final_labels -> crop, rotation augmentation
     image size up
     
"""

# %% Import libraries
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# %% Import dataset

# defining root directory
from PIL import Image

root_dir = '/Users/ihaeyeong/Desktop/ml-study/Kaggle/pockmon/images'

files = os.path.join(root_dir)
File_names = os.listdir(files)
print("This is the list of all the files present in the path given to us:\n")
print(File_names)

# plot here
"""
- plt.subplots(2, 3, figsize=(15, 8)) : 2x3 크기의 서브플롯 생성
- fig : 전체 그림을 나타내는 Figure 객체
- axes : 각 서브플롯을 나타내는 Axes 객체의 2차원 배열
- figsize : 전체 그림의 크기를 지정
"""
fig, axes = plt.subplots(2, 3, figsize=(15, 8)) # 2x3 크기의 서브플롯 생성
first_five = File_names[0:6] 
print('first_five', first_five)

def subplots():
    # Use the axes for plotting
    i = 0 # row
    j = 0 # column
    k = 0 # 이미지 파일의 인덱스

    for k in range(5):
        state = os.path.join(root_dir, first_five[k])
        img = Image.open(state)
        axes[i, j].imshow(img)

        if k == 2:
            i += 1
            j = 0
        else:
            j += 1

    plt.tight_layout(pad=2) # 서브플롯들 간의 간격을 설정하여 레이아웃 조정

subplots()
    

# %%
data = pd.read_csv('/Users/ihaeyeong/Desktop/ml-study/Kaggle/pockmon/pokemon.csv')

data.head()
# %%
# We are going to use Type1 column as our labels. Each Name is unique and classified into 18 Type1 types

data_dict = {}

for key, val in zip(data['Name'], data['Type1']):
    data_dict[key] = val

print('data_dict', data_dict)

# %%
labels = data["Type1"].unique()
print('labels', labels)

# %%
# Create a dictionary and assign each label in labels list a unique id from 1 to 18. Name the dictionary as "labels_idx"
ids = list(range(18))
labels_idx = dict(zip(labels, ids))

print('labels_idx', labels_idx)

# %%
final_images = []
final_labels = []
count = 0
files = os.path.join(root_dir)

for file in File_names:

    count += 1
    img = cv2.imread(os.path.join(root_dir, file), cv2.COLOR_BGR2GRAY)

    fileName = file.split(".")[0]
    label = labels_idx[data_dict[fileName]]

    """
    - OpenCV를 사용하여 읽어온 이미지는 Numpy 배열이 아니라, OpenCV의 이미지 객체이다.
    - 따라서 'np.array(img)'를 사용하여 OpenCV의 이미지 객체를 Numpy 배열로 변환
    """
    final_images.append(np.array(img)) # 이미지를 Numpy배열로 변환
    final_labels.append(np.array(label))


# converting lists into numpy array
# normalizing and reshaping the data
final_images = np.array(final_images, dtype = np.float32) / 255.0
final_labels = np.array(final_labels, dtype = np.int8).reshape(809, 1)

print('final_images',final_images)
print('final_labels',final_labels[0])

# %%
# 훈련, 검증, 테스트 데이터 분리
X_train, X_temp, y_train, y_temp = train_test_split(final_images, final_labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# %%
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(120, 120, 3)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(18)
])

model.summary()

# %%

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics='accuracy')


history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))


# %%
# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\n테스트 정확도:', test_acc)

# %%

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(final_images)

print('\n', predictions[0])
id = np.argmax(predictions[0])
print("\nid that we got from the model as prediction: {}\nType of pokemon associted with that id: {} ".format(id,labels[id]))
print("accuracy of the model", history.history['accuracy'][-1])
# %%

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test)

print('\n', predictions[0])
id = np.argmax(predictions[0])
print("\nid that we got from the model as prediction: {}\nType of pokemon associted with that id: {} ".format(id,labels[id]))
print("accuracy of the model", history.history['accuracy'][-1])
# %%
