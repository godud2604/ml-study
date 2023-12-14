# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
from glob import glob

import cv2
import xml.etree.ElementTree as et

from matplotlib.patches import Rectangle
from sklearn.model_selection import KFold, StratifiedKFold

sns.set_style('whitegrid') # 그래프의 배경을 whitegrid 스타일로 설정

# %%
"""
 - skiprows=6 => 파일의 시작 부분에서 6줄을 건너뛰고 그 이후의 데이터를 읽어옵니다
 - delimiter = ' ' => 각 열을 구분할 때 공백을 사용한다는 의미. (파일의 내용은 데이터가 공백으로 구분된 형태이기 때문.)
"""
df = pd.read_csv('./data/annotations/list.txt', skiprows=6 ,delimiter=' ', header=None)
df.columns = ['file_name', 'id', 'species', 'breed']
df

# %%
"""
species 1: cat / 2: dog
"""
print(df['species'].value_counts().sort_index())

value_counts = df['species'].value_counts().sort_index()

# value_counts.values : array([2370, 4978])
# range(len(value_counts)) : range(0, 2)
# value_counts.index : Int64Index([1, 2], dtype='int64')
# value_counts.index.values : array([1, 2])

plt.bar(range(len(value_counts)), value_counts.values, align='center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

plt.show()

# %%
print(df['id'].value_counts().sort_index())

value_counts = df['id'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align='center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

plt.tight_layout() # x label 겹치지 않게
plt.show()
# %% 고양이의 종류 count
value_counts = df[df['species'] == 1]['breed'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align='center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

plt.tight_layout() # x label 겹치지 않게
plt.show()

# %% 강아지의 종류 count
value_counts = df[df['species'] == 2]['breed'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align='center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

plt.tight_layout() # x label 겹치지 않게
plt.show()

# %%
image_dir = 'data/images/'
bbox_dir = 'data/annotations/xmls/'
seg_dir = 'data/annotations/trimaps/'

# %%
"""
glob(image_dir + '*.jpg')
=> 디렉토리 안에 있는 모든 확장자가 .jpg인 파일을 찾아서 해당 파일들의 경로를 리스트로 반환합니다.
"""
image_files = glob(image_dir + '*.jpg')
len(image_files)
image_files[:10]

# %%
seg_files = glob(seg_dir + '*.png')
len(seg_files)
seg_files[:10]

# %%
bbox_files = glob(bbox_dir + '*.xml')
len(bbox_files)
bbox_files[:10]

# %%
image_path = image_files[110]
bbox_path = image_path.replace(image_dir, bbox_dir).replace('jpg', 'xml')

image = cv2.imread(image_path) # openCV는 rgb가 아니라, bgr형식으로 읽어옴
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tree = et.parse(bbox_path)

xmin = float(tree.find('./object/bndbox/xmin').text) # 실수 변환
xmax = float(tree.find('./object/bndbox/xmax').text) # 실수 변환
ymin = float(tree.find('./object/bndbox/ymin').text) # 실수 변환
ymax = float(tree.find('./object/bndbox/ymax').text) # 실수 변환

rect_x = xmin
rect_y = ymin
rect_w = xmax - xmin
rect_h = ymax - ymin

rect = Rectangle((rect_x, rect_y), rect_w, rect_h, fill=False, color='red')
plt.axes().add_patch(rect)
plt.imshow(image)

plt.show()

# %%
image_path = image_files[110]
seg_path = image_path.replace(image_dir, seg_dir).replace('jpg', 'png')

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

seg_map = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(seg_map)

plt.show()

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)

df['fold'] = -1
# t: train, v: validation
for idx, (t, v) in enumerate(kf.split(df), 1):
    print(len(t), len(v))
    print(idx, t, v)

    df.loc[v, 'fold'] = idx

# %%
print(len(df[df['fold'] == 1]))
print(len(df[df['fold'] != 1]))
# %%

value_counts = df[df['fold'] != 5]['id'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align='center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

plt.tight_layout() # x label 겹치지 않게
plt.show()
# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df['fold'] = -1
# t: train, v: validation
for idx, (t, v) in enumerate(skf.split(df, df['id']), 1):
    print(len(t), len(v))
    print(idx, t, v)

    df.loc[v, 'fold'] = idx

# %%
value_counts = df[df['fold'] != 5]['id'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align='center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

plt.tight_layout() # x label 겹치지 않게
plt.show()
# %%
df.to_csv('data/kfolds.csv', index=False)
# %%
