# %%
import numpy as np
import os
import glob
import re

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform

from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.models import save_model
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Description

# %% [markdown]
# ## Strategy

# %% [markdown]
# 

# %% [markdown]
# ## Sources

# %% [markdown]
# - Chess FEN Generator: https://www.kaggle.com/code/koryakinp/chess-fen-generator/notebook
# - Chess FEN Generator Improved: https://www.kaggle.com/code/meditech101/chess-fen-generator-improved
# - Chess Positions FEN Prediction (EDA + CNN Model): https://www.kaggle.com/code/ibrahimsoboh/chess-positions-fen-prediction-eda-cnn-model
# 

# %% [markdown]
# # Data Loading

# %%
train_size = 10000
test_size = 3000

train = glob.glob("dataset/dataset/train/*.jpeg")
test = glob.glob("dataset/dataset/test/*.jpeg")

shuffle(train)
shuffle(test)

train = train[:train_size]
test = test[:test_size]

piece_symbols = 'prbnkqPRBNKQ'

# %%
def fen_from_filename(filename):
  base = os.path.basename(filename)
  return os.path.splitext(base)[0]

# %%
f, axarr = plt.subplots(1,3, figsize=(120, 120))

for i in range(0,3):
    axarr[i].set_title(fen_from_filename(train[i]), fontsize=70, pad=30)
    axarr[i].imshow(mpimg.imread(train[i]))
    axarr[i].axis('off')

# %% [markdown]
# # Preprocessing

# %%
def onehot_from_fen(fen):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = re.sub('[-]', '', fen)

    for char in fen:
        if(char in '12345678'):
            output = np.append(
              output, np.tile(eye[12], (int(char), 1)), axis=0)
        else:
            idx = piece_symbols.index(char)
            output = np.append(output, eye[idx].reshape((1, 13)), axis=0)

    return output

def fen_from_onehot(one_hot):
    output = ''
    for j in range(8):
        for i in range(8):
            if(one_hot[j][i] == 12):
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if(j != 7):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output

# %%
def process_image(img):
    downsample_size = 200
    square_size = int(downsample_size/8)
    img_read = io.imread(img)
    img_read = transform.resize(
      img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)

# %%
def train_gen(features, labels, batch_size):
    for i, img in enumerate(features):
        y = onehot_from_fen(fen_from_filename(img))
        x = process_image(img)
        yield x, y

def pred_gen(features, batch_size):
    for i, img in enumerate(features):
        yield process_image(img)

# %% [markdown]
# # Model

# %%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(25, 25, 3), activation='relu', kernel_initializer='he_normal'))
model.add(layers.Conv2D(32, (3, 3), input_shape=(25, 25, 3), activation='relu', kernel_initializer='he_normal'))
model.add(layers.Conv2D(32, (3, 3), input_shape=(25, 25, 3), activation='relu', kernel_initializer='he_normal'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# %%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
es = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    mode='min', 
    verbose=1)

# %%
model.fit_generator(train_gen(train, None, 64), steps_per_epoch=train_size, callbacks=es)

# %%
# Save the model to a file
save_model(model, "model.h5")

# %% [markdown]
# # Evaluation

# %%
res = (
  model.predict_generator(pred_gen(test, 64), steps=test_size)
  .argmax(axis=1)
  .reshape(-1, 8, 8)
)

# %%
pred_fens = np.array([fen_from_onehot(one_hot) for one_hot in res])
test_fens = np.array([fen_from_filename(fn) for fn in test])

final_accuracy = (pred_fens == test_fens).astype(float).mean()

print("Final Accuracy: {:1.5f}%".format(final_accuracy))

# %%
test_fens

# %%
confusion_matrix(test_fens, pred_fens)

# %%
def display_with_predicted_fen(image):
    pred = model.predict(process_image(image)).argmax(axis=1).reshape(-1, 8, 8)
    fen = fen_from_onehot(pred[0])
    imgplot = plt.imshow(mpimg.imread(image))
    plt.axis('off')
    plt.title(fen)
    plt.show()

# %%
display_with_predicted_fen(test[0])
display_with_predicted_fen(test[1])
display_with_predicted_fen(test[2])


