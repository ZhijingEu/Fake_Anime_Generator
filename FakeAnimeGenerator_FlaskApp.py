import tensorflow as tf
import string
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

# Read the dataset from csv
dataset = pd.read_csv('ParseHubExtract.csv')

def tokenize_corpus(corpus, num_words=-1):
  # Fit a Tokenizer on the corpus
  if num_words > -1:
    tokenizer = Tokenizer(num_words=num_words)
  else:
    tokenizer = Tokenizer()
  tokenizer.fit_on_texts(corpus)
  return tokenizer

def create_corpus(dataset, field):
  # Remove all other punctuation
  dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
  # Make it lowercase
  dataset[field] = dataset[field].str.lower()
  # Make it one long string to split by line
  lyrics = dataset[field].str.cat()
  corpus = lyrics.split('\n')
  # Remove any trailing whitespace
  for l in range(len(corpus)):
    corpus[l] = corpus[l].rstrip()
  # Remove any empty lines
  corpus = [l for l in corpus if l != '']

  return corpus

corpus = create_corpus(dataset, 'selection1_name')
# Tokenize the corpus
tokenizer = tokenize_corpus(corpus, num_words=500)
total_words = tokenizer.num_words

sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		sequences.append(n_gram_sequence)

# Pad sequences for equal input length 
max_sequence_len = max([len(seq) for seq in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

# Split sequences between the "input" sequence and "output" predicted word
input_sequences, labels = sequences[:,:-1], sequences[:,-1]
# One-hot encode the labels
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

sypnosis_checkpoint_path = "TextModel_chkpoint/sypnosis.ckpt"
sypnosis_checkpoint_dir = os.path.dirname(sypnosis_checkpoint_path)

#Titles checkpoint
title_checkpoint_path = "TextModel_chkpoint/titles.ckpt"
title_checkpoint_dir = os.path.dirname(title_checkpoint_path)

# Create a callback that saves the model's weights
title_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=title_checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

sypnosis_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=sypnosis_checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

title_model = Sequential()
title_model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
title_model.add(Bidirectional(LSTM(20)))
title_model.add(Dense(total_words, activation='softmax'))
title_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
title_model.load_weights(title_checkpoint_path)

def MakeTitle(seed_text,next_words):
    for _ in range(next_words):
      token_list = tokenizer.texts_to_sequences([seed_text])[0]
      token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
      predicted_probs = title_model.predict(token_list)[0]
      predicted = np.random.choice([x for x in range(len(predicted_probs))],
                                   p=predicted_probs)
      output_word = ""
      for word, index in tokenizer.word_index.items():
        if index == predicted:
          output_word = word
          break
      seed_text += " " + output_word
    return seed_text

sypnosis_corpus = create_corpus(dataset, 'selection1_Sypnosis')
# Tokenize the corpus
sypnosis_tokenizer = tokenize_corpus(sypnosis_corpus, num_words=2500)
sypnosis_total_words = sypnosis_tokenizer.num_words

sypnosis_sequences = []
for line in sypnosis_corpus:
	sypnosis_token_list = sypnosis_tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(sypnosis_token_list)):
		sypnosis_n_gram_sequence = sypnosis_token_list[:i+1]
		sypnosis_sequences.append(sypnosis_n_gram_sequence)

# Pad sequences for equal input length 
sypnosis_max_sequence_len = max([len(seq) for seq in sypnosis_sequences])
sypnosis_sequences = np.array(pad_sequences(sypnosis_sequences, maxlen=sypnosis_max_sequence_len, padding='pre'))

# Split sequences between the "input" sequence and "output" predicted word
sypnosis_input_sequences, sypnosis_labels = sypnosis_sequences[:,:-1], sypnosis_sequences[:,-1]
# One-hot encode the labels
sypnosis_one_hot_labels = tf.keras.utils.to_categorical(sypnosis_labels, num_classes=sypnosis_total_words)

sypnosis_model = Sequential()
sypnosis_model.add(Embedding(sypnosis_total_words, 64, input_length=sypnosis_max_sequence_len-1))
sypnosis_model.add(Bidirectional(LSTM(20)))
sypnosis_model.add(Dense(sypnosis_total_words, activation='softmax'))
sypnosis_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

sypnosis_model.load_weights(sypnosis_checkpoint_path)

SeedWord=["In","When","As","Before","After","On","During","At","The","The","The","The","The","The","The"]
def MakeSypnosis(next_words):
    seed_text=np.random.choice(SeedWord)
    for _ in range(next_words):
      token_list = sypnosis_tokenizer.texts_to_sequences([seed_text])[0]
      token_list = pad_sequences([token_list], maxlen=sypnosis_max_sequence_len-1, padding='pre')
      predicted_probs = sypnosis_model.predict(token_list)[0]
      predicted = np.random.choice([x for x in range(len(predicted_probs))],
                                   p=predicted_probs)
      output_word = ""
      for word, index in sypnosis_tokenizer.word_index.items():
        if index == predicted:
          output_word = word
          break
      seed_text += " " + output_word
    return seed_text

import os
import numpy as np
from PIL import Image

# Defining an image size and image channel
# We are going to resize all our images to 128X128 size and since our images are colored images
# We are setting our image channels to 3 (RGB)

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = 'ImageDataset/'

# Defining image dir path. Change this if you have different directory
images_path = IMAGE_DIR 

training_data = np.load('ProcessedImages/processed_images.npy')

from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# Preview image Frame
PREVIEW_ROWS = 2
PREVIEW_COLS = 3
PREVIEW_MARGIN = 4
SAVE_FREQ = 5
# Size vector to generate images from
NOISE_SIZE = 100
# Configuration
EPOCHS = 500 # number of iterations
BATCH_SIZE = 32
GENERATE_RES = 3
IMAGE_SIZE = 128 # rows/cols
IMAGE_CHANNELS = 3

def build_discriminator(image_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
    input_shape=image_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)

def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation='relu',input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    for i in range(GENERATE_RES):
         model.add(UpSampling2D())
         model.add(Conv2D(256, kernel_size=3, padding='same'))
         model.add(BatchNormalization(momentum=0.8))
         model.add(Activation('relu'))
    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding='same'))
    model.add(Activation('tanh'))
    input = Input(shape=(noise_size,))
    generated_image = model(input)
    
    return Model(input, generated_image)

def save_images(cnt, noise):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c +
                        IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1
            
    output_path = 'OutputImages/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

generator_optimizer = Adam(1.5e-4, 0.5)
discriminator_optimizer = Adam(1.5e-4, 0.5)


discriminator = build_discriminator(image_shape)
discriminator.compile(loss='binary_crossentropy',optimizer=discriminator_optimizer, metrics=['accuracy'])

generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)
random_input = Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)

validity = discriminator(generated_image)
combined = Model(random_input, validity)
combined.compile(loss='binary_crossentropy',optimizer=generator_optimizer, metrics=['accuracy'])

discriminator.trainable = False

y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))
fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))
cnt = 1

discriminator=load_model("ImageModel_checkpoints/SpellMLGPU_10000Epochs/discriminator_model10k.h5")
generator=load_model("ImageModel_checkpoints/SpellMLGPU_10000Epochs/generator_model10k.h5")
combined=load_model("ImageModel_checkpoints/SpellMLGPU_10000Epochs/combined_model10k.h5")

import matplotlib as plt
from matplotlib.pyplot import imshow
from matplotlib.figure import Figure
import random
import re

genre_list=('Drama','Detective','Sports','Historical', 'Seinen', 'Thriller','Slice of Life', 'Isekai','Comedy', 'Romance', 'School', 'Action', 'Supernatural','Science-Fiction', 'Fantasy', 'Psychological')

class imgcounterclass():
    """Class container for processing stuff."""

    _counter = 0

    def addcounter(self):
        self._counter += 1

postercounter=imgcounterclass()

import io
import random
from flask import Response, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/FakeAnimeGenerator')
def make_page():
    titlex,ratingx,sypnosisx, genresx,postercounterstring = randomize_new(postercounter)
    return render_template('FakeAnimeGenerator.html',title=titlex,rating=ratingx,sypnosis=sypnosisx,genres=genresx,postercounter=postercounterstring)

def randomize_new(postercounter):
    postercounter.addcounter()
    titleNo_of_words = np.random.randint(2,5)
    SampleTitle=str(MakeTitle("",titleNo_of_words)).title().strip()
    Sypnosis_No_of_words=np.random.randint(70,150)
    SampleSypnosis=re.sub("\s\s+", " ",str(MakeSypnosis(Sypnosis_No_of_words)))
    titleNo_of_genres=np.random.randint(2,5)
    titleRating=round(np.random.uniform(5.0,9.5),2)
    genres=random.sample(genre_list, titleNo_of_genres)
    postercounterstring=str(postercounter._counter)
    SampleImage=Generate_Image().save(f'static/{postercounterstring}posterimage.png')
    return SampleTitle,titleRating,SampleSypnosis,genres,postercounterstring

def Generate_Image():
    PREVIEW_ROWS = 1
    PREVIEW_COLS = 1
    PREVIEW_MARGIN = 4
    # Size vector to generate images from
    NOISE_SIZE = 100
    noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c +
                        IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1
    im = Image.fromarray(image_array)
    im = im.resize((400, 600))
    #imshow(np.asarray(im))
    #im.show()
    return im

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
