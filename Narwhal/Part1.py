#!/usr/bin/env python

#imports
import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import imageio
import gdown


# ### Prevent exponential memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#tf.config.list_physical_devices('GPU')

#physical_devices = tf.config.list_physical_devices('GPU')
#try:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#except:
#    pass

print('\nGPU Check: Success \n')

#Part2___________________________________________________________________

#Download and extract data
#url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
#output = 'data.zip'
#gdown.download(url, output, quiet=False)
#gdown.extractall('data.zip')


# ### Data Loading Function

# turn video into usable data
def load_video(path:str) -> List[float]:
    
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        #isolates mouth use Dlib to isolate mouth to update
        frames.append(frame[190:236, 80:220,:])        
    cap.release()
    
    #help focus on vital data
    mean = tf.math.reduce_mean(frames)
    std =tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames-mean), tf.float32) / std

# all available charcters
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

#convert text to num and vice versa
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

#print(
    #f"The vocabulary is: {char_to_num.get_vocabulary()} "
    #f"(size ={char_to_num.vocabulary_size()})\n"
#)
print('\nPart 2 Done!!!\n')

# Part 3________________________________________________________________________________


#use alignments

def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        #ignore silence value
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


#load alignements and videos simultaneously

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    #file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments


# ### Test Code

test_path = './data/s1/bbal6n.mpg'

tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('/')[-1].split('.')[0]

frames, alignments = load_data(tf.convert_to_tensor(test_path))
max_frame=len(frames)

# characters from data
print([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])
print('\n')
#condensed words from data
tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])


def mappable_function(path:str) ->List[str]:
    #tensorflow for pure string processing needs py_function
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

print('\nPart 3 Done!!!\n')

#Part4_______________________________________________________________________________

#make dataset
data = tf.data.Dataset.list_files('./data/s1/*.mpg')
#shuffle data
data = data.shuffle(500, reshuffle_each_iteration=False)
#transfer data through pipeline and get it back
data = data.map(mappable_function)
#ensure 75 frames and 40 tokens
data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))
#optimizng preloading data
data = data.prefetch(tf.data.AUTOTUNE)
# Added for split 
train = data.take(450)
test = data.skip(450)

#print('CheckPoint1 !!!\n')
frames, alignments = data.as_numpy_iterator().next()

sample = data.as_numpy_iterator()

val =sample.next(); val[0]

#print('CheckPoint2 !!!\n')
#changes dataset into a gif 
fv=(val[0][1].astype(np.uint8) * 255).squeeze()
imageio.mimsave('./animation.gif',fv, duration=50)

#print('CheckPoint3 !!!\n')
#decodes gif
samp=val[1][1]
print([bytes.decode(x) for x in num_to_char(samp).numpy()])
print('\n')
#print('CheckPoint4 !!!\n')
print('\nPart 4!!! Done\n')
#Part 5______________________________________________________________________________________
# # Design Deep Neural Network
# 
#import more dependencies
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

print('CheckPoint 1 !!!\n')

data.as_numpy_iterator().next()[0][0].shape

model = Sequential()
# 3 sets of convulusions
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

#flatten convulsions w/ timedistrubted layer
model.add(TimeDistributed(Flatten()))

#Two sets of LSTMs
model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

#Dense Layer
model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

print('\nCheckPoint 2 !!!\n')

#testing [predictions data]
yhat = model.predict(val[0])


#the predicted output in charactesr
sam=tf.argmax(yhat[0],axis=1)
print([bytes.decode(x) for x in num_to_char(sam).numpy()])
print('\n')


print('\nCheckPoint 3 !!!\n')

print('\nPart 5 Done !!!\n')

#Part 6______________________________________________________________________________

# # Training Setup and Train
# 

# In[39]:


# define learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


print('CheckPoint 1 !!!\n')


#define CTC loss
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


print('CheckPoint 2 !!!\n')


class ProduceExample(tf.keras.callbacks.Callback):
    
    def __init__(self, dataset) -> None: 
        self.dataset = dataset.as_numpy_iterator()
    
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)

#Compiles our model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
print('CheckPoint 3 !!!\n')

# ### Define Callbacks
#Save model checkpoints
checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint'), monitor='loss', save_weights_only=True) 

#Scheulder should drop the learning rate
schedule_callback = LearningRateScheduler(scheduler)


#make predictions after each epoch
example_callback = ProduceExample(test)


print('CheckPoint 4 !!!\n')


model.fit(train, validation_data=test, epochs=30, callbacks=[checkpoint_callback, schedule_callback, example_callback])

print('\nCheckPoint 5 !!!\n')

print('Part 5 Done !!!\n')