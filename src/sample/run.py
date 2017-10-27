#!/usr/bin/env python3

import argparse
from os.path import isfile
from os import remove, listdir

# initialize an argument parser
ap = argparse.ArgumentParser()

# add an argument for output name, else saves to default name
ap.add_argument(
    '-o',
    '--output',
    metavar='OUTPUT',
    type=str,
    help='the filename to write the model to'
)

# add an argument for how many rounds of training to undergo
ap.add_argument(
    '-r',
    '--rounds',
    metavar='ROUNDS',
    default=1,
    type=int,
    help='the number of rounds of training to undergo'
)

# add an argument for loading in a model
ap.add_argument(
    '-l',
    '--load',
    metavar='LOAD',
    type=str,
    help='the filepath of an existing model which you want to load'
)

ap.add_argument(
    '-i',
    '--img',
    metavar='IMG',
    type=str,
    default='tmp',
    help='the filepath to save screenshots to'
)

# add an argument for using a pretrained model
ap.add_argument('--pretrained', action='store_true', help='use a pretrained model')

# add an argument for if you've loaded a model, still training it again
ap.add_argument('-t', action='store_true', help='force training of the model')

# add an argument for automatically saving the model
ap.add_argument('-y', action='store_true', help='autosave the model')

args = vars(ap.parse_args())

# import the model after parsing arguments, since it takes a bit
from ml.ml_model import Model

# instantiate a model for the whole file to use
m = Model()

# get the model built and ready to go
def initialize():
    m.load_data('data')

    if args['pretrained']:
        m.load_model('', pretrained=True)

    elif not args['load']:
        m.build_model()

    else:
        m.load_model(args['load'])

# train the model n rounds
def train_model(n):
    # only train if forced or if nothing was loaded
    if (args['t'] or not args['load']) and not args['pretrained']:
        for i in range(n):
            m.train()  

# save the model
def save_model():
    # in case the user wanted to save the model automatically
    if args['y']:
        ans = 'y'
    else:
        ans = ''
    
    # check with the user and see if they want to save
    while not (ans == 'y' or ans == 'n'):
        ans = input('Do you want to save the model? (y/N) ').lower()

    if ans == 'y':
        m.save(save_path=args['output'])

def run_im_capture():
    # import computer vision after the machine learning has gone well
    import cv.im_capture as im

    # runs the image capture and lets you save the screenshot
    im.run(filepath=args['img'])
    

def predict():
    # we need some extra packages
    from keras.preprocessing.image import img_to_array, load_img
    from config import target_dim1, target_dim2
    import numpy as np

    # if the image exists, and we're using a pretrained model
    if isfile(args['img'] + '.png') and args['pretrained']:
        image = load_img(args['img'] + '.png', target_size=(299, 299))

    elif isfile(args['img'] + '.png'):
        image = load_img(args['img'] + '.png', target_size=(target_dim1, target_dim2))

    # convert the image to a proper numpy array
    image = img_to_array(image)

    # add a dimension to this array so that it fits what our model expects
    image = np.expand_dims(image, axis=0)

    # actually run the prediction
    m.predict(image)

    # remove the temporary file we created
    remove(args['img'] + '.png')

# a method for actually evaluating the model
def evaluate():
    m.evaluate()
    return m.evaluation

# TODO: Make this into some form of loop for interacting live with the model, a la an interpreter
# run the actual file
if __name__ == '__main__':
    initialize()

    train_model(args['rounds'])

    save_model()

    run_im_capture()

    predict()

    for k in m.prediction:
        print('It is {} with {}% likelihood!'.format(k, 100 * m.prediction[k]))

    # single letter variable names seemed apropos here, may need to change later
    e = evaluate()

    print('On evaluation, metrics were as follows:')
    for k in e:
        print('\t{}: {}'.format(k, e[k] * 100))
        


    
