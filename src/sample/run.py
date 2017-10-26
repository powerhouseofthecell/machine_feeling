#!/usr/bin/env python3

import argparse
from os.path import isfile

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
        ans = input('Do you want to save the model? (y/N)').lower()

    if ans == 'y':
        m.save(save_path=args['output'])

# run the actual file
if __name__ == '__main__':
    initialize()

    train_model(args['rounds'])

    save_model()

    # import computer vision after the machine learning has gone well
    import cv.im_capture as im

    # runs the image capture and lets you save the screenshot
    im.run(filepath=args['img'])
    
    if isfile(args['img'] + '.png'):
        print('success')
    # TODO: have the machine predict on image
    # TODO: print the prediction
    try:
        pass

    except KeyboardInterrupt:
        print('[+] Good bye!')
        exit(0)