#!/usr/bin/env python3

import argparse

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

# add an argument for if you've loaded a model, still training it again
ap.add_argument('-t', action='store_true')

# add an argument for automatically saving the model
ap.add_argument('-y', action='store_true')

args = vars(ap.parse_args())

# import the model after parsing arguments
from ml.ml_model import Model

# instantiate a model for the whole file to use
m = Model()

# get the model built and ready to go
def initialize():
    m.load_data('data')

    if not args['load']:
        m.build_model()
    else:
        m.load_model(args['load'])

# train the model n rounds
def train_model(n):
    # only train if forced or if nothing was loaded
    if args['t'] or not args['load']:
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