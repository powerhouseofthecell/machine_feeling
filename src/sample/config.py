############################################
################ CONFIG ####################
############################################
# this file holds all of the configuration #
# variables for this project.              #
############################################
############################################
############################################

from os import listdir

# the available model types to work with
available_model_types = ['mul_class']


# Data Generator Parameters
## rotation range for images, in degrees
rot_range = 0

## how much to shift the width and height, as proportions
w_shift = 0.2
h_shift = 0.1

## how much to shear images (see Internet for specifics)
shear = 0.01

## horizontal and vertical flipping of images
h_flip = False
v_flip = False


# Data Parameters
## the dimensions of the images, rows then cols
target_dim1 = 200
target_dim2 = 200

## how many images to generate at a time
batch_size = 200

## the type of classifier, if applicable
classification_type = 'categorical'


# Model Parameters
## activation distributions for the layers
activations = ['relu', 'softmax']

## dimensionality of a filter to be run over the image
filter_dim = (3, 3)

# the number of classes found in test and train directories
num_classes = len(listdir('data/train')) - 1


# Compilation Parameters - see Keras docs for lists
## the metric for loss, found in Keras documentation
l_type = classification_type + '_crossentropy'

## the optimizer, a tool for how weights are decided
opt = 'adam'

## the metrics against which things are judged, can be several
met = ['accuracy']


# Train Parameters
## the number of steps to take in a given round (this will later be divided by the batch size)
step_num = 4000

## the number of rounds to run
epoch_num = 15

## number of workers to unleash on doing calculations
worker_num = 4

## whether or not to use multiprocessing
multiprocessing = False


# Evaluation Parameters
## the number of evaluation steps to run
eval_steps = 10