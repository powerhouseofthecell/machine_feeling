############################################
################ CONFIG ####################
############################################
# this file holds all of the configuration #
# variables for the machine learning       #
# aspect of this project                   #
############################################
############################################
############################################

# the available model types to work with
available_model_types = ['mul_class']

# Data Generator Parameters
## rotation range for images, in degrees
rot_range = 360

## how much to shift the width and height, as proportions
w_shift = 0.5
h_shift = 0.5

## how much to shear images (see Internet for specifics)
shear = 0.5

## horizontal and vertical flipping of images
h_flip = True
v_flip = True

# Data Parameters
## the dimensions of the images, rows then cols
target_dim1 = 150
target_dim2 = 150

## how many images to generate at a time
batch_size = 100

## the type of classifier, if applicable
classification_type = 'categorical'

# Model Parameters
## activation distributions for the first and last layers
input_activation = 'relu'
output_activation = 'sigmoid'

## dimensionality of a filter to be run over the image
filter_dim = (3, 3)

# Compilation Parameters - see Keras docs for lists
## the metric for loss, found in Keras documentation
l_type = classification_type + '_crossentropy'

## the optimizer, a tool for how weights are decided
opt = 'adam'

## the metrics against which things are judged, can be several
met = ['accuracy']

# Train Parameters
## the number of steps to take in a given round
step_num = 600

## the number of rounds to run
epoch_num = 10

