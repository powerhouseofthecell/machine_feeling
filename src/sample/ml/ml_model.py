import os
from datetime import datetime as d

#######################################################################
################ THE MACHINE LEARNING MODEL CLASS #####################
#######################################################################
#### Feel free to play around with how this class is built as you  ####
#### see fit. It was designed as a wrapper around Keras' nice,     ####
#### built-in classes, in order to provide an extremely high level ####
#### interface. If there are things you wish to modify, feel free. ####
#######################################################################

# will hide TensorFlow warnings, comment or remove to have those warnings back
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the pieces from keras to make the model work
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import AlphaDropout, GaussianNoise, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers

from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils

# import our configuration variables
from config import *

# define a decorator that makes sure we don't try to run something like a prediction without a model
def model_required(f):
    def wrapper(*args, **kwargs):
        if args[0].model:
            f(*args, **kwargs)
        else:
            print('[-] Please compile the model using the "build_model" method or load a model before attempting this')
    return wrapper

# also a decorator, makes sure we don't try to train a model without data
def data_required(f):
    def wrapper(*args, **kwargs):
        if args[0].loaded:
            f(*args, **kwargs)
        else:
            print('[-] Please load data using the "load_data" method before attempting this')
    return wrapper

# the class definition
class Model():
    # initialization of the class object, defaults to a multi-class classifier
    def __init__(self, model_type='mul_class'):
        self.model_type = model_type
        self.model = None
        self.loaded = False

        # if the selected model is not available, say as much
        if not model_type in available_model_types:
            print(
                '[-] Invalid model type input - {} - please use one of the following: {}'.format(
                    model_type,
                    ' '.join([ x for x in available_model_types ])
                )
            )

    # a method for loading the model if it has already been created and saved
    def load_model(self, model_path, pretrained=False):
        self.pretrained = pretrained

        if not pretrained:   
            try:
                if os.path.isfile(model_path):
                    self.model = load_model(model_path)
                    print('[+] Model loading complete')

                else:
                    print('[-] Model loading incomplete, could not find model - {}'.format(model_path))

            except Exception as err:
                print('[-] Model loading unsuccessful, please check your model file:')
                print(err)
        else:
            from keras.applications import InceptionV3
            self.model = InceptionV3(weights='imagenet')

        # a "begin" marker to time how long it takes (in real time) to compile
        start_compile = d.now()

        # actually compile the model
        self.model.compile(
            loss=l_type,
            optimizer=opt,
            metrics=met
        )

        # a calculation of the compile time, in seconds
        compile_time = (d.now() - start_compile).total_seconds()

        print('[+] Model successfully compiled in {:.3f} seconds'.format(compile_time))


    # a method for loading in the data given path (and many optional arguments)
    # note, this data path should point to a folder with the data
    def load_data(self, data_path, data_type='img'):
        # check that the data exists and has two subfolders, 'test' and 'train'
        if os.path.isdir(data_path) and os.path.isdir(data_path + '/train/') and os.path.isdir(data_path + '/test/'):
            if data_type == 'img':
                # for images, we can increase the size/variety of our data set using generators, courtesy of Keras
                train_gen = ImageDataGenerator(
                    rotation_range=rot_range,
                    width_shift_range=w_shift,
                    height_shift_range=h_shift,
                    rescale=1./255,
                    shear_range=shear,
                    fill_mode='nearest', 
                    horizontal_flip=h_flip,
                    vertical_flip=v_flip
                )

                test_gen = ImageDataGenerator(rescale=1./255)

                # uses the generator to make data, uses parameters from config
                self.train_data = train_gen.flow_from_directory(
                    data_path + '/train/',
                    target_size=(target_dim1, target_dim2),
                    batch_size=batch_size,
                    class_mode=classification_type
                )

                self.test_data = test_gen.flow_from_directory(
                    data_path + '/test/',
                    target_size=(target_dim1, target_dim2),
                    batch_size=batch_size,
                    class_mode=classification_type
                )

                print('[+] Data successfully loaded')
                self.loaded = True

            else:
                print('[-] Datatype {} not yet supported.'.format(str(data_type)))

        else:
            print('[-] The path provided is not a folder containing "train" and "test" or does not exist, please try again.')
    
    # method for building the actual model
    def build_model(self, r_params=False):
        self.pretrained = False

        # define the model type, from Keras
        model = Sequential()

        if self.model_type == 'mul_class':
            model.add(Conv2D(
                32,
                filter_dim,
                input_shape=(target_dim1, target_dim2, 3),
                activation=activations[0]
            ))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            model.add(Conv2D(128, filter_dim, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten())

            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.3))

            model.add(Dense(num_classes, activation=activations[-1]))

            # a "begin" marker to time how long it takes (in real time) to compile
            start_compile = d.now()

            # actually compile the model
            model.compile(
                loss=l_type,
                optimizer=opt,
                metrics=met
            )

            # a calculation of the compile time, in seconds
            compile_time = (d.now() - start_compile).total_seconds()

            self.model = model

            print('[+] Model successfully compiled in {:.3f} seconds'.format(compile_time))


    # a method for actually training the model on the supplied data
    # TODO: maybe allow override of config stuff?
    @model_required
    @data_required
    def train(self):
        # we cannot train a pretrained model on our dataset
        if not self.pretrained:
            try:
                # again, a variable for timing
                fit_start = d.now()

                # fit the model to the data
                self.model.fit_generator(
                    self.train_data,
                    steps_per_epoch=step_num // batch_size,
                    epochs=epoch_num,
                    validation_data=self.test_data,
                    validation_steps=step_num // batch_size
                )

                # another time calculation, but for fitting, in seconds
                fit_time = (d.now() - fit_start).total_seconds() / 60

                print('[+] Training completed in {:.3f} minutes'.format(fit_time))

            except KeyboardInterrupt:
                print('\nUser aborted...')

    # method for predicting on an input data piece
    @model_required
    def predict(self, raw_input):
        try:
            if not self.pretrained:

                pred = self.model.predict(raw_input)

                # get the list of labels
                labels = listdir('data/train')

                # remove hidden files from the labels list
                while labels[0].startswith('.'):
                    labels.pop(0)

                # initialize a dictionary for storing label to probability mappings
                pmap = dict()
                for i in range(len(labels)):
                    pmap[labels[i]] = list(pred[0])[i]
                
                self.prediction = pmap
                print('[+] Prediction successfully completed')

            else:
                # preprocess the image for the pretrained net
                image = preprocess_input(raw_input)

                # make predictions
                prediction = self.model.predict(image)
                preds = imagenet_utils.decode_predictions(prediction)

                # create a dictionary to store the top five predictions with their probabilities
                p = dict()
                for (i, (imagenetID, label, prob)) in enumerate(preds[0]):
                    p[label] = prob

                self.prediction = p
                print('[+] Prediction successfully completed')

        except ValueError as err:
            print('[-] Prediction failed, please check the input shape:')
            print(err)

            
    # method for saving the model to file
    @model_required
    def save(self, save_path=None):
        if not save_path:
            save_path = 'Model_{}'.format(d.now().strftime('%A%B%-d%Y%H%M%S'))
        try:
            self.model.save(save_path + '.h5')
            print('[+] Model successfully saved to "{}"'.format(os.path.abspath(save_path)))

        except Exception as err:
            print('[-] Model could not be saved:')
            print(err)
 