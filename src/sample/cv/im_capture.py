import cv2
import numpy as np
from PIL import ImageGrab

# set some global variables
is_clicked = False
coors = list()
loop = True
filename = 'tmp'

def run(filepath='tmp'):
    global img, loop, filename

    # set the file path if it was passed
    filename = filepath

    try:
        # create a window to hold the feed, and set up the mouse callback
        cv2.namedWindow('Feed', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Feed', click_and_crop)

        # infinitely monitor the screen (albeit somewhat slowly)
        while loop:
            # grabs a screenshot of the entire screen
            raw_grab = ImageGrab.grab()

            # converts that screenshot to a NumPy array (and array of numbers)
            img = np.array(raw_grab)

            # show the image on the display window we created earlier
            cv2.imshow('Feed', img)

            # if the user quits as with pressing 'q'
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        
        # close any open windows
        cv2.destroyAllWindows()

    # some basic error handling
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

    except Exception as err:
        cv2.destroyAllWindows()
        print(err)
        exit(1)

# a mouse callback that allows our mouse events to be registered and perform some actions
def click_and_crop(event, x, y, flags, param):
    global coors, is_clicked, loop

    # if the button is pressed
    if event == cv2.EVENT_LBUTTONDOWN and not is_clicked:
        coors.append((x, y))
        is_clicked = True

    # if the button is released
    elif event == cv2.EVENT_LBUTTONUP and is_clicked:
        coors.append((x, y))
        is_clicked = False

    # if we have two coordinates
    if len(coors) == 2 and not is_clicked:
        # take the selection and write it to file
        cv2.imwrite(filename + '.png', img[coors[0][1]:coors[1][1], coors[0][0]:coors[1][0]])
        
        # stop looping
        loop = False
