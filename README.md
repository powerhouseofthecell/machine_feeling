# :smiley: machine_feeling :smiley:
An Emoji-ful introduction to computer vision and machine learning.

-----

### Introduction :smirk:
This small project is intended as a beginning to some of the fundamental ideas of machine learning and computer vision. Specifically, this is a look at the problem of categorizing images (along with some potential solutions).

Our goal in this project is to walk through how something like this may work.

-----

### Goal :relaxed:
The problem herein is that we want to be able to identify whether the emoji we see on the Internet is positive :blush:, neutral :neutral_face:, or negative :angry:. In this (somewhat contrived) scenario, we, the enterprising young computer scientists, have decided to build software to perform several tasks in accomplishing this goal:
1. We need a way to "take a picture" or photograph something within our computer (i.e. something we see on the Internet). We could do this by using our phone or a screenshot software. But we are grade-A hipsters, so we'll be using OpenCV, a computer vision (*buzzword!*) platform, to build this ourselves.
2. This picture should then be plopped into some machine-learning algorithm (*yay! more buzzwords!*) that will tell us (hopefully), which if our emoji is looking positive, negative or somewhere in between. 

-----

### Some Notes :grin:
- Yes, this is an arguably absurdly contrived example on which to learn some basic machine learning and computer vision. 
- Yes, there are likely better ways to do this than using machine learning, especially since we can usually tell what emotion an emoji is already
- Given the above two things, we designed the project this way because it seemed fun and entertaining
- Please do not take this project in any way, shape, or form as a representation of *all* that either machine learning or computer vision are capable - this barely scratches the surface of either field

-----

### To Get Started
- First, I would recommend using some form of dedicated environment, as with virtualenv, but if you don't wish to, that's ok too
- The rest is as easy as installing the requirements with ```pip install -r requirements.txt```
- Then, navigate to the directory containing run (assuming you're in the machine_feeling/ directory, ```cd src; cd sample``` and run "run.py" ```./run.py -h```
  - If you receive a message about your ability to run that file, go ahead and change the permissions ```chmod +x run.py``` and you should be able to run it

----- 

### Conclusion :heart:
We sincerely hope that you enjoy working through this project and go on to pursue your interest in both machine learning and computer vision!

If you find a bug, a typo, or simply wish to contribute, please do make a pull request with your edit!

Sincerely,


potc
