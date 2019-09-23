from .vgg import *

'''
vgg model zoo:
* vgg11
* vgg13
* vgg16
* vgg19
'''

def select_model(name):
    lname=name.lower()
    if lname=='vgg':
        return VGG