import os


def _depth_map(npyImage):
    pass


DEPTH_ESTIMATOR_SET_UP = False


def depth_map(image):
    global DEPTH_ESTIMATOR_SET_UP

    if not DEPTH_ESTIMATOR_SET_UP:
        bootstrapper = os.path.join(os.path.split(__file__)[0], 'bootstrap.py')
        exec(open(bootstrapper, 'r').read())
        DEPTH_ESTIMATOR_SET_UP = True

    return _depth_map(image)
