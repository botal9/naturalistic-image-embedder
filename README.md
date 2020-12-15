# naturalistic-image-embedder

### Run

Execute `python3 sample.py` to run all samples. The results will be in the `/out` subdirectory.

Run in the following way on your own images:
```
python3 sample.py <background image path> <foreground image path> <out image path> <offset by X coordinate> <offset by Y coordinate> <naive | poisson_blending | color_transfer>
```

### Troubleshooting

```
OpenEXR.cpp:36:10: fatal error: ImathBox.h: No such file or directory
#include <ImathBox.h>
^~~~~~~~~~~~
compilation terminated.
error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```
If you see something like this while during the requirements installation, install the "libopenexr-dev" library.
```
sudo apt-get install libopenexr-dev
sudo apt-get install openexr
```

### Acknowledgements

Pre-trained [Places365-CNN model](naturalistic_image_embedder/third_party/io_classification/wideresnet18_places365.pth.tar) is provided by the Places Project authors. Visit [Project page](http://places2.csail.mit.edu) for the additional information. 

The original paper: [IEEE Transaction on Pattern Analysis and Machine Intelligence](http://places2.csail.mit.edu/PAMI_places.pdf).
