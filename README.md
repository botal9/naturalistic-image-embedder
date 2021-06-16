# naturalistic-image-embedder

### Pre-trained models
The pre-trained models for [embedding](https://disk.yandex.ru/d/Cn_5wbw_bQcugg) and [generator](https://disk.yandex.ru/d/QoCi6cz0ulZp8g) should be placed at "checkpoints" directory.

### Run

Execute `python3 sample.py` to run all samples. The results will be in the `/out` subdirectory.

Run in the following way on your own images:
```
python3 sample.py <background image path> <foreground image path> <out image path> <offset by X coordinate> <offset by Y coordinate> <naive | poisson_blending | color_transfer>
```

### Acknowledgements

Pre-trained [Places365-CNN model](naturalistic_image_embedder/third_party/io_classification/wideresnet18_places365.pth.tar) is provided by the Places Project authors. Visit [Project page](http://places2.csail.mit.edu) for the additional information. 

The original paper: [IEEE Transaction on Pattern Analysis and Machine Intelligence](http://places2.csail.mit.edu/PAMI_places.pdf).
