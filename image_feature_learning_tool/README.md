# Learning deep image features with pretrained neural network

This is a demonstration to extract deep images features from Spatial Transcriptomics images using Google Inception-v3 pretrained on ImageNet.

The pretained Inception-v3 model was obtained from [Tensorflow Hub](https://tfhub.dev/). It takes in images with `299*299*3` size and output the last full connection layer with `2048` dimensions as features.

## Installation

Download conda, install and activate it. 

Create a new conda environment named 'muse' and install the required packages:

```bash
conda install mamba -n base -c conda-forge  
mamba create -n muse python=3.6
mamba env update -n muse --file muse.yml
```

## Copyright
Software provided as is under **MIT License**.

Copyright (c) 2020 Altschuler and Wu Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

