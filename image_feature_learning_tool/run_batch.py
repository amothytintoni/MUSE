
##################### Parameters

import os
import argparse
import glob

parser = argparse.ArgumentParser(
    prog='run_batch',
    usage='%(prog)s -i DIR -o PREFIX [options]',
    add_help=False
    )
required = parser.add_argument_group('Required arguments')
optional = parser.add_argument_group('Optional parameters')
general = parser.add_argument_group('General help')

general.add_argument(
    '-h', '--help', action='help', default=argparse.SUPPRESS,
    help='Show this help message and exit'
)
required.add_argument(
    '-i', '--in-dir', type=str, required=True, help='Path to images. Each image should be in .npy format with size 299 x 299 x 3.'
)
required.add_argument(
    '-o', '--out-prefix', type=str, required=True, help='Prefix to save Inception features. Will be suffixed with `_inception.py`'
)
optional.add_argument(
    '--chunk-size', type=int, default=1000, help='Chunk size to feed to Inception'
)

args = parser.parse_args()

print(args)

# Parse args
image_folder = args.in_dir
save_prefix = args.out_prefix
chunk_size = args.chunk_size
model_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'inception_v3'
)

##################### Import libs and helpers

import tensorflow.compat.v1 as tf
# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
import tensorflow_hub as hub
import numpy as np
import glob
from tqdm import tqdm

# Load saved inception-v3 model
module = hub.Module(model_path)

# images should be resized to 299x299
input_imgs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
features = module(input_imgs)

# Provide the file indices
# This can be changed to image indices in strings or other formats
image_no = len(glob.glob(f'./{image_folder}/Img*npy'))
print(f'Number of images: {image_no}')

# Generate chunk ids
image_ids = np.arange(image_no)
chunk_ids = np.array_split(image_ids, np.arange(chunk_size, len(image_ids), chunk_size))
num_chunks = len(chunk_ids)

# Feature bucket
fea_out = []

for chunk in tqdm(range(num_chunks), desc='Processing chunks'):

    curr_chunk_ids = chunk_ids[chunk]
    curr_chunk_size = len(curr_chunk_ids)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img_all = np.zeros([curr_chunk_size, 299, 299, 3])

        # Load all images and combine them as a single matrix
        # This loop can be changed to match the provided image information
        for i in range(curr_chunk_size):
            # Here, all images are stored in example_img and in *.npy format
            # if using image format, np.load() can be replaced by cv2.imread()
            file_name = f'./{image_folder}/Img_{curr_chunk_ids[i]}.npy'
            temp = np.load(file_name)
            temp = np.repeat(temp[:,:,None], 3, axis=2)
            temp2 = temp.astype(np.float32) / 255.0
            img_all[i, :, :, :] = temp2

        # Check if the image are loaded successfully.
        if (i == curr_chunk_size - 1):
            print('+++Successfully load all images+++')
        else:
            print('+++Image patches missing+++')

        # Input combined image matrix to Inception-v3 and output last layer as deep feature
        fea = sess.run(features, feed_dict={input_imgs: img_all})

        # Concat to bucket
        fea_out.append(fea)

# Stack all chunks
fea_out = np.row_stack(fea_out)

# Save inferred image features
np.save(f'{save_prefix}_inception.npy', fea_out)
