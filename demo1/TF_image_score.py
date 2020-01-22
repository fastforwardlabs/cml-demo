# Tensorflow on Spark
#
# Before running:
# pip install -r /home/cdsw/utils/requirements3.txt
# close session, start a new session in the same project

MODEL_URL = 'https://ml-field.s3-us-west-2.amazonaws.com/demo/tensorflow/model/inception-2015-12-05.tgz'
MODEL_DIR = '/tmp/imagenet'
IMAGE_DIR = 's3://ml-field/demo/tensorflow/data'

# Number of images per batch.
# 1 batch corresponds to 1 RDD row.
image_batch_size = 4

num_top_predictions = 5

import numpy as np
#import random
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import os.path
import re
import sys
import tarfile
from pyspark.sql import SparkSession
from subprocess import Popen, PIPE, STDOUT

def maybe_download_and_extract():
  """Download and extract model tar file."""
  from six.moves import urllib
  dest_directory = MODEL_DIR
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = MODEL_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    filepath2, _ = urllib.request.urlretrieve(MODEL_URL, filepath)
    print("filepath2", filepath2)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  else:
    print('Data already downloaded:', filepath, os.stat(filepath))

maybe_download_and_extract()

model_path = os.path.join(MODEL_DIR, 'classify_image_graph_def.pb')
with tf.gfile.GFile(model_path, 'rb') as f: 
  model_data = f.read()

sc = SparkSession\
    .builder\
    .appName("S3 Image Scoring")\
    .config("spark.executor.memory", "4g")\
    .config("spark.executor.instances", 2)\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field/demo/")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "us-west-2")\
    .getOrCreate()._sc

from IPython.core.display import HTML
HTML('<a href="http://spark-{}.{}">Spark UI</a>'.format(os.getenv("CDSW_ENGINE_ID"),os.getenv("CDSW_DOMAIN")))    

model_data_bc = sc.broadcast(model_data)

class NodeLookup(object):
  """Converts integer node IDs to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          MODEL_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          MODEL_DIR, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

node_lookup = NodeLookup().node_lookup
# Broadcast node lookup table to use on Spark workers
node_lookup_bc = sc.broadcast(node_lookup)

def run(cmd):
  p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
  return p.stdout.read()


#build image index from S3 bucket list
#assuming img_dir is something like 's3://ml-field/demo/tensorflow/data'
def read_file_index_s3(img_dir):
  import boto3
  from botocore.handlers import disable_signing
  #import s3fs
  
  s3 = boto3.resource('s3')
  s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
  
  if img_dir.split('/')[0] == 's3:': 
    bucket_name = img_dir.split('/')[2]
    bucket = s3.Bucket(bucket_name)
    prefix = '/'.join(img_dir.split('/')[3:]) + '/'
  else: 
    print("img_dir is not an S3 URL")
    exit()
  
  input_data = []
  ## List objects within a given prefix
  for obj in bucket.objects.filter(Delimiter='/', Prefix=prefix):
    file_name = os.path.basename(obj.key)
    #input_data.append((file_name, 's3a://' + bucket_name + '/' + obj.key))
    input_data.append((file_name, 'https://' + bucket_name + '.s3-us-west-2.amazonaws.com/' + obj.key))
    #https://ml-field.s3-us-west-2.amazonaws.com/demo/tensorflow/data/2737866473_7958dc8760.jpg
    """Return the list without the directory name"""
    """Should do some checking to make sure its actually a JPG"""
  input_data = input_data[1:]
  return [input_data[i:i+image_batch_size] for i in range(0,len(input_data), image_batch_size)]

batched_data = read_file_index_s3(IMAGE_DIR)


def read_file_index():
  from six.moves import urllib
  content = urllib.request.urlopen(IMAGES_INDEX_URL)
  data = content.read()
  lines = [l for l in data.split(b'\n') if l != ""]
  input_data = [tuple(elts.split()) for elts in lines]
  return [input_data[i:i+image_batch_size] for i in range(0,len(input_data), image_batch_size)]

#batched_data = read_file_index()
#print "There are %d batches" % len(batched_data)

def run_inference_on_image(sess, img_id, img_url, node_lookup):
  """Download an image, and run inference on it.

  Args:
    image: Image file URL

  Returns:
    (image ID, image URL, scores),
    where scores is a list of (human-readable node names, score) pairs
  """
  from six.moves import urllib
  #from urllib2 import HTTPError
  try:
    image_data = urllib.request.urlopen(img_url, timeout=1.0).read()
  except:
    return (img_id, img_url, None)
  # Some useful tensors:
  # 'softmax:0': A tensor containing the normalized prediction across
  #   1000 labels.
  # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
  #   float description of the image.
  # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
  #   encoding of the image.
  # Runs the softmax tensor by feeding the image_data as input to the graph.
  softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
  try:
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
  except:
    # Handle problems with malformed JPEG files
    return (img_id, img_url, None)
  predictions = np.squeeze(predictions)
  top_k = predictions.argsort()[-num_top_predictions:][::-1]
  scores = []
  for node_id in top_k:
    if node_id not in node_lookup:
      human_string = ''
    else:
      human_string = node_lookup[node_id]
    score = predictions[node_id]
    scores.append((human_string, score))
  return (img_id, img_url, scores)

def apply_inference_on_batch(batch):
  """Apply inference to a batch of images.

  We do not explicitly tell TensorFlow to use a GPU.
  It is able to choose between CPU and GPU based on its guess of which will be faster.
  """
  with tf.Graph().as_default() as g:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model_data_bc.value)
    print("Graph def done")
    tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
      print(batch)
      labeled = [run_inference_on_image(sess, img_id, img_url, node_lookup_bc.value) for (img_id, img_url) in batch]
      print(labeled)
      return [tup for tup in labeled if tup[2] is not None]

urls = sc.parallelize(batched_data)
labeled_images = urls.flatMap(apply_inference_on_batch)

local_labeled_images = labeled_images.collect()

#sc.stop()

#local_image = random.choice(local_labeled_images)
from IPython.display import Image
from IPython.core.display import HTML
Image(url=local_labeled_images[11][1])
print(local_labeled_images[11][2])

#To be saved to RDBMS or SOLR
[[i[1] for i in local_labeled_images], [i[2][0] for i in local_labeled_images]]
