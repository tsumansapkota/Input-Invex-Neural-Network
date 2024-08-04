import numpy as np
from urllib import request
import gzip
import pickle
import os

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


class MNIST:

    def download_mnist(self, dirs="data/MNIST/"):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) #+ '/'
        
        for name in filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], dirs+name[1])
        print("Download complete.")

    def save_mnist(self, dirs="data/MNIST/"):
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) #+ '/'
        
        mnist = {}
        for name in filename[:2]:
            with gzip.open(dirs + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in filename[-2:]:
            with gzip.open(dirs + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(dirs + "mnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    def init(self):
        self.download_mnist()
        self.save_mnist()

    def load(self, dirs="data/MNIST/"):
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) #+ '/'

        with open(dirs + "mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


class FashionMNIST:

    def download_mnist(self, dirs="data/F-MNIST/"):
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) #+ '/'
        
        for name in filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], dirs+name[1])
        print("Download complete.")

    def save_mnist(self, dirs="data/F-MNIST/"):
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) #+ '/'
        
        mnist = {}
        for name in filename[:2]:
            with gzip.open(dirs + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in filename[-2:]:
            with gzip.open(dirs + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(dirs + "fmnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    def init(self):
        self.download_mnist()
        self.save_mnist()

    def load(self, dirs="data/F-MNIST/"):
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) #+ '/'

        with open(dirs + "fmnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
