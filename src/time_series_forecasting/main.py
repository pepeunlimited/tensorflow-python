#
#  main.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import requests
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import matplotlib as mpl


def main():
    print(sys.version)
    print(requests.__version__)
    print(tf.__version__)
    print(pd.__version__)
    print(np.__version__)
    print(keras.__version__)
    print(mpl.__version__)


if __name__ == "__main__":
    main()
