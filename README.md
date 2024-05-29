# Design of Y-splitters using a fully unsupervised neural network

This code uses a deep convolutional neural network to design the Y-splitters with different split ratios. Our network is fully unsuopervised, which means that you do not need to pre-generate any structures before training.

This code is based on `tensorflow-1.15`, and the version of each package can be found at environment.yml. You can set up the environment using:
```sh
conda env create -f environment.yml
```

You can then train the model using:
```sh
python voxelmorph-waveguide_structure_calculate/scripts/tensorflow/train_waveguide.py
```

We provided one example for the training data in the folder `data_training`. You can buind your own training samples using `Untitled4_reduce_simulate_region.ipynb`.

After training, you can use `voxelmorph-waveguide_structure_calculate/scripts/tensorflow/test_waveguide.py` to generate the results for the test set.

To evaluate the power splitting accuracy and transmission efficiency, you can use the code in `Untitled5_test_reduce_simulate_region_small_interpolate_alltest.ipynb`.

If you have any comments or suggestions, please feel free to tell us. We appreciate all your suggestions.
