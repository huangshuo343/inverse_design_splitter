# Design Y-splitters using a fully unsupervised neural network with dimension expansion

This code uses a deep convolutional neural network to design the Y-splitters with different split ratios. Our network is fully unsuopervised, which means that you do not need to pre-generate any structures before training.

This code is based on `tensorflow-1.15`, and we have trained our model both on a NVIDIA GTX 3050 GPU (Windows 7) and a Tesla V100 GPU (Linux 3.10.0). It requires GPU memory for about 1.5 GB, and CPU memory for about 3 GB. Since not all GPUs support tensorflow-1.15, please check your GPU before running the code. For NVIDIA GTX 3050 GPU, the tensorflow is from https://github.com/Fannhhyy/tensorflow1.15-whl-and-cpp-api-for-win-and-rtx3090.

The version of each package can be found at environment.yml. You can set up the environment using:
```sh
conda env create -f environment.yml
```

You can then train the model using:
```sh
python voxelmorph-waveguide_structure_calculate/scripts/tensorflow/train_waveguide.py
```
The list of the training samples is in the file `training_set_reduce_simulate_area_symmetric_separatechannel12.txt`.

It is noticeable that you need to manually change the binarization wight in the line 43 of ``layers.py`` after every 50 epochs:`n_predict_allresults = (n_predict_allresults - 0.5) * 150`. For epochs 0-50, the binarization wight number 150 should be 5, then 25, 50, 100, 150 for epochs 51-100, 101-150, 151-200 and 201-250.

We provided he training samples in the folder `data_training`. You can buind your own training samples using `Untitled4_reduce_simulate_region.ipynb`.

After training, you can use `voxelmorph-waveguide_structure_calculate/scripts/tensorflow/test_waveguide.py` to generate the results for the test set. You can use the code in `Untitled4_reduce_simulate_region_test.ipynb` to generate the inputs for testing. We provided our model after 250 epochs in `models`, 6 test inputs in `test_inputs`, and the generated structures of the 6 test inputs in `test_results`.

To evaluate the power splitting accuracy and transmission efficiency, you can use the code in `Untitled5_test_reduce_simulate_region_small_interpolate_alltest.ipynb`.

The training time for 250 epochs is 280 seconds, and the time for generating 1 test structure is about 1 second.

If you have any comments or suggestions, please feel free to tell us. We appreciate all your suggestions.
