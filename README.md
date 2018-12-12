# Modal Classification

This is a CS221 class project at Stanford, which aims at jointing classifying
object shape and material from input modal sound.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This should find and build the dependencies and create a `simulator` binary.

## Run Modal Sound Simulator

Before running anything, we will make a symlink to the data folder.

    cd code
    ln -s ../data .

From within the `build` directory just issue:

    ./simulator -m <mesh> -s <surface_mode> -t <material> [-n <num_offline_samples>] [-o <offline_samples_output>] [-l <offline_samples_sec]

If no optional arguments, a glfw app should launch displaying the mesh with
interactive sound demo. Shift+left click to sample the sound from the vertex
under mouse cursor.

An example with the GUI is (run in `build` directory)

    ./simulator -m wine/wine.tet.obj -s wine/wine_surf.modes -t ../materials/ceramics.txt

And an example with just the offline sampling is

    ./simulator -m ../data/objects/wine/wine.tet.obj -s ../data/objects/wine/wine_surf.modes -t ../data/materials/ceramics.txt -n 10

## Run Training

Alters install paths from `python/create_training_set.py`.
From within the `data` directory, run

    python ../python/create_training_set.py

This should create a set of binary data files in the `data/dataset4` directory.
Then, within the dataset directory, run

    python ../../python/convert_features_to_binary.py

to compute the features and store in binary format. Then run

    python -u ../../python/train.py | tee train.log

This should read the binary data, produce wav files for each sample, and compute
features automatically. It will also split the test set and estimate the error
for the learned models on the test set.

## Dependencies

The C++ dependencies are stl, eigen, [libigl](http://libigl.github.io/libigl/) and
the dependencies of the `igl::opengl::glfw::Viewer`, and
[PortAudio](http://www.portaudio.com/) for sound.
We recommend you to install libigl using git via:

    git clone https://github.com/libigl/libigl.git
    cd libigl/
    git submodule update --init --recursive
    cd ..

If you have installed libigl at `/path/to/libigl/` then a good place to clone
this library is `/path/to/libigl-example-project/`.

 The python dependencies are
[scikit-learn](https://scikit-learn.org/stable/) and
[pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis), both can be
installed using `pip`.
