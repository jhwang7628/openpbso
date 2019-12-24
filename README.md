# Real-time Modal Sound Demo

This is a demo written for our KleinPAT paper. It runs modal sound synthesis in
real-time.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This will create a binary named `real_time_modal_sound_bin`.

## Required runtime data

The synthesizer uses preprocessed data to run the modal sound synthesis. You
will need the following data files.
* .obj file
* surface modes file
* modal material txt file
* FFAT maps folder built by KleinPAT

## Running the synthesizer

There are two ways of loading the data files. If everything is named properly
like the examples we provided, you can simply run

    ./real_time_modal_sound_bin -d <data_root_folder> -name <obj_name>

An example of `<obj_name>` would be `wine`. `<data_root_folder>` is where the
.obj file can be located.

Alternatively, you can also specify each required files/folder:

    ./real_time_modal_sound_bin -m <obj_file> -s <modes_file> -t <material_file> -p <ffat_maps_folder>

## Dependencies

The C++ dependencies are stl, eigen, [libigl](http://libigl.github.io/libigl/) and
the dependencies of the `igl::opengl::glfw::Viewer`.

We recommend you to install libigl using git via:

    git clone https://github.com/libigl/libigl.git
    cd libigl/
    git submodule update --init --recursive
    cd ..

If you have installed libigl at `/path/to/libigl/` then a good place to clone
this library is `/path/to/libigl-example-project/`.
