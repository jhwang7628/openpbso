# Real-time Modal Sound Demo

This is a demo written for our KleinPAT paper. It runs modal sound synthesis in
real-time.

## Dependencies

Below is a list of dependencies along with the version that are tested:
* [CMake](https://cmake.org/): 3.12.1
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page): 3.2.10
* [Libigl](http://libigl.github.io/libigl/): 2.1.0 (modified)
* [PortAudio](http://www.portaudio.com/): latest stable release
  [here](http://www.portaudio.com/archives/pa_stable_v190600_20161030.tgz)
* [Protobuf](https://developers.google.com/protocol-buffers): 3.7.1

We recommend you to install libigl using git via:

    git clone https://github.com/libigl/libigl.git
    cd libigl/
    git submodule update --init --recursive
    cd ..

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This will create a binary named `real_time_modal_sound_bin`. C++11 or above is
recommended.

Since the code uses serialized data, it is recommended to run the protocol
buffer compiler at the root directory:

    protoc --cpp_out=. ./ffat_map.proto

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

    ./real_time_modal_sound_bin -d <data_folder> -name <obj_name>

An example of `<obj_name>` would be `wine`. `<data_folder>` is where the
.obj file can be located.

Alternatively, you can also specify each required files/folder:

    ./real_time_modal_sound_bin -m <obj_file> -s <modes_file> -t <material_file> -p <ffat_maps_folder>

## Known issues

#### `fatal error: 'google/protobuf/port_def.inc' file not found`
This is likely due to older protoc version. Please try to upgrade the protocol
buffer.
