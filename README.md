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

## Run

From within the `build` directory just issue:

    ./simulator -m <mesh> -s <surface_mode> -t <material> [-n <num_offline_samples>] [-o <offline_samples_output>] [-l <offline_samples_sec]

If no optional arguments, a glfw app should launch displaying the mesh with
interactive sound demo. Shift+left click to sample the sound from the vertex
under mouse cursor.

## Dependencies

The dependencies are stl, eigen, [libigl](http://libigl.github.io/libigl/) and
the dependencies of the `igl::opengl::glfw::Viewer`, and
[PortAudio](http://www.portaudio.com/) for sound.

We recommend you to install libigl using git via:

    git clone https://github.com/libigl/libigl.git
    cd libigl/
    git submodule update --init --recursive
    cd ..

If you have installed libigl at `/path/to/libigl/` then a good place to clone
this library is `/path/to/libigl-example-project/`.
