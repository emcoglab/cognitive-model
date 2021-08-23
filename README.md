# Spreading Activation model
Prototype for the linguistic spreading-activation component of the cognitive model.

## Installation

Make sure to rebuild the cythonised bits for the machine on which you'll be running it:

    python setup.py build_ext --inplace

Then move the created file (called something like `maths_core.cpython-37m-darwin.so`) into `utils/`.
