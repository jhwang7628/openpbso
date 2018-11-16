#!/usr/bin/env python
import numpy as np
import struct,sys
with open(sys.argv[1], 'r') as stream:
    N_samples, N_steps = struct.unpack('ii', stream.read(8))
    print N_samples, N_steps
    for sample in range(N_samples):
        vid = struct.unpack('i', stream.read(4))
        data = np.array(struct.unpack('%uf' %(N_steps), stream.read(N_steps*4)))
        print vid, data

