#!/usr/bin/env python
# coding: utf-8
import math
import json
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp

def run(i_t, J, L, N_ITER, MODEL):
    
    model = MODEL
    
    edge_colors = []
    for i in range(L):
        edge_colors.append([i, (i+1)%L, 1])
        edge_colors.append([i, (i+2)%L, 2])

    g = nk.graph.Graph(edges=edge_colors)
    sigmaz = [[1, 0], [0, -1]]
    mszsz = (np.kron(sigmaz, sigmaz))
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

    bond_operator = [
        (J[0] * mszsz).tolist(),
        (J[1] * mszsz).tolist(),
        (-J[0] * exchange).tolist(),  
        (J[1] * exchange).tolist(),
    ]

    bond_color = [1, 2, 1, 2]
    hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)
    op = nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max = 2)

    vs = nk.vqs.MCState(sa, model, n_samples=1008)

    opt = nk.optimizer.Sgd(learning_rate=0.01)

    sr = nk.optimizer.SR(diag_shift=0.01)

    gs = nk.VMC(hamiltonian=op, optimizer=opt, variational_state=vs, preconditioner=sr)
    
    sf = []
    sites = []
    structure_factor = nk.operator.LocalOperator(hi, dtype=complex)
    for i in range(0, L):
        for j in range(0, L):
            structure_factor += (nk.operator.spin.sigmaz(hi, i)*nk.operator.spin.sigmaz(hi, j))*((-1)**(i-j))/L

    path = 'logs/'+ 'n_14_' + str(i_t) + '_test'
    gs.run(out=path, n_iter=N_ITER, obs={'Structure Factor': structure_factor})    


class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128, 
                     kernel_init=nn.initializers.normal(stddev=0.01), 
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x

radl = []
sinl = []
cosl = []
for i_t in range(0,360):
    rad = math.radians(i_t); sin = math.sin(rad);  cos = math.cos(rad);
    radl.append(rad);  sinl.append(sin); cosl.append(cos);
    print(i_t,rad,cos,sin)  
    J      = [cos,sin]
    L      = 14   
    N_ITER = 600
    MODEL  = FFNN()
    run(i_t, J, L, N_ITER, MODEL)


#J  = [1, 0.2]; 
