import numpy as np


def blend_models(G_source, G_target):
    state_dict_source = G_source.synthesis.state_dict()
    state_dict_target = G_target.synthesis.state_dict()

    alphas = np.linspace(-3, 0, 15)

    for key in state_dict_target:
        if key[:1] != 'L':
            continue
        alpha = alphas[int(key.split('_')[0][1:])]
        if 'affine' in key:
            alpha = 0

        state_dict_target[key] = state_dict_source[key] * alpha + state_dict_target[key] * (1 - alpha)

    G_target.synthesis.load_state_dict(state_dict_target)
    return G_target
