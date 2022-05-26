def blend_models(G_source, G_target):
    state_dict_source = G_source.synthesis.state_dict()
    state_dict_target = G_target.synthesis.state_dict()

    alphas = [0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.5, 0.7, 0.8, 0.8, 0.8, 0.8, 1]

    for key in state_dict_target:
        if key[:1] != 'L':
            continue
        alpha = alphas[int(key.split('_')[0][1:])]
        if 'affine' in key:
            alpha = 0

        state_dict_target[key] = state_dict_source[key] * alpha + state_dict_target[key] * (1 - alpha)

    G_target.synthesis.load_state_dict(state_dict_target)
    return G_target
