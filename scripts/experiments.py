from itertools import product

# ***** SACCADE EXPERIMENTS GRID SEARCH *****

DURATION_EXPERIMENTS = False


# Grid parameters
learning_rates = [0.01, 0.1, 0.001]
batch_sizes = [64, 128, 256]
weight_decays = [0, 0.0001]


fixed_params = dict(
    training="true",
    final_testing="false",
    subset="false",
    nworkers=4,
)

seeds = [42, 84, 128, 256, 512, 1024]

EXPERIMENTS: list[dict] = []

for i, (lr, bs, wd) in enumerate(product(learning_rates, batch_sizes, weight_decays)):
    seed = seeds[i % len(seeds)]

    EXPERIMENTS.append(
        {
            **fixed_params,
            "learning_rate": lr,
            "batch_size": bs,
            "weight_decay": wd,
            "epochs": 30,
            "seed": seed,
        }
    )
# ***** DURATION EXPERIMENTS GRID SEARCH *****

if DURATION_EXPERIMENTS:
    alpha_beta = [(1, 3), (2, 4), (2, 6)]

    delta_vals = [0.05]
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [128]
    weight_decays = [0, 0.001]
    seeds = [42, 84, 128, 256, 512, 1024]

    EXPERIMENTS: list[dict] = []

    for i, ((alpha, beta), delta, lr, bs, wd) in enumerate(
        product(alpha_beta, delta_vals, learning_rates, batch_sizes, weight_decays)
    ):
        seed = seeds[i % len(seeds)]

        EXPERIMENTS.append(
            {
                **fixed_params,
                "alpha_g": alpha,
                "beta_g": beta,
                "delta_g": delta,
                "learning_rate": lr,
                "batch_size": bs,
                "weight_decay": wd,
                "epochs": 30,
                "seed": seed,
            }
        )
