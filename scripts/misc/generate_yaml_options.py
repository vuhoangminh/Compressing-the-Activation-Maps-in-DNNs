import os
import yaml
import pprint
import itertools
from engine.utils.path_utils import get_path_yaml


pp = pprint.PrettyPrinter(indent=4)


def generate_compressed_layers(l=["conv1", "relu1", "conv2", "maxpool"]):
    C = []
    for i in range(0, len(l) + 1):
        c = list(itertools.combinations(l, i))
        for j in c:
            C.append("-".join(j))
    C.append("all")
    return C


def generate_yaml(
    project="mnist",
    model_name="mnistnet_compressed",
    compression_method="wt",
    compression_parameters={"wave": "db3", "compression_ratio": 0.9, "n_levels": 3},
    compressed_layers="all",
):
    dir_yaml = get_path_yaml(
        model_name, compression_method, compression_parameters, compressed_layers
    )
    path_yaml = dir_yaml + ".yaml"
    save_dir = "database/options/{}".format(project)
    save_dir = os.path.join(save_dir, path_yaml)
    stream = {
        "logs": {"dir_logs": "database/logs/{}/{}".format(project, dir_yaml)},
        "data": {"dir": "database/data", "dataset": project},
        "model": {
            "arch": model_name,
            "compression_method": compression_method,
            "wave": compression_parameters["wave"],
            "compression_ratio": compression_parameters["compression_ratio"],
            "n_levels": compression_parameters["n_levels"],
            "compressed_layers": compressed_layers,
        },
        "optim": {
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 80,
            "patience": 10,
            "early_stopping": 80,
        },
    }
    with open(save_dir, "w") as f:
        yaml.dump(stream, f, default_flow_style=False, sort_keys=False)

    cmd = "python main.py --path_opt database/options/{}/{} --is_test 0".format(
        project, path_yaml
    )

    print(cmd)

    return cmd


def main():
    """cifar10"""
    # for wave in filters:
    #     generate_yaml(
    #         project="cifar10",
    #         model_name="mobilenet_compressed_two_layers",
    #         wave=wave,
    #         compression_ratio=0.9,
    #     )

    # for compression_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    #     generate_yaml(
    #         project="cifar10",
    #         model_name="mobilenet_compressed_two_layers",
    #         wave="db3",
    #         compression_ratio=compression_ratio,
    #     )

    # for wave in filters:
    #     generate_yaml(
    #         project="cifar10",
    #         model_name="mobilenet_compressed_six_layers",
    #         wave=wave,
    #         compression_ratio=0.9,
    #     )

    # for compression_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    #     generate_yaml(
    #         project="cifar10",
    #         model_name="mobilenet_compressed_six_layers",
    #         wave="db3",
    #         compression_ratio=compression_ratio,
    #     )

    # for wave in filters:
    #     generate_yaml(
    #         project="cifar10",
    #         model_name="mobilenet_compressed_two_layers_2b",
    #         wave=wave,
    #         compression_ratio=0.9,
    #     )

    # for compression_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    #     generate_yaml(
    #         project="cifar10",
    #         model_name="mobilenet_compressed_two_layers_2b",
    #         wave="db3",
    #         compression_ratio=compression_ratio,
    #     )

    # for wave in filters:
    #     generate_yaml(
    #         project="cifar10",
    #         model_name="mobilenet_compressed_two_layers_2c",
    #         wave=wave,
    #         compression_ratio=0.9,
    #     )

    # for compression_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    #     generate_yaml(
    #         project="cifar10",
    #         model_name="mobilenet_compressed_two_layers_2c",
    #         wave="db3",
    #         compression_ratio=compression_ratio,
    #     )

    # for wave in filters:
    #     generate_yaml(project='mnist',
    #                   model_name='mobilenet_compressed_two_layers',
    #                   wave=wave,
    #                   compression_ratio=0.9)

    # for compression_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    #     generate_yaml(project='mnist',
    #                   model_name='mobilenet_compressed_two_layers',
    #                   wave='db3',
    #                   compression_ratio=compression_ratio)

    # for compression_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    #     generate_yaml(project='mnist',
    #                   model_name='mnistnet_compressed',
    #                   wave='db3',
    #                   compression_ratio=compression_ratio)

    # for wave in filters:
    #     generate_yaml(project='mnist',
    #                   model_name='mnistnet_compressed_one_layer',
    #                   wave=wave,
    #                   compression_ratio=0.9)

    # for compression_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    #     generate_yaml(project='mnist',
    #                   model_name='mnistnet_compressed_one_layer',
    #                   wave='db3',
    #                   compression_ratio=compression_ratio)

    """mnist"""
    # filters = ["bior1.1", "bior2.2", "coif3", "db1", "db3", "db5", "rbio1.3", "sym3"]
    filters = ["sym5", "db3", "haar"]
    compression_methods = ["wt", "random"]
    compression_ratios = [0.3, 0.6, 0.9, 0.99, 0.999]
    for model_name in ["mnistnet_compressed"]:
        for compression_method in compression_methods:
            for wave in filters:
                for n_levels in [3]:
                    for compressed_layers in generate_compressed_layers(
                        ["conv1", "relu1", "conv2", "maxpool"]
                    ):
                        for compression_ratio in compression_ratios:
                            cmd = generate_yaml(
                                project="mnist",
                                model_name=model_name,
                                compression_method=compression_method,
                                compression_parameters={
                                    "wave": wave,
                                    "compression_ratio": compression_ratio,
                                    "n_levels": n_levels,
                                },
                                compressed_layers=compressed_layers,
                            )

                        print("\n" * 2)


if __name__ == "__main__":
    main()

    # generate_compressed_layers()

    """
    @TODO
    number of compressed layers: 1, 2
    compression ratio: 0.01, 0.03, 0.1, 0.3, 0.6, 0.99
    wave: 'bior1.1', 'bior2.2', 'coif3', 'db1', 'db3', 'db5', 'rbio1.3', 'sym3'
    number of levels is also a hyper-parameter: 2, 3, 4, 5, 6
    training time per epoch (Fig 7) + confidence intervals
    """
