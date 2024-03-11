# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import os
import ntpath
from engine.utils.print_utils import print_separator


def get_project_dir(path, project_name):
    paths = path.split(project_name)
    return paths[0] + project_name


def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()
    return folders


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]


def make_dir(dir):
    if not os.path.exists(dir):
        print_separator()
        print("making dir", dir)
        os.makedirs(dir)


def get_path_yaml(
    model_name="mnistnet_compressed",
    compression_method="wt",
    compression_parameters={"wave": "db3", "compression_ratio": 0.9, "n_levels": 3},
    compressed_layers="all",
):
    if compression_method == "wt":
        dir_yaml = "{}_{}_{}_{}_{}_{}".format(
            model_name,
            compression_method,
            compression_parameters["wave"],
            str(compression_parameters["compression_ratio"]),
            str(compression_parameters["n_levels"]),
            compressed_layers,
        )
    elif compression_method == "th":
        dir_yaml = "{}_{}_{}_{}".format(
            model_name,
            compression_method,
            str(compression_parameters["compression_ratio"]),
            compressed_layers,
        )
    elif compression_method == "dct":
        dir_yaml = "{}_{}_{}_{}".format(
            model_name,
            compression_method,
            str(compression_parameters["compression_ratio"]),
            compressed_layers,
        )
    return dir_yaml


def main():
    output_dir = "/mnt/sda2/3DUnetCNN_BRATS/projects/pros/database/prediction/pros_2018_is-256-256-128_crop-0_bias-0_denoise-0_norm-11_hist-0_ps-128-128-128_segnet3d_crf-0_loss-dice_xent_aug-1_model/validation_case_956"


if __name__ == "__main__":
    main()
