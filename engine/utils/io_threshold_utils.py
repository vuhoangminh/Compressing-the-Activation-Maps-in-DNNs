import copy
import json
import numpy as np
import os
import operator
from collections import defaultdict


class Experiment(object):
    def __init__(self, name):
        """Create an experiment"""
        super(Experiment, self).__init__()

        self.name = name
        self.logged = defaultdict(dict)
        self.meters = defaultdict(dict)

    def add_meters(self, tag, meters_dict):
        assert tag not in (self.meters.keys())
        for name, meter in meters_dict.items():
            self.add_meter(tag, name, meter)

    def add_meter(self, tag, name, meter):
        assert name not in list(
            self.meters[tag].keys()
        ), "meter with tag {} and name {} already exists".format(tag, name)
        self.meters[tag][name] = meter

    def log_meter(self, tag, name, n=1):
        meter = self.get_meter(tag, name)
        if name not in self.logged[tag]:
            self.logged[tag][name] = {}
        self.logged[tag][name][n] = meter.value()

    def log_meters(self, tag, n=1):
        for name, meter in self.get_meters(tag).items():
            self.log_meter(tag, name, n=n)

    def reset_meters(self, tag, num_items=1):
        meters = self.get_meters(tag)
        for name, meter in meters.items():
            meter.reset(num_items=num_items)
        return meters

    def get_meters(self, tag):
        assert tag in list(self.meters.keys())
        return self.meters[tag]

    def get_meter(self, tag, name):
        assert tag in list(self.meters.keys())
        assert name in list(self.meters[tag].keys())
        return self.meters[tag][name]

    def to_json(self, filename):
        os.system("mkdir -p " + os.path.dirname(filename))
        var_dict = copy.copy(vars(self))
        var_dict.pop("meters")
        for key in ("viz", "viz_dict"):
            if key in list(var_dict.keys()):
                var_dict.pop(key)
        with open(filename, "w") as f:
            json.dump(var_dict, f)

    def from_json(self, filename):
        with open(filename, "r") as f:
            var_dict = json.load(f)
        xp = Experiment("")
        xp.date_and_time = var_dict["date_and_time"]
        xp.logged = var_dict["logged"]
        # TODO: Remove
        if "info" in var_dict:
            xp.info = var_dict["info"]
        xp.name = var_dict["name"]
        return xp


class ListAvgMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_item=1):
        self.num_item = num_item
        self.reset()

    def reset(self, num_items=1):
        self.val = [0] * num_items
        self.avg = [0] * num_items
        self.sum = [0] * num_items
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = list(map(operator.add, self.sum, [element * n for element in val]))
        self.count += n
        self.avg = [element / self.count for element in self.sum]

    def value(self):
        return self.avg


def make_meters():
    meters_dict = {"value": ListAvgMeter()}
    return meters_dict


def main():
    logger = Experiment("threshold")
    logger.add_meters("energy", make_meters())

    meters = logger.reset_meters("energy", num_items=3)
    for epoch in range(3):
        val = list(np.random.randint(1000, size=3))

        print(val)
        meters["value"].update(val, n=1)
        logger.log_meters("energy", n=epoch)

    print(logger.logged)

    a = 2


if __name__ == "__main__":
    main()
