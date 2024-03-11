import torch
import pickle


def numpy2pytorch(a):
    return torch.from_numpy(a.copy()).to(torch.device("cuda"))


def pytorch2numpy(a):
    return a.detach().cpu().numpy()


def pickle_dump_pytorch_tensor(item, out_file):
    item = pytorch2numpy(item)
    pickle_dump(item, out_file)


def pickle_load_pytorch_tensor(in_file):
    item = pickle_load(in_file)
    return numpy2pytorch(item)


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)
