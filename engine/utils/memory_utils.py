import sys
import psutil
import os
import gc
import torch
from torch.utils.checkpoint import checkpoint_sequential


def numpy2pytorch(a):
    return torch.from_numpy(a.copy()).to(torch.device("cuda"))


def pytorch2numpy(a):
    return a.detach().cpu().numpy()


def get_size_tensor_in_gpu(tensor):
    return tensor.element_size() * tensor.nelement()


def get_size_torch_size_in_gpu(size):
    # because each dimension is represented by a 64-bit (8-byte) integer.
    return len(size) * 8


def get_size_list_tensor_in_gpu(list_tensor):
    if isinstance(list_tensor, list):
        total = 0
        for tensor in list_tensor:
            if not isinstance(tensor, torch.Size):
                total += tensor.element_size() * tensor.nelement()
    else:
        total = list_tensor.element_size() * list_tensor.nelement()
    return total


def total_tensor_size(tensor_tuple):
    total_size = 0
    for tensor_list in tensor_tuple:
        for item in tensor_list:
            if isinstance(
                item, torch.Tensor
            ):  # Check if the object is a PyTorch tensor
                total_size += get_size_tensor_in_gpu(item)
            elif isinstance(item, torch.Size):
                total_size += get_size_torch_size_in_gpu(item)
            elif isinstance(item, list):  # Check if the object is a list
                for tensor in item:
                    if isinstance(tensor, torch.Tensor):
                        total_size += get_size_tensor_in_gpu(tensor)
                    elif isinstance(tensor, torch.Size):
                        total_size += get_size_torch_size_in_gpu(tensor)
                    else:
                        print(
                            f"Warning: Found a non-tensor or non-size object {tensor} in the nested list."
                        )
            else:
                print(
                    f"Warning: Found a non-tensor and non-list object {item} in the list."
                )
    return total_size


def format_e(n):
    a = "%E" % n
    return a.split("E")[0].rstrip("0").rstrip(".") + "e" + a.split("E")[1]


def round_decimal(arr, n_digits=2):
    return (arr * 10**n_digits).round() / (10**n_digits)


def print_memory(verbose=0):
    max_memory_allocated = round(
        torch.cuda.max_memory_allocated(device=None) / 1024 / 1024, 2
    )
    max_memory_cached = round(
        torch.cuda.max_memory_cached(device=None) / 1024 / 1024, 2
    )
    memory_allocated = round(torch.cuda.memory_allocated(device=None) / 1024 / 1024, 2)
    memory_cached = round(torch.cuda.memory_cached(device=None) / 1024 / 1024, 2)
    if verbose:
        print("-" * 60)
        print("max_memory_allocated: {0:.2f}MB".format(max_memory_allocated))
        print("max_memory_cached: {0:.2f}MB".format(max_memory_cached))
        print("memory_allocated: {0:.2f}MB".format(memory_allocated))
        print("memory_cached: {0:.2f}MB".format(memory_cached))
        print("-" * 60)
    return max_memory_allocated, max_memory_cached, memory_allocated, memory_cached


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def get_tensor_size_in_megabyte(tensor):
    return round(tensor.element_size() * tensor.nelement() / 1024 / 1024, 3)


def get_all_nested_tensors(tensors):
    res = []
    for val in tensors:
        if type(val) not in [list, set, tuple]:
            res.append(val)
        else:
            res.extend(get_all_nested_tensors(val))
    return res


def get_tuple_tensor_size_in_megabyte(tuple_tensors):
    s = 0
    list_tensors = get_all_nested_tensors(tuple_tensors)

    for tensor in list_tensors:
        if isinstance(tensor, torch.Tensor):
            s += get_tensor_size_in_megabyte(tensor)

    return s


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0**30  # memory use in GB...I think
    print("memory GB:", memoryUse)


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append(
            {
                "layer_idx": idx,
                "call_idx": call_idx,
                "layer_type": type(self).__name__,
                "exp": exp,
                "hook_type": hook_type,
                "mem_all": mem_all,
                "mem_cached": mem_cached,
            }
        )

    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, "pre", exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, "fwd", exp))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, "bwd", exp))
    hr.append(h)


def log_mem(model, inp, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f"exp_{len(mem_log)}"
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    # try:
    #     out = model(inp)
    #     loss = out.sum()
    #     loss.backward()
    # finally:
    #     [h.remove() for h in hr]

    #     return mem_log

    # try:
    out = model(inp)
    print("-" * 20)
    print_memory(verbose=1)
    loss = out.sum()
    print("-" * 20)
    print_memory(verbose=1)
    loss.backward()
    print("-" * 20)
    print_memory(verbose=1)

    [h.remove() for h in hr]

    return mem_log


def log_mem_compressed(model, inp, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f"exp_{len(mem_log)}"
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    # try:
    out = model("db3", 0.9, 3, inp)
    loss = out.sum()
    loss.backward()
    # finally:
    [h.remove() for h in hr]

    return mem_log


def log_mem_cp(model, inp, mem_log=None, exp=None, cp_chunks=3):
    mem_log = mem_log or []
    exp = exp or f"exp_{len(mem_log)}"
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = checkpoint_sequential(model, cp_chunks, inp)
        loss = out.sum()
        loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log


def log_mem_amp(model, inp, mem_log=None, exp=None):
    from apex import amp

    mem_log = mem_log or []
    exp = exp or f"exp_{len(mem_log)}"
    hr = []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    amp_model, optimizer = amp.initialize(model, optimizer)
    for idx, module in enumerate(amp_model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = amp_model(inp)
        loss = out.sum()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log


def log_mem_amp_cp(model, inp, mem_log=None, exp=None, cp_chunks=3):
    from apex import amp

    mem_log = mem_log or []
    exp = exp or f"exp_{len(mem_log)}"
    hr = []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    amp_model, optimizer = amp.initialize(model, optimizer)
    for idx, module in enumerate(amp_model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = checkpoint_sequential(amp_model, cp_chunks, inp)
        loss = out.sum()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log
