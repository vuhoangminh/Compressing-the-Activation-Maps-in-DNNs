import operator


def is_first_epoch(T):
    if not T:  # if T is an empty list
        return True
    else:
        return False


def get_threshold_factor_list(E_prev, factor_up=1.1, factor_down=0.9):
    if len(E_prev) > 1:
        e_prev = E_prev[-2]
        e = E_prev[-1]
        threshold_factor_list = list()

        for i in range(len(e_prev)):
            try:
                new, old = e[i], e_prev[i]
                if new > old:
                    threshold_factor_list.append(factor_up)
                elif new < old:
                    threshold_factor_list.append(factor_down)
                else:
                    threshold_factor_list.append(1)
            except:
                threshold_factor_list.append(1)

        return threshold_factor_list

    elif len(E_prev) == 1:
        threshold_factor_list = list()
        e_prev = E_prev[-1]
        for i in range(len(e_prev)):
            threshold_factor_list.append(1)
        return threshold_factor_list

    else:
        return None


def get_last_two_E_prev(E_prev, layer):
    if E_prev is None:
        return None
    elif len(E_prev) == 1:
        return [E_prev[-1][layer]]
    else:
        E_last_two = list()
        E_last_two.append(E_prev[-1][layer])
        E_last_two.append(E_prev[-2][layer])
        return E_last_two


def get_number_layers(x):
    return len(x[0])


def detach_E(E_batch_):
    E_batch = []
    for i in E_batch_:
        E_batch_i = []
        for j in i:
            E_batch_j = []
            for k in j:
                E_batch_j.append(k.detach())
            E_batch_i.append(E_batch_j)
        E_batch.append(E_batch_i)

    return E_batch


def update_energy_per_batch(E_batch, E):
    if E_batch is None:
        E_batch_new = detach_E(E)
    else:
        E_batch = detach_E(E_batch)
        E = detach_E(E)
        E_batch_new = list()
        l = get_number_layers(E)
        for i in range(l):
            E_batch_item = E_batch[0][i]
            E_item = E[0][i]
            energy_l = list(map(operator.add, E_batch_item, E_item))
            E_batch_new.append(energy_l)
        E_batch_new = [E_batch_new]
    return E_batch_new


def update_threshold_per_batch(T_batch, T, batch=1):
    if T_batch is None:
        T_batch_new = detach_E(T)
    else:
        T_batch = detach_E(T_batch)
        T = detach_E(T)
        T_batch_new = list()
        l = get_number_layers(T)
        for i in range(l):
            T_item = T[0][i]
            threshold_l_prev = T_batch[0][i]
            T_batch_sum = [j * batch for j in threshold_l_prev]
            T_batch_sum = list(map(operator.add, T_batch_sum, T_item))
            threshold_l = [j / (batch + 1) for j in T_batch_sum]
            T_batch_new.append(threshold_l)
        T_batch_new = [T_batch_new]
    return T_batch_new


def update_per_epoch(x_epoch, x_batch):
    if x_epoch is None:
        x_epoch = x_batch
    else:
        x_epoch = x_epoch + x_batch
    return x_epoch
