from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

SCALE = 1.5

# FONTSIZE = 22
# LABELSIZE = 16

TITLESIZE = 20
FONTSIZE = 15
LABELSIZE = 13
LEGENDSIZE = 10

# FONTSIZE = 12
# LABELSIZE = 10

TITLESIZE = int(SCALE * TITLESIZE)
FONTSIZE = int(SCALE * FONTSIZE)
LABELSIZE = int(SCALE * LABELSIZE)
LEGENDSIZE = int(SCALE * LEGENDSIZE)


def map_old_compression_ratio_to_new_compression_ratio(x):
    if isinstance(x, list):
        return [round((1 / (1 - i)) * 10) / 10 for i in x]
    else:
        return round((1 / (1 - float(x))) * 10) / 10


def pp(df, exp):
    df_exp = df[df.exp == exp]
    df_pprint = (
        df_exp.assign(
            open_layer=lambda ddf: ddf.hook_type.map(
                lambda x: {"pre": 0, "fwd": 1, "bwd": 2}[x]
            )
            .rolling(2)
            .apply(lambda x: x[0] == 0 and x[1] == 0)
        )
        .assign(
            close_layer=lambda ddf: ddf.hook_type.map(
                lambda x: {"pre": 0, "fwd": 1, "bwd": 2}[x]
            )
            .rolling(2)
            .apply(lambda x: x[0] == 1 and x[1] == 1)
        )
        .assign(
            indent_level=lambda ddf: (
                ddf.open_layer.cumsum() - ddf.close_layer.cumsum()
            )
            .fillna(0)
            .map(int)
        )
        .sort_values(by="call_idx")
        .assign(mem_diff=lambda ddf: ddf.mem_all.diff() // 2**20)
    )
    pprint_lines = [
        f"{'    ' * row[1].indent_level}{row[1].layer_type} {row[1].hook_type}  {row[1].mem_diff or ''}"
        for row in df_pprint.iterrows()
    ]
    for x in pprint_lines:
        print(x)


def plot_mem(
    df,
    colors,
    exps=None,
    exps_name=None,
    normalize_call_idx=True,
    # normalize_mem_all=True,
    normalize_mem_all=False,
    filter_fwd=False,
    return_df=False,
    output_file=None,
    color_bl="#EA580C",
):
    df = df.reset_index(drop=True)

    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots()

    for i_exp, exp in enumerate(exps):
        df_ = df[df.exp == exp]

        # clean
        df_ = df_.drop(df_[df["hook_type"] == "pre"].index)  # remove "pre" hooks
        df_ = df_.drop(df_[df["layer_type"] == "ReLU"].index)  # remove "ReLU" layers
        df_ = df_.reset_index(drop=True)

        if normalize_mem_all:
            df_.mem_all = (
                df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            )
        df_.mem_all = round(df_.mem_all / 1024 / 1024, 3)

        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[
                (df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")
            ]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]

        df_["xtick"] = df_["layer_type"] + "_" + df_["hook_type"]

        x = [i for i in range(len(df_))]
        y = list(df_["mem_all"])

        print(y)

        if i_exp == 0:
            plt.plot(x, y, label=exps_name[0][i_exp], color=color_bl, linewidth=3)
        else:
            plt.plot(
                x,
                y,
                label=exps_name[0][i_exp],
                color=colors[(i_exp - 1) * 3],
                linewidth=1,
            )
            # plt.plot(x, y, label=exps_name[i_exp], linewidth=1)

    xticks_label = list(df_["xtick"])
    xticks = [i for i in range(len(x))]

    # plt.xticks(xticks)
    # ax.set_xticklabels(xticks_label, rotation=45)

    xmax = df_[["mem_all"]].idxmax()[0]
    # plt.axvline(x=xmax, color="gray", linewidth=1.5, linestyle="--")
    plt.axvline(x=len(x) / 2, color="gray", linewidth=1.5, linestyle="--")

    # xtick = [len(x) / 4, len(x) / 2, 3 * len(x) / 4] # Input Output Input (Input in the middle)
    xtick = [0, len(x) / 2, len(x)]
    xtick_label = ["Input", "Output", "Input"]

    plt.xticks(xtick)
    ax.set_xticklabels(xtick_label)
    # ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(which="both", length=0)

    # ax.set_xticks([])
    ax.tick_params(axis="x", labelsize=FONTSIZE)
    ax.set_ylabel("Memory (MB)", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=LABELSIZE)

    # show a legend on the plot
    legend = ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    # legend_title = legend.get_title()
    # legend_title.set_fontsize(FONTSIZE)

    fig.set_size_inches((8, 3), forward=False)

    if "unet" in output_file:
        ax.set_ylim([0, 2000])
    else:
        # ax.set_ylim([0, 4000])
        # ax.set_ylim([0, 700])
        ax.set_ylim([0, 1000])

    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.LinearLocator(3))
    fig.suptitle(exps_name[1], fontsize=TITLESIZE, y=1.02)

    # Display a figure.
    # plt.show()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0)

    if return_df:
        return df_


# plt.xticks(x, labels, rotation='vertical')
