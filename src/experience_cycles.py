import networkx as nx
import pandas as pd
from tqdm import trange
import time
from pathlib import Path
import argparse

from graph import load_imdb, load_gplus, load_wiki, extract_random_subgraph, cycle_count
from cycles import estimate_cycles_smooth, estimate_cycles_smooth_without_count


BASE_DIR = Path(__file__).resolve().parent.parent
DIR_LOGS = BASE_DIR / "logs"

ALGOS = {
    "smooth": estimate_cycles_smooth,
    "inf": estimate_cycles_smooth_without_count,
}


def experience_cycle(graph, param):
    for _ in trange(param["nb_iter"]):
        extracted_graph = extract_random_subgraph(graph, param["graph_size"])
        true_cycle = cycle_count(extracted_graph, 4)
        start_time = time.time()
        count, bias, noise, cost = ALGOS[param["algorithm"]](
            extracted_graph,
            param["privacy_budget"],
            param["sample"],
        )
        result = pd.DataFrame(
            [
                {
                    **param,
                    "true_count": true_cycle,
                    "count": count,
                    "bias": bias,
                    "noise": noise,
                    "download_cost": cost,
                    "execution_time": time.time() - start_time,
                }
            ]
        )
        result.to_csv(
            DIR_LOGS
            / "{exp_name}_g{graph}_n{graph_size}_a{algorithm}_e{privacy_budget}_"
            "s{sample}_{date}.csv".format(**param, date=time.time()),
            index=False,
        )


def get_parser():
    parser = argparse.ArgumentParser(
        description="estimate the number of 4-cycles in a graph"
    )
    parser.add_argument(
        "-o", "--exp_name", type=str, default="test", help="name of the experiment"
    )

    parser.add_argument(
        "-g", "--graph", type=str, default="wiki", choices=["imdb", "gplus", "wiki"]
    )
    parser.add_argument(
        "-n",
        "--graph_size",
        type=int,
        default=7115,
        help="size of the graph to extract",
    )
    parser.add_argument(
        "-a", "--algorithm", type=str, default="smooth", help="algorithm to run"
    )
    parser.add_argument(
        "-e",
        "--privacy_budget",
        type=float,
        default=1,
        help="privacy budget of the algorithm",
    )
    parser.add_argument("-s", "--sample", type=int, default=32, help="sample factor")
    parser.add_argument("-i", "--nb_iter", type=int, default=1, help="number of runs")
    return parser


def get_graph(graph_name):
    if graph_name == "imdb":
        return load_imdb()
    elif graph_name == "gplus":
        return load_gplus()
    elif graph_name == "wiki":
        return load_wiki()
    else:
        raise ValueError("Graph {} is unknown.".format(graph_name))


if __name__ == "__main__":
    config = vars(get_parser().parse_args())
    graph = get_graph(config["graph"])
    experience_cycle(graph, config)
