from pathlib import Path
from typing import Any
import tensorneko_util as N
import pandas as pd
from rich.console import Console
from chatgpt import TreeGenerator

console = Console()
temp_dir = "temp"
Path(temp_dir).mkdir(exist_ok=True, parents=True)


def merge_tree(current_level: int) -> dict[str, dict[str, Any | None]]:
    current_tree = N.io.read.json(f"{temp_dir}/tree_{current_level}.json")
    if current_level == 0:
        return {_k: {sub_group: None for sub_group in _v} for _k, _v in current_tree.items()}

    previous_tree = merge_tree(current_level - 1)
    merged_tree = {}
    for group, sub_groups in current_tree.items():
        merged_tree[group] = {}
        for sub_group in sub_groups:
            merged_tree[group][sub_group] = previous_tree[sub_group]

    return merged_tree


def generate_tree_bottom_up(all_leafs: list[str], init_tree: dict[str, list[str]] | None = None, output_file: str = "final_tree.json"):
    # now we're using the groups of the tree generator to generate higher level groups
    i = 0
    while len(all_leafs) > 4:  # let's set a limit of 4 groups
        console.log(f"Generating tree from [cyan]{args.init_tree}[/cyan] in level {i}...")
        tree_generator = TreeGenerator(all_leafs) if i > 0 else TreeGenerator(all_leafs, init_tree)
        tree_generator.generate_tree_one_layer(all_leafs, True, console)

        N.io.write.json(f"{temp_dir}/tree_{i}.json", tree_generator.current_tree, fast=False)

        all_leafs = list(tree_generator.current_tree.keys())
        i += 1

    final_tree = merge_tree(i - 1)
    N.io.write.json(output_file, final_tree, fast=False)
    return final_tree


def generate_top_groups_items(all_leafs: list[str], top_groups: list[str]):
    # when preparing the top down tree, we just group the leafs to the top groups then build the tree based on the top groups
    tree_generator = TreeGenerator(all_leafs, {g: [] for g in top_groups})
    console.log(f"Generating top groups items [cyan]{top_groups}[/cyan]...")
    tree_generator.generate_tree_one_layer(all_leafs, False, console)
    N.io.write.json(f"{temp_dir}/tree_top_groups_{args.top_groups}.json", tree_generator.current_tree, fast=False)
    return tree_generator.current_tree

def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()
    return lines

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tree")
    parser.add_argument("--leafs", type=str, required=True, help="Leafs")
    parser.add_argument("--init_tree", type=str, default="scratch", help="Initial tree")
    parser.add_argument("--top_groups", type=str, default="none", help="Top groups")
    args = parser.parse_args()

    match args.leafs:
        case "scannet":
            labels_df = pd.read_table("semantics/scannetv2-labels.combined.tsv")
            all_leafs = labels_df.category.tolist()
        case "replica":
            labels_json = N.io.read.json("semantics/replica_info_semantic.json")
            all_leafs = [each["name"] for each in labels_json["classes"]]
        case "scannetpp100":
            all_leafs = read_txt_list("semantics/top100.txt")
        case "scannetpp":
            all_leafs = read_txt_list("semantics/semantic_classes.txt")
        case _:
            raise ValueError("Invalid argument --leafs")
        
    match (args.leafs, args.init_tree):
        case ("scannet", "nyu40"):
            init_tree = {}
            for i, row in labels_df.iterrows():
                if row["nyu40class"] not in init_tree:
                    init_tree[row["nyu40class"]] = []
                init_tree[row["nyu40class"]].append(row["category"])

            # remove "others"
            keys_others = [k for k in init_tree.keys() if k.startswith("other")]
            for k in keys_others:
                del init_tree[k]
        case (_, "scratch"):
            init_tree = None
        case _:
            raise ValueError("Invalid argument --init_tree")
        
    top_groups_size = ["small", "medium", "large"]
    top_groups_shape = ["multi-plane", "single-plane", "other"]
    match args.top_groups:
        case "none":
            top_groups = None
        case "size":
            top_groups = top_groups_size
        case "shape":
            top_groups = top_groups_shape
        case "size_shape":
            top_groups = [f"{s}_{sh}" for s in top_groups_size for sh in top_groups_shape]
        case _:
            raise ValueError("Invalid argument --top_groups")
            
    if top_groups is None:
        with console.status("[bold green]Generating tree..."):
            generate_tree_bottom_up(all_leafs, init_tree, output_file=f"final_tree_{args.leafs}_from_{args.init_tree}.json")
    else:
        with console.status(f"[bold green]Generating tree for top groups...") as status:
            groups = generate_top_groups_items(all_leafs, top_groups)
            for group, items in groups.items():
                status.update(f"[bold green]Generating tree for [cyan]{group}[/cyan]...")
                generate_tree_bottom_up(items, output_file=f"{temp_dir}/tree_{args.leafs}_from_{args.init_tree}_{group}.json")
            # then merge them
            final_tree = {group: N.io.read.json(f"{temp_dir}/tree_{args.leafs}_from_{args.init_tree}_{group}.json") for group in top_groups}
            N.io.write.json(f"final_tree_{args.leafs}_from_{args.init_tree}_{args.top_groups}.json", final_tree, fast=False)