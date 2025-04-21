from __future__ import annotations

import json

from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from rich.console import Console

load_dotenv()

from openai import OpenAI

openai_client = OpenAI()

# OPENAI_MODEL = "gpt-4-turbo"
OPENAI_MODEL = "gpt-4o-mini"


def inference(messages):
    return openai_client.chat.completions.create(model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=messages)


def get_price(response):
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    match OPENAI_MODEL:
        case "gpt-4-turbo":
            prompt_price = 10 / 1e6
            completion_price = 30 / 1e6
        case "gpt-4o-mini":
            prompt_price = 0.15 / 1e6
            completion_price = 0.6 / 1e6
        case _:
            raise ValueError(f"Unknown model: {OPENAI_MODEL}")
    return round(prompt_tokens * prompt_price + completion_tokens * completion_price, 4)


def get_json(response):
    out_json = json.loads(response.choices[0].message.content)
    # make sure the keys are lowercased
    return {k.lower(): v for k, v in out_json.items()}


def log_info(response, extras, missings, all_leafs, current_tree, console):
    console.log(f"Price: ${get_price(response)}")
    console.log("Extras:", len(extras), "\nMissings:", len(missings), "\nDone:", len(all_leafs) - len(missings))
    console.log(f"Current groups: {list(current_tree.keys())}")


def validate_labels(all_leafs, current_tree) -> tuple[list[str], list[str]]:
    output_labels = set([e for each in current_tree.values() for e in each])
    input_labels = set(all_leafs)

    extra_labels = output_labels - input_labels
    missings = input_labels - output_labels
    return list(extra_labels), list(missings)


class TreeGenerator:

    def __init__(self, all_leafs: list[str], init_tree: dict[str, list[str]] | None = None):
        self.all_leafs = all_leafs
        self.missings: list[str] = []
        self.extras: list[str] = []
        self.current_tree: dict[str, list[str]] = dict()

        # initialize the tree with the first prompting
        self.init_prompt = self.gen_init_prompt()
        if init_tree is None:
            self.response = inference([{"role": "user", "content": self.init_prompt}])
            self.current_tree = get_json(self.response)
        else:
            self.response = None
            self.current_tree = init_tree

    def gen_init_prompt(self) -> str:
        return """You're a smart bot who can accurately divide the items into groups.

Group the following items into groups.

""" + str(self.all_leafs) + """

Make sure the name of items keep the same, and the sizes of each group are similar. The output must be the same JSON format as below.

The group name should be meaningful, such as "furniture", "kitchenware", etc. But do not use "other" or other similar names as a group name.

{"<GROUP_1>": ["<ITEM_1>", "<ITEM_2>", ...], "<GROUP_2>": ["ITEM_3", "ITEM_4", ...], ...}"""

    def gen_missing_prompt(self, allow_new_groups: bool = True) -> str:
        if allow_new_groups:
            return """You missed the following items: """ + str(self.missings) + """. Please add them to the previous groups or create new groups.

The previous groups are: """ + str(list(self.current_tree.keys())) + """

Make sure the name of items keep the same, and the sizes of each group are similar. The output must be the same JSON format as below.

The group name should be meaningful, such as "furniture", "kitchenware", etc. But do not use "other" or other similar names as a group name. You should create new groups if necessary.

{"<GROUP_1>": ["<ITEM_1>", "<ITEM_2>", ...], "<GROUP_2>": ["ITEM_3", "ITEM_4", ...], ...}"""
        else:
            return """You missed the following items: """ + str(self.missings) + """. Please add them to the previous groups. Do not generate new groups.

The previous groups are: """ + str(list(self.current_tree.keys())) + """

Make sure the name of items keep the same, and the sizes of each group are similar. The output must be the same JSON format as below.

{"<GROUP_1>": ["<ITEM_1>", "<ITEM_2>", ...], "<GROUP_2>": ["ITEM_3", "ITEM_4", ...], ...}"""


    def validate_labels(self) -> tuple[list[str], list[str]]:
        self.extras, self.missings = validate_labels(self.all_leafs, self.current_tree)
        return self.extras, self.missings

    def grow_tree(self, allow_new_groups: bool = True):
        missing_prompt = self.gen_missing_prompt(allow_new_groups)
        self.response = inference([
            {"role": "user", "content": self.init_prompt},
            {"role": "assistant", "content": json.dumps(self.current_tree)},
            {"role": "user", "content": missing_prompt}
        ])
        output_new = get_json(self.response)

        # merge the new output with the current output
        for k, v in output_new.items():
            # if the group is not in the current tree and we don't allow new groups, skip it
            if k not in self.current_tree and not allow_new_groups:
                continue
            # else merge the new output with the current output
            self.current_tree[k] = self.current_tree.get(k, []) + v
            # remove duplicates
            self.current_tree[k] = list(set(self.current_tree[k]))

    def generate_tree_one_layer(self, all_leafs: list[str], allow_new_groups: bool, console: Console):
        while True:
            extras, missings = self.validate_labels()

            if self.response is not None:
                self.log_info(console)

            # remove the extra labels from current output
            for extra in extras:
                if extra not in all_leafs:
                    for k, v in self.current_tree.items():
                        if extra in v:
                            v.remove(extra)

            if len(missings) == 0:
                break

            # missing labels, ask LLM to add them to the current groups or create new groups
            self.grow_tree(allow_new_groups)

    def log_info(self, console: Console):
        return log_info(self.response, self.extras, self.missings, self.all_leafs, self.current_tree, console)
