# LLM-based Hier-SLAM Tree Generator

This script generates hierarchical tree structures from a flat list of semantic class labels using bottom-up or top-down approaches with LLM (ChatGPT) integrated. Outputs hierarchical trees as JSON files for use in scene understanding.

Supports datasets like ScanNet, Replica, NYU40, ScanNet++.

## Setup
Insert your `OPENAI_API_KEY` in `.env`.

  ```bash
  pip install -r requirements.txt
  ```

## Usage Instructions

Run the script with the following command-line arguments to customize the tree generation process:

### General Bash Command
```bash
python gen_tree.py --leafs [scannet|replica|nyu40] --init_tree [scratch|nyu40] --top_groups [none|size|shape|size_shape]
```

### `--leafs` Options 
Specifies the source of leaf nodes (semantic class labels):
- **`scannet`**: Uses all ScanNet categories from `semantics/scannetv2-labels.combined.tsv`.
- **`replica`**: Uses Replica semantic classes from `semantics/replica_info_semantic.json`.

### `--init_tree` Options 
Defines the initial tree structure to start from:
- **`scratch`**: Starts with no initial grouping (default for all `--leafs`).
- **`nyu40`**: Uses NYU40 classes as initial groups (only valid with `--leafs scannet`; removes "other*" categories).

### `--top_groups` Options
Sets the top-down grouping strategy:
- **`none`**: Builds the tree bottom-up without predefined top groups.
- **`size`**: Groups into `small`, `medium`, `large`.
- **`shape`**: Groups into `multi-plane`, `single-plane`, `other`.
- **`size_shape`**: Combines size and shape (e.g., `small_multi-plane`, `medium_single-plane`, etc., 9 total groups).