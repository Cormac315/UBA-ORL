# Third-Party Code

The `patches/d3rlpy_*.py` files are based on d3rlpy v1.0.0:

- Repository: <https://github.com/takuseno/d3rlpy>
- Tag: `v1.0.0`
- Commit: `8eb11db2d6f406cfab6d08adc4e0c08666dd063e`

The patch retains the upstream implementations and adds the minimal two-stage
unlearning hooks required for TD3+BC, BCQ, and IQL. These hooks follow the
custom d3rlpy package distributed with TrajDeleter:

- Repository: <https://github.com/2019ChenGong/TrajDeleter>
- Commit: `9b8142116088889f6d9632463bedc6e4fcda2336`

The `torch_utility.py` patch also converts CUDA device strings to integer device
indices for compatibility with the PyTorch version used by the experiments.

Both upstream repositories provide the same MIT license text. A copy is included
at `LICENSES/d3rlpy-and-TrajDeleter-MIT.txt`.

The remainder of this repository does not receive a project-wide open-source
license through this third-party notice.
