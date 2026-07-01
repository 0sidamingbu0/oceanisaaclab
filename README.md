# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Python 运行约定（重要）

本工程的所有 Python 脚本都必须用 Isaac Lab 自带的解释器运行，而**不是系统 `python`**：

```bash
./_isaaclab/isaaclab.sh -p <脚本> [参数]
```

系统 `python` 缺少 `tensorboard`、`torch`、`isaaclab` 等依赖，直接跑会 `ModuleNotFoundError`。
例如离线解析训练曲线：

```bash
./_isaaclab/isaaclab.sh -p scripts/analyze_tb.py logs/rsl_rl/<task>/<run>
```

## Changelog / 修改记录

每天的修改记录统一放在 [`changelog/`](changelog/) 文件夹中，按日期命名（`YYYY-MM-DD.md`，如 [`changelog/2026-06-25.md`](changelog/2026-06-25.md)）。

目的：记录每天改了什么、为什么改、改动效果如何，便于理解工程的历史上下文，也方便其他工程师快速了解本工程的演进履历。

每个日志建议包含：改动背景与动机、涉及文件、具体改动、验证方式、训练/实验观察、待办与建议。

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    _isaaclab/isaaclab.sh -p -m pip install -e source/oceanisaaclab

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        _isaaclab/isaaclab.sh -p scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        # python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        _isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Template-Oceanisaaclab-Direct-v0 --viz kit
        ```
    - Tensorboard:
        ```bash
        ./_isaaclab/isaaclab.sh -p -m tensorboard.main --logdir=logs
        http://localhost:6006/
        ```
    - play:
        ```bash
        ./_isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py --task Ocean-BDX-Stand-Direct-v0 --num_envs 16 --viz kit --checkpoint /home/ocean/oceanisaaclab/oceanisaaclab/logs/rsl_rl/bdx_stand_direct/2026-06-24_10-22-28/model_6700.pt
        ```
    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            _isaaclab/isaaclab.sh -p scripts/zero_agent.py --task=Template-Oceanisaaclab-Direct-v0
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            _isaaclab/isaaclab.sh -p scripts/random_agent.py --task=Template-Oceanisaaclab-Direct-v0
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/oceanisaaclab/oceanisaaclab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/oceanisaaclab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```