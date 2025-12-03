from pathlib import Path
import json
import itertools
from typing import Any


def change_core_params(fp: Path) -> None:
    base_cfg: dict[str, Any] = {
        "weight_width": 0,
        "lcn": 0,
        "input_width": 1,
        "spike_width": 1,
        "neuron_num": 1888,
        "pool_max": 0,
        "tick_wait_start": 1,
        "tick_wait_end": 1,
        "snn_en": 0,
        "target_lcn": 0,
        "test_chip_addr": 64,
        "n_repeat_nram": 1,
    }

    # Start from core (0,0) on chip (1,0)
    chip_list = [(1, 0), (0, 0), (1, 1), (0, 1)]
    count = 0
    result = {}

    for chip_x, chip_y in chip_list:
        chip_key = f"({chip_x},{chip_y})"
        result[chip_key] = {}

        for i in range(32):
            for j in range(32):
                if not (i >= 28 and j >= 28):
                    core_key = f"({i},{j})"
                    config = base_cfg.copy()
                    config["name"] = f"OfflineCorePlacement_{count}"
                    result[chip_key][core_key] = config
                    count += 1

    with (fp / "core_params_changed.json").open("w") as f:
        json.dump(result, f, indent=2)


def change_input_proj_info(fp: Path):
    pass


if __name__ == "__main__":
    build_dir = Path(__file__).parent
    change_core_params(build_dir)
