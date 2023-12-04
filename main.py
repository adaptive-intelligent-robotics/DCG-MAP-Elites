from typing import Tuple
from dataclasses import dataclass

from utils import Config

import hydra
from hydra.core.config_store import ConfigStore


@hydra.main(version_base="1.2", config_path="configs/", config_name="config")
def main(config: Config) -> None:
    if config.algo.name == "me":
        import main_me as main
    elif config.algo.name == "me_es":
        import main_me_es as main
    elif config.algo.name == "pga_me":
        import main_pga_me as main
    elif config.algo.name == "qd_pg":
        import main_qd_pg as main
    elif config.algo.name == "dcg_me":
        import main_dcg_me as main
    elif config.algo.name == "dcg_me_gecco":
        import main_dcg_me_gecco as main
    elif config.algo.name == "ablation_ai":
        import main_ablation_ai as main
    elif config.algo.name == "ablation_actor":
        import main_ablation_actor as main
    else:
        raise NotImplementedError

    main.main(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
