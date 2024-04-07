import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import sys
sys.path.insert(0,'/home/cuctt/credit/src')
from data.australian import AustralianDataset
import rootutils
from data.base_data import CreditDataModule
from models.base import BaseModel
from grid_search.grid_search import BaseGrid
import warnings
warnings.filterwarnings("ignore")

path = rootutils.find_root(search_from=__file__, indicator=".project-root")

rootutils.set_root(
    path=path, 
    project_root_env_var=True, 
    dotenv=True, 
    pythonpath=True, 
    cwd=True,
)

@hydra.main(version_base=None,config_path="/home/cuctt/credit/config/", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # loda dataset from cfg.data
    logging.info(f"Loading dataset... {cfg.data._target_}")
    data: CreditDataModule = hydra.utils.instantiate(cfg.data)

    # load model from cfg.model
    # logging.info(f"Loading model... {cfg.model._target_}")
    # model: BaseModel = hydra.utils.instantiate(cfg.model)


    logging.info(f"Grid Search model... {cfg.grid_search._target_}")

    logging.info("Grid search...")
    grid_search: BaseGrid = hydra.utils.instantiate(cfg.grid_search)
        
    logging.info("Fitting model...")
    grid_search.grid(data)

    logging.info("Validating model...")
    grid_search.validate(data)

if __name__=="__main__":
    main()