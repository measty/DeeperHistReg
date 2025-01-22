"""
This example shows how to run the nonrigid registration using the library installed from PyPi (or manually from the repo). 
"""
import pathlib
from typing import Union
from pathlib import Path
import argparse
import os

import deeperhistreg

def run_on_one(source_path: Union[str, pathlib.Path], target_path: Union[str, pathlib.Path], output_path: Union[str, pathlib.Path]):
    ### Define Inputs/Outputs ###
    #source_path : Union[str, pathlib.Path] = Path(r"/media/u2071810/Extra Data/HIMICO/slides/B-1989502_B11_HE.mrxs")
    #target_path : Union[str, pathlib.Path] = Path(r"/media/u2071810/Extra Data/HIMICO/slides/B-1989502_B11_HE_CDX2p_MUC2y_MUC5g_CD8dab.mrxs")
    #output_path : Union[str, pathlib.Path] = Path(r"/media/u2071810/Extra Data/HIMICO/Janssen/reg")

    ### Define Params ###
    registration_params : dict = deeperhistreg.configs.default_nonrigid_high_resolution() # Alternative: # registration_params = deeperhistreg.configs.load_parameters(config_path) # To load config from JSON file
    save_displacement_field : bool = True # Whether to save the displacement field (e.g. for further landmarks/segmentation warping)
    copy_target : bool = False # Whether to copy the target (e.g. to simplify the further analysis
    delete_temporary_results : bool = False # Whether to keep the temporary results
    case_name : str = "Example_Nonrigid" # Used only if the temporary_path is important, otherwise - provide whatever
    output_folder = Path(output_path).parent
    temporary_path : Union[str, pathlib.Path] = output_folder / "_temp2" # Will use default if set to None

    # modify defaults
    registration_params["loading_params"]['loader'] = 'openslide'
    registration_params["loading_params"]['source_resample_ratio'] = 0.05
    registration_params["loading_params"]['target_resample_ratio'] = 0.05
    registration_params["loading_params"]["final_level"] = 0

    ### Create Config ###
    config = dict()
    config['source_path'] = str(source_path)
    config['target_path'] = str(target_path)
    config['output_path'] = output_folder
    config['registration_parameters'] = registration_params
    config['case_name'] = case_name
    config['save_displacement_field'] = save_displacement_field
    config['copy_target'] = copy_target
    config['delete_temporary_results'] = delete_temporary_results
    config['temporary_path'] = temporary_path
    
    ### Run Registration ###
    deeperhistreg.run_registration(**config)
    # rename output_folder/warped_source.tiff to output_path
    os.rename(output_folder / "warped_source.tiff", output_path)
    os.rename(output_folder / "displacement_field.mha", output_folder / (Path(output_path).stem + "_displacement_field.mha"))
    print(f"Saved registered image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeeperHistReg arguments")
    parser.add_argument('--source', type=str,
                        help="Path to the source images")
    parser.add_argument('--target', type=str,
                        help="Path to the target images")
    parser.add_argument('--output_folder', dest='output_folder', type=str, help="Path to the output folder")
    args = parser.parse_args()
    target_list = list(Path(args.target).parent.glob(Path(args.target).name))
    source_folder = Path(args.source).parent
    #source_list = list(Path(args.source).parent.glob(Path(args.source).name))
    source_list = [source_folder / p.stem(p.stem[:-2] + "CDX2p_MUC2y_MUC5g_CD8dab.mrxs") for p in target_list]
    #source_list = args.source
    #target_list = args.target
    for source_path, target_path in zip(source_list, target_list):
        print(f"Processing {source_path}")
        output_path = Path(args.output_folder) / (Path(source_path).stem + "_reg.tiff")
        if output_path.exists():
            print(f"Output {output_path} already exists. Skipping.")
            continue
        try:
            run_on_one(Path(source_path), Path(target_path), output_path)
        except Exception as e:
            print(f"Failed to process {source_path}: {e}")
            continue