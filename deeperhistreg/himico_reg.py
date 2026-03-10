"""
This example shows how to run the nonrigid registration using the library installed from PyPi (or manually from the repo). 
"""
import pathlib
from typing import Union
from pathlib import Path
import argparse
import os
os.environ['VIPS_CONCURRENCY'] = '4'

import deeperhistreg

post_he_to_4plex_mapping = {
    "IMU003_HE_reg.tiff": "B-2081919-01-10_CDX2pMUC5g2yCD8d_20240816",
    "IMU004_HE_reg.tiff": "B-2085765-01-08_CDX2pMUC5g2yCD8d_20240816",
    "IMU005_HE_reg.tiff": "B-2086168-01-10_CDX2pMUC5g2yCD8d_20240816",
    "IMU007_HE_reg.tiff": "B-2090081-01-08_CDX2pMUC5g2yCD8d_20240816",
    "IMU009_HE_reg.tiff": None,
    "IMU011_HE_reg.tiff": None,
    "IMU012_HE_reg.tiff": "B-2103655-01-05_CDX2pMUC5g2yCD8d_20240816",
    "IMU013_HE_reg.tiff": None,
    "IMU015_HE_reg.tiff": None,
    "IMU018_HE_reg.tiff": "B-2129062-01-12_CDX2pMUC5g2yCD8d_20240816",
    "IMU020_HE_reg.tiff": "B-2129674-01-17_CDX2pMUC5g2yCD8d_20240816",
    "IMU021_HE_reg.tiff": "B-2131171-01-15_CDX2pMUC5g2yCD8d_20240816",
    "IMU025_HE_reg.tiff": "B-2139804-01-13_CDX2pMUC5g2yCD8d_20240816",
    "IMU027_HE_reg.tiff": None,
    "IMU028_HE_reg.tiff": "B-2145157-01-06_CDX2pMUC5g2yCD8d_20240816",
    "IMU029_HE_reg.tiff": "B-2148280-01-05_CDX2pMUC5g2yCD8d_20240816"
}

inv_mapping = {v: k[0:6] for k, v in post_he_to_4plex_mapping.items() if v is not None}

def get_source(source_folder, p):
    if p.name != "IMU029_HE_reg.tiff":
        return None # temporarily just do one
    if p.name not in post_he_to_4plex_mapping:
        return None
    mapped_name = post_he_to_4plex_mapping[p.name]
    if mapped_name is None:
        return None
    return Path(source_folder) / (mapped_name + ".mrxs")

def get_target(target_folder, p):
    if p.stem not in inv_mapping:
        return None
    mapped_name = inv_mapping[p.stem]
    if mapped_name is None:
        return None
    return Path(target_folder) / mapped_name / "morphology_focus/morphology_focus_0000.ome.tif"

def run_on_one(source_path: Union[str, pathlib.Path], target_path: Union[str, pathlib.Path], output_path: Union[str, pathlib.Path]):
    ### Define Inputs/Outputs ###
    #source_path : Union[str, pathlib.Path] = Path(r"/media/u2071810/Extra Data/HIMICO/slides/B-1989502_B11_HE.mrxs")
    #target_path : Union[str, pathlib.Path] = Path(r"/media/u2071810/Extra Data/HIMICO/slides/B-1989502_B11_HE_CDX2p_MUC2y_MUC5g_CD8dab.mrxs")
    #output_path : Union[str, pathlib.Path] = Path(r"/media/u2071810/Extra Data/HIMICO/Janssen/reg")

    ### Define Params ###
    registration_params : dict = deeperhistreg.configs.default_initial_nonrigid_high_resolution() # Alternative: # registration_params = deeperhistreg.configs.load_parameters(config_path) # To load config from JSON file
    save_displacement_field : bool = True # Whether to save the displacement field (e.g. for further landmarks/segmentation warping)
    copy_target : bool = False # Whether to copy the target (e.g. to simplify the further analysis
    delete_temporary_results : bool = False # Whether to keep the temporary results
    case_name : str = "Example_Nonrigid" # Used only if the temporary_path is important, otherwise - provide whatever
    output_folder = Path(output_path).parent
    temporary_path : Union[str, pathlib.Path] = output_folder / "_temp2" # Will use default if set to None
    save_final_images : bool = True # Whether to save the final images (e.g. for further analysis)
    save_quality_maps : bool = False # Whether to save the quality maps (e.g. for further analysis)

    # modify defaults
    registration_params["loading_params"]['loader'] = 'tiatoolbox' # 'tiatoolbox' # 'openslide'
    registration_params["loading_params"]['source_resample_ratio'] = 0.14
    registration_params["loading_params"]['target_resample_ratio'] = 0.14
    registration_params["loading_params"]["final_level"] = 0
    registration_params["initial_registration_params"]["initial_registration_function"] = "multi_feature" # "instance_optimization_affine_registration" # multi_feature
    registration_params["initial_registration_params"]["transform_type"] = "affine"
    registration_params['save_final_displacement_field'] = save_displacement_field
    registration_params["save_final_images"] = save_final_images
    registration_params["save_quality_maps"] = save_quality_maps
    #registration_params["nonrigid_registration_params"]["registration_size"] = 10240
    registration_params["nonrigid_registration_params"]["cost_function_params"]["return_map"] = True

    # additional reg parameters for affine initial reg
    affine_init_reg_params = {
        "cost_function": "mind_loss",
        "cost_function_params": {
            "win_size": 7
        },
        "regularization_function": "diffusion_relative",
        "regularization_function_params": {},
        "registration_size": 2048,
        "num_levels": 7,
        "used_levels": 7,
        "iterations": [
            120,
            100,
            100,
            150,
            150,
            200,
            200
        ],
        "learning_rate": 0.01,
    }
    if registration_params["initial_registration_params"]["initial_registration_function"] == "instance_optimization_affine_registration":
        for key, value in affine_init_reg_params.items():
            registration_params["initial_registration_params"][key] = value

    ### Create Config ###
    config = dict()
    config['source_path'] = str(source_path)
    config['target_path'] = str(target_path)
    config['output_path'] = output_folder
    config['registration_parameters'] = registration_params
    config['case_name'] = case_name
    config['copy_target'] = copy_target
    config['delete_temporary_results'] = delete_temporary_results
    config['temporary_path'] = temporary_path
    config['save_displacement_field'] = save_displacement_field
    
    ### Run Registration ###
    deeperhistreg.run_registration(**config)
    # rename output_folder/warped_source.tiff to output_path
    if save_final_images:
        os.rename(output_folder / "warped_source.tiff", output_path)
    if save_displacement_field:
        os.rename(output_folder / "displacement_field.mha", output_folder / (Path(output_path).stem + "_displacement_field.mha"))
    if save_quality_maps:
        os.rename(temporary_path / case_name / "quality_map.png", output_folder / (Path(output_path).stem + "_quality_map.png"))
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
    #source_list = list(Path(args.source).parent.glob(Path(args.source).name))
    #target_list = [get_target(Path(args.target), p) for p in source_list]
    if len(target_list) == 1:
        # run using provided source and target
        source_path = Path(args.source)
        target_path = Path(args.target)
        output_path = Path(args.output_folder) / (Path(source_path).stem + "_reg.tiff")
        if output_path.exists():
            print(f"Output {output_path} already exists. Skipping.")
        else:
            print(f"Registering {source_path} to {target_path}")
            run_on_one(source_path, target_path, output_path)
    elif len(target_list) == 0:
        print(f"No target images found in {args.target}")
    else:
        # run on a bunch of slides
        source_suffix = Path(args.source).name[1:]
        target_suffix = Path(args.target).name[1:]
        source_folder = Path(args.source).parent
        #source_list = list(Path(args.source).parent.glob(Path(args.source).name))
        #source_list = [source_folder / p.name.replace(target_suffix, source_suffix) for p in target_list]
        source_list = [get_source(source_folder, p) for p in target_list]
        #source_list = args.source
        #target_list = args.target
        for source_path, target_path in zip(source_list, target_list):
            if source_path is None or target_path is None:
                continue
            print(f"Registering {source_path} to {target_path}")
            output_path = Path(args.output_folder) / (Path(source_path).stem + "_reg_test2.tiff")
            if output_path.exists():
                print(f"Output {output_path} already exists. Skipping.")
                continue
            #try:
            run_on_one(Path(source_path), Path(target_path), output_path)
            #except Exception as e:
            #    print(f"Failed to process {source_path}: {e}")
            #    continue

    print("Done")