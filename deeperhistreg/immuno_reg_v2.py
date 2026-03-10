"""register immuno with dapi from spatial. Regsiter panel 1 to dapi, then register other panels to panel 1."""
from pathlib import Path
import argparse
import os
os.environ['VIPS_CONCURRENCY'] = '4'
from deeperhistreg.himico_reg import run_on_one


immuno_mapping = {
    "IMU002": "6CDH",
    "IMU004": "1YQX",
    "IMU007": "AKE3",
    "IMU009": "ZXTA",
    "IMU010": "GRPM",
    "IMU011": "UV5I",
    "IMU012": "KLDM",
    "IMU013": "XM3D",
    "IMU015": "3PE5",
    "IMU016": "JM52",
    #"IMU018": "V47F",
    "IMU020": "9E3P",
    "IMU021": "G7R8",
    "IMU025": "BXEU",
    "IMU027": "AQ1H",
    "IMU028": "P4DW",
    "IMU029": "1Y2I",
    "IMU032": "Z3EH",
}

def get_source(target_path, immuno_mapping, panel, source_base):
    """get target path from source path"""
    id = target_path.stem.split("_")[0]
    if id != "IMU028":
        return None
    if id in immuno_mapping:
        source_id = immuno_mapping[id] + f".0{panel}"
        source_base = Path(source_base) / f"Panel{panel}"
        source_path = list(source_base.glob(f"{source_id}*.qptiff"))[0]
        return source_path
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeeperHistReg arguments")
    parser.add_argument('--source', type=str,
                        help="Path to the source images") # immuno panels
    parser.add_argument('--target', type=str,
                        help="Path to the target images") # H&E
    parser.add_argument('--output_folder', dest='output_folder', type=str, help="Path to the output folder")
    args = parser.parse_args()
    target_folder = Path(args.target)
    target_list = list(target_folder.glob("*pyramidal.ome.tif"))
    output_folder = Path(args.output_folder)
    #sort
    target_list.sort()
    print(f"Found {len(target_list)} images")
    print(f"images are: {target_list}")

    for i in range(len(target_list)):
        # register panel 1 to dapi
        source = get_source(target_list[i], immuno_mapping, 1, Path(args.source))
        if source is None:
            print(f"Source not found for {target_list[i]} in panel {1}")
            continue
        target = target_list[i].parent / target_list[i].name[0:6] / "morphology_focus" / "morphology_focus_0000.ome.tif" # dapi
        print(f"Registering {source} to {target}")
        p1_path = output_folder / (source.stem + "_reg2dapi.tiff")
        if not p1_path.exists():
            run_on_one(source, target, p1_path)
        p1_path = target # register all to dapi
        
        for panel in range(2, 4):
            source = get_source(target_list[i], immuno_mapping, panel, Path(args.source))
            if source is None:
                print(f"Source not found for {target_list[i]} in panel {panel}")
                continue
            target = p1_path # registered panel 1
            out_path = output_folder / (source.stem + "_reg2dapi.tiff")
            if not out_path.exists():
                print(f"Registering {source} to {target}")
                run_on_one(source, target, out_path)
        print(f"Processed {target_list[i]}")