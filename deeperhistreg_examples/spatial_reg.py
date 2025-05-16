"""register immuno with H&E"""
from pathlib import Path
import argparse
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
    "IMU018": "V47F",
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
    if id != "IMU029":
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
                        help="Path to the spatial images")
    parser.add_argument('--output_folder', dest='output_folder', type=str, help="Path to the output folder")
    args = parser.parse_args()
    source_folder = Path(args.source)
    source_list = list(source_folder.glob("*pyramidal.ome.tif"))
    output_folder = Path(args.output_folder)
    #sort
    source_list.sort()
    print(f"Found {len(source_list)} images")
    print(f"images are: {source_list}")

    for i, source in enumerate(source_list):
        if source.name[0:6] != "IMU029":
            continue
        target = source.parent / source.name[0:6] / "morphology_focus" / "morphology_focus_0000.ome.tif" # dapi
        if target is None:
            print(f"Target not found for {source} at {target}")
            break
        print(f"Registering {source} to {target}")
        run_on_one(source, target, output_folder / (source.name[0:6] + "_HE_reg.tiff"))
        print(f"Processed {target}")