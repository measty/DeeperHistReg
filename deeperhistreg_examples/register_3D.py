"""register consecutive slices of a 3D histology image stack"""
from pathlib import Path
import argparse
from deeperhistreg.himico_reg import run_on_one

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeeperHistReg arguments")
    parser.add_argument('--source', type=str,
                        help="Path to the source images")
    parser.add_argument('--output_folder', dest='output_folder', type=str, help="Path to the output folder")
    args = parser.parse_args()
    source_folder = Path(args.source)
    source_list = list(source_folder.glob("*/*/*.qptiff"))
    output_folder = Path(args.output_folder)
    #sort
    source_list.sort()
    print(f"Found {len(source_list)} images")
    print(f"images are: {source_list}")
    # in sequence, we will register each slice to the registered version of the previous one
    # apart from the second which we will register to the original first
    # first & second slice reg is special case
    run_on_one(source_list[1], source_list[0], output_folder / (source_list[1].stem + "_reg.tiff"))
    for i in range(2, len(source_list)):
        run_on_one(source_list[i], output_folder / (source_list[i-1].stem + "_reg.tiff"), output_folder / (source_list[i].stem + "_reg.tiff"))
        print(f"Processed {source_list[i]}")