import h5py
import argparse

def main():
    parser = argparse.ArgumentParser(description="Print HDF5 file structure")
    parser.add_argument("filename", help="Path to HDF5 file")
    args = parser.parse_args()

    with h5py.File(args.filename, "r") as f:
        def show_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(show_info)
        

if __name__ == "__main__":
    main()