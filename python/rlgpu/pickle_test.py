import pickle
import os
import sys
import torch
from generate_all_plots import CPU_Unpickler
import bz2
import gzip

"""Conclusion: gzip is the way to go because the file sizes are the same as bz2
but it is much faster.

https://stackoverflow.com/questions/18474791/decreasing-the-size-of-cpickle-objects
https://unix.stackexchange.com/questions/106275/du-gives-two-different-results-for-the-same-file
"""


def zip_all_existing_data():
    space_saved = 0
    dirs = os.listdir("data")
    for dir_ in os.listdir("data"):
        for file in os.listdir(os.path.join("data", dir_)):
            if file[-3:] == "pkl":
                print(f"processing file {file}")
                fpath = os.path.join("data", dir_, file)
                old_size = os.path.getsize(fpath)
                with open(fpath, 'rb') as f:
                    data = CPU_Unpickler(f).load()
                new_fpath = fpath[:-3] + "pgz"
                with gzip.GzipFile(new_fpath, 'w') as f:
                    pickle.dump(data, f, protocol=4)
                new_size = os.path.getsize(new_fpath)
                reduction = old_size - new_size
                space_saved += reduction
                os.remove(fpath)
                print(f"Space saved: {reduction / 10**6 :.2f} Mb")
    print(f"Total space saved: {space_saved / 10**6 :.2f} Mb")


def main():
    prefix = "data/H_curr_long_steps"
    fname = "data_--checkpoint__220617220458137766__--timeout__5000__--plot_values__--des_dir_coef__50__--des_dir__0__--footstep_targets_in_place__--add_ss__--ss_infill__1.0__--ss_height_var__0.1.pkl"

    path = os.path.join(prefix, fname)

    prev_size = os.path.getsize(path)
    print(f"Size of file on disk {prev_size / 10**6 :.2f} Mb")

    with open(path, 'rb') as f:
        # data = pickle.load(f)
        data = CPU_Unpickler(f).load()

    full_size = get_size(data)
    print(f"Size of unpickled file {full_size / 10**6 :.2f} Mb")

    new_pkl_location = "/home/jcoholich/isaacgym/python/rlgpu"
    for prot in [4]:
        test_fname = "pickle_test_file.pkl"
        test_fname = "pickle_test_file.pbz2"
        test_fname = "pickle_test_file.pgz"
        new_path = os.path.join(new_pkl_location, test_fname)
        # with open(new_path, 'wb') as f:
        # with bz2.BZ2File(new_path, 'w') as f:
        with gzip.GzipFile(new_path, 'w') as f:
            pickle.dump(data, f, protocol=prot)
        new_size = os.path.getsize(new_path)
        print(f"File size with protocol {prot}: {new_size / 10**6 :.2f} Mb")

    # make sure I can read the test file
    with gzip.GzipFile(new_path, 'r') as f:
        data = pickle.load(f)
    full_size = get_size(data)
    print(f"Size of uncompressed file {full_size / 10**6 :.2f} Mb")




def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif isinstance(obj, torch.Tensor):
        size += obj.element_size() * obj.nelement()
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


if __name__ == "__main__":
    # main()
    zip_all_existing_data()
