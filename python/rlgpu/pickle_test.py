import pickle
import os
import sys
import torch

def main():
    prefix = "/home/jcoholich/isaacgym/python/rlgpu/data/H_ss_state_02_rand_newdd_200_bb_0_225"
    fname = "data_--checkpoint__220224123057801520__--timeout__5000__--plot_values__--des_dir_coef__200__--des_dir__0__--box_len__0.225__--footstep_targets_in_place__--add_ss__--ss_infill__1.0__--ss_height_var__0.1.pkl"

    path = os.path.join(prefix, fname)

    prev_size = os.path.getsize(path)
    print(f"Size of file on disk {prev_size / 10**6 :.2f} Mb")

    with open(path, 'rb') as f:
        data = pickle.load(f)

    full_size = get_size(data)
    print(f"Size of unpickled file {full_size / 10**6 :.2f} Mb")

    new_pkl_location = "/home/jcoholich/isaacgym/python/rlgpu"
    test_fname = "pickle_test_file.pkl"
    new_path = os.path.join(new_pkl_location, test_fname)
    with open(new_path, 'wb') as f:
        pickle.dump(data, f, protocol=5)

    new_size = os.path.getsize(new_path)
    print(f"New size of file on disk {new_size / 10**6 :.2f} Mb")



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
    main()
