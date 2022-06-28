from distutils.log import error
from re import L
import subprocess
import time
from datetime import datetime
import sys


def main():
    s = time.time()
    node_list = get_node_list()
    print("list of all nodes", node_list)
    blacklist = get_down_nodes()
    print("down nodes: ", blacklist)
    checks = [docker_running, docker_nvidia_issue]  # list of function handles
    for node in node_list:
        if node in blacklist:
            continue
        print(f"checking node {node}")
        if any(not check(node) for check in checks):
            blacklist.append(node)
    blacklist.sort()
    print("blacklist obtained", blacklist)
    write_blist(blacklist)
    print(f"total time: {(time.time()-s)/60.0 :.2f} min")


def get_down_nodes():
    cmd = ["ssh", "jcoholich3@sky1.cc.gatech.edu", "sinfo -d -p short -o='%N'"]
    output, err = run_cmd(cmd)
    if err:
        print(err)
        sys.exit()
    node_list = output.split()[1][1:].split(",")
    return node_list


def docker_running(node):
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "jcoholich3@" + node + ".cc.gatech.edu",
        "docker container ls",
    ]
    output, stderr = run_cmd(cmd)
    if output == "":
        return False
    else:
        return True


def docker_nvidia_issue(node):
    """Here is the solution to this issue: https://github.com/NVIDIA/nvidia-docker/issues/1243
    I just can't fix things without root"""
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "jcoholich3@" + node + ".cc.gatech.edu",
        "docker run --gpus all  nvidia/cuda:10.0-base nvidia-smi",
    ]
    out, err = run_cmd(cmd)
    if out == "":
        return False
    else:
        return True

def write_blist(blacklist):
    with open("blacklisted_nodes.txt", "w") as f:
        f.write("generated at " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n")
        for node in blacklist:
            f.write(node + "\n")


def get_node_list():
    output, err = run_cmd(["ssh", "jcoholich3@sky1.cc.gatech.edu", "sinfo -o='%n'"])
    if err:
        print(err)
        sys.exit()
    node_list = [s[1:] for s in output.split()[1:]]
    return node_list


def run_cmd(cmd):
    output = subprocess.run(cmd, capture_output=True, encoding="UTF-8")
    return output.stdout, output.stderr


if __name__ == "__main__":
    main()
