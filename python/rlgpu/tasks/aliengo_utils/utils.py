import torch


@torch.jit.script
def x_traj(phases):
    """Take a tensor of phases and return tensor of same shape of
    x foot positions. Frequency is 1. Amplitude goes from -1 to 1.
    """
    output = torch.zeros_like(phases)
    assert (phases >= 0.0).all()
    norm_phases = (phases % 1.0) * 2
    first_idcs = norm_phases < 1.0
    output[first_idcs] = 2 * (-2*norm_phases[first_idcs].pow(3) + 3*norm_phases[first_idcs].square()) - 1.0
    second_idcs = norm_phases >= 1.0
    output[second_idcs] = 2 * (2*norm_phases[second_idcs].pow(3) - 9*norm_phases[second_idcs].square() + 12*norm_phases[second_idcs] - 4) - 1.0
    return output


@torch.jit.script
def z_traj(phases):
    """Take a tensor of phases and return tensor of same shape of
    z foot positions. Frequency is 1. Amplitude goes from 0 to 1.
    """
    output = torch.zeros_like(phases)
    assert (phases >= 0.0).all()
    norm_phases = (phases % 1.0) * 4 - 2.0
    first_idcs = torch.logical_and(norm_phases >= 0.0, norm_phases < 1.0)
    output[first_idcs] = -2*norm_phases[first_idcs].pow(3) + 3*norm_phases[first_idcs].square()
    second_idcs = torch.logical_and(norm_phases >= 1.0, norm_phases <= 2.0)
    output[second_idcs] = 2*norm_phases[second_idcs].pow(3) - 9*norm_phases[second_idcs].square() + 12*norm_phases[second_idcs] - 4.0
    return output

@torch.jit.script
def x_traj_jump(phases):
    """Take a tensor of phases and return tensor of same shape of
    x foot positions. Frequency is 1. Amplitude goes from -1 to 1.
    """
    output = torch.zeros_like(phases)
    assert (phases >= 0.0).all()
    norm_phases = (phases % 1.0) * 2
    first_idcs = norm_phases < 1.0
    output[first_idcs] = 2 * (-2*norm_phases[first_idcs].pow(3) + 3*norm_phases[first_idcs].square()) - 1.0
    second_idcs = torch.logical_and(norm_phases >= 1.0, norm_phases < 1.75)
    output[second_idcs] = 2 * (2*norm_phases[second_idcs].pow(3) - 9*norm_phases[second_idcs].square() + 12*norm_phases[second_idcs] - 4) - 1.0
    jump_idcs = norm_phases >= 1.75
    output[jump_idcs] = -1.0
    return output


@torch.jit.script
def z_traj_jump(phases):
    """Take a tensor of phases and return tensor of same shape of
    z foot positions. Frequency is 1. Amplitude goes from -0.15 to 1.
    """
    output = torch.zeros_like(phases)
    assert (phases >= 0.0).all()
    norm_phases = (phases % 1.0) * 4 - 2.0
    jump_idcs = torch.logical_and(norm_phases >= -0.5, norm_phases < 0.0)
    output[jump_idcs] = -0.5
    first_idcs = torch.logical_and(norm_phases >= 0.0, norm_phases < 1.0)
    output[first_idcs] = -2*norm_phases[first_idcs].pow(3) + 3*norm_phases[first_idcs].square()
    second_idcs = torch.logical_and(norm_phases >= 1.0, norm_phases <= 2.0)
    output[second_idcs] = 2*norm_phases[second_idcs].pow(3) - 9*norm_phases[second_idcs].square() + 12*norm_phases[second_idcs] - 4.0
    return output


def batch_z_rot_mat(theta):
    """Takes a 1D tensor of z-rotations and returns a 3x3 rotation matrix."""
    rot_mat = torch.zeros(*theta.shape, 3, 3, device=theta.device)
    rot_mat[..., 0, 0] = torch.cos(theta)
    rot_mat[..., 0, 1] = -torch.sin(theta)
    rot_mat[..., 1, 0] = torch.sin(theta)
    rot_mat[..., 1, 1] = torch.cos(theta)
    rot_mat[..., 2, 2] = torch.ones_like(rot_mat[..., 2, 2])
    return rot_mat

def batch_z_2D_rot_mat(theta):
    """Takes a 1D tensor of z-rotations and returns a 2x2 rotation matrix."""
    rot_mat = torch.zeros(*theta.shape, 2, 2, device=theta.device)
    rot_mat[..., 0, 0] = torch.cos(theta)
    rot_mat[..., 0, 1] = -torch.sin(theta)
    rot_mat[..., 1, 0] = torch.sin(theta)
    rot_mat[..., 1, 1] = torch.cos(theta)
    return rot_mat


def batch_y_rot_mat(theta):
    """Takes a 1D tensor of z-rotations and returns a 3x3 rotation matrix."""
    rot_mat = torch.zeros(*theta.shape, 3, 3, device=theta.device)
    rot_mat[..., 0, 0] = torch.cos(theta)
    rot_mat[..., 0, 2] = torch.sin(theta)
    rot_mat[..., 1, 1] = 1.0
    rot_mat[..., 2, 0] = -torch.sin(theta)
    rot_mat[..., 2, 2] = torch.cos(theta)
    return rot_mat


def batch_x_rot_mat(theta):
    """Takes a 1D tensor of z-rotations and returns a 3x3 rotation matrix."""
    rot_mat = torch.zeros(*theta.shape, 3, 3, device=theta.device)
    rot_mat[..., 0, 0] = 1.0
    rot_mat[..., 1, 1] = torch.cos(theta)
    rot_mat[..., 1, 2] = -torch.sin(theta)
    rot_mat[..., 2, 1] = torch.sin(theta)
    rot_mat[..., 2, 2] = torch.cos(theta)
    return rot_mat


# @torch.jit.script
def batch_quat_to_euler(q):
    """https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/src/Bullet3Common/b3Quaternion.h
    """
    # TODO make sure I'm wrapping angles if I need to. However, if quats from root tensor are all unit quats, is there no need to?
    e = torch.zeros(q.shape[0], 3, device=q.device)
    e[:, 0] = torch.atan2(2 * (q[:, 3] * q[:, 0] + q[:, 1] * q[:, 2]),
                          q[:, 3].square() - q[:, 0].square() - q[:, 1].square() + q[:, 2].square())
    # sarg = -2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])
    e[:, 1] = torch.asin((2 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0])).clamp(-1.0, 1.0))
    e[:, 2] = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                          q[:, 3].square() + q[:, 0].square() - q[:, 1].square() - q[:, 2].square())
    if e.isnan().any():
        print('There are nans')
        breakpoint()
    return e


@torch.jit.script
def batch_quat_to_6d(q):
    """https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Rotation_matrices
    and
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d
    """
    o = torch.zeros(q.shape[0], 6, device=q.device)
    x, y, z, w = 0, 1, 2, 3
    o[:, 0] = q[:, w].square() + q[:, x].square() - q[:, y].square() - q[:, z].square()
    o[:, 1] = 2 * (q[:, x] * q[:, y] - q[:, w] * q[:, z])
    o[:, 2] = 2 * (q[:, w] * q[:, y] + q[:, x] * q[:, z])
    o[:, 3] = 2 * (q[:, x] * q[:, y] + q[:, w] * q[:, z])
    o[:, 4] = q[:, w].square() - q[:, x].square() + q[:, y].square() - q[:, z].square()
    o[:, 5] = 2 * (q[:, y] * q[:, z] - q[:, w] * q[:, x])
    return o
