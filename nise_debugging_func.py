import torch


def gen_rand_flow(batch_size, h, w):
    return torch.rand(*[batch_size, 2, h, w])

def gen_rand_img(batch_size, h, w):
    return torch.rand(*[batch_size, 3, h, w])


def gen_rand_img(batch_size,num_joints, h, w):
    return torch.rand(*[batch_size, num_joints, h, w])

def gen_rand_bboxes(batch_size, num_people,h,w):
    return torch.rand(*[batch_size, num_joints, h, w]) #ng