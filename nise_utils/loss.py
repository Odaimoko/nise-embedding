import torch


def tag_loss_batch(embedding, track_id, joints, sigma = 1):
    '''
        Compute pull and push loss in a batch
        For now regard treat samples indepentdently
    :param embedding: bs x len_embed x *heatmap_size
    :param track_id: bs
    :param gt_joints_pos: bs x num_joints x 3, within heatmap size
    :param sigma: I dont know what for, so set to 1
    :return:
    '''
    # TODO: what if within a batch a track_id appears more than once
    batch_size = embedding.size(0)
    gt_joints_pos = joints[:, :, :2]
    gt_joints_visible = joints[:, :, 2] > 0
    num_joints = joints.size(0)
    loss_pull = 0.
    loss_push = 0.
    ref_embs = []
    for person in range(batch_size):
        single_emb, t_id, gt_pos = embedding[person], track_id[person], gt_joints_pos[person]
        # corresponding gt poses' embedding
        gt_pos_emb = single_emb[:, gt_pos[:, 0], gt_pos[:, 1]]  # len_embed x num_joints
        # emb for a single person
        reference_embedding = torch.mean(gt_pos_emb, dim = 1)  # len_embed
        ref_embs.append(reference_embedding)  # used to calc push loss
        # calc loss... # len_embed  x num_joints,unsqueeze for broadcasting
        diff_ref_joint = gt_pos_emb - reference_embedding.unsqueeze(1)
        squared_diff = diff_ref_joint * diff_ref_joint
        # TODO: filter out invisible joints, or should we?
        single_tag_loss = torch.sum(squared_diff, dim = 0) * gt_joints_visible.float()
        loss_pull += torch.sum(single_tag_loss)
    loss_pull = loss_pull / num_joints / batch_size  # normalize
    
    ref_embs = torch.stack(ref_embs)  # bs x len_embed
    for i in range(batch_size):
        diff_between_people = ref_embs - ref_embs[i, :]  # len_embed  x num_joints
        squared_diff = diff_between_people * diff_between_people  # len_embed  x num_joints
        e = torch.exp(- torch.sum(squared_diff, dim = 1) / (2 * sigma ** 2))  # num_joints
        loss_push += torch.sum(e)  # scalar
    loss_push -= batch_size  # if regard the batch as diff people, the same people shouldnt be calced in push loss
    loss_push = loss_push / batch_size / batch_size
    return loss_pull, loss_push
