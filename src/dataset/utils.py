import pdb
import torch


# Custom collate function
def collate_fn(batch):
    (
        feat_stpp,
        feat_dur,
        history_points,
        curr_point,
        curr_dur,
        boxes,
        flag_conditions,
    ) = zip(*batch)

    # Find the max length in the batch
    max_len = max(element.size(0) for element in feat_stpp)

    # Pad sequences
    feat_stpp_padded = torch.stack(
        [
            torch.cat([el1, torch.zeros(max_len - el1.size(0), el1.size(1))])
            for el1 in feat_stpp
        ]
    )

    # feat_dur_padded = torch.stack(
    #    [
    #        torch.cat([el2, torch.zeros(max_len - el2.size(0), el2.size(1))])
    #        for el2 in feat_dur
    #    ]
    # )
    history_points_padded = torch.stack(
        [
            torch.cat([el3, torch.zeros(max_len - el3.size(0), el3.size(1))])
            for el3 in history_points
        ]
    )

    feat_dur_padded = torch.stack(
        [
            torch.cat([el3, torch.zeros(max_len - el3.size(0), el3.size(1))])
            for el3 in feat_dur
        ]
    )

    # feat_dur = torch.stack(feat_dur).squeeze(-1)
    curr_points = torch.stack(curr_point)
    curr_durs = torch.stack(curr_dur)

    boxes = torch.stack(boxes)
    flag_conditions = torch.stack(flag_conditions)
    return (
        feat_stpp_padded,
        feat_dur_padded,
        history_points_padded,
        curr_points,
        curr_durs,
        boxes,
        flag_conditions,
    )


# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
