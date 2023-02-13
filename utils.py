import torch
from torch import Tensor


def hbox2qbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
    return torch.cat([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1)


def rbox2qbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
    vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.cat([pt1, pt2, pt3, pt4], dim=-1)
