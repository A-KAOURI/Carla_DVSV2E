from pathlib import Path
import torch
import numpy
import imageio.v2 as imageio
from tqdm import tqdm


def plot_points_on_background(points_coordinates,
                              background,
                              points_color=[0, 0, 255]):
    """
    Args:
        points_coordinates: array of (y, x) points coordinates
                            of size (number_of_points x 2).
        background: (3 x height x width)
                    gray or color image uint8.
        color: color of points [red, green, blue] uint8.
    """
    if not (len(background.size()) == 3 and background.size(0) == 3):
        raise ValueError('background should be (color x height x width).')
    _, height, width = background.size()
    background_with_points = background.clone()
    y, x = points_coordinates.transpose(0, 1)
    if len(x) > 0 and len(y) > 0: # There can be empty arrays!
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
            raise ValueError('points coordinates are outsize of "background" '
                             'boundaries.')
        background_with_points[:, y, x] = torch.Tensor(points_color).type_as(
            background).unsqueeze(-1)
    return background_with_points


def events_to_event_image(event_sequence, height, width, background=None, rotation_angle=None, crop_window=None,
                          horizontal_flip=False, flip_before_crop=True):
    polarity = event_sequence[:, 3] == -1.0
    x_negative = event_sequence[~polarity, 1].astype(int)
    y_negative = event_sequence[~polarity, 2].astype(int)
    x_positive = event_sequence[polarity, 1].astype(int)
    y_positive = event_sequence[polarity, 2].astype(int)

    positive_histogram, _, _ = numpy.histogram2d(
        x_positive,
        y_positive,
        bins=(width, height),
        range=[[0, width], [0, height]])
    negative_histogram, _, _ = numpy.histogram2d(
        x_negative,
        y_negative,
        bins=(width, height),
        range=[[0, width], [0, height]])

    # Red -> Negative Events
    red = numpy.transpose((negative_histogram >= positive_histogram) & (negative_histogram != 0))
    # Blue -> Positive Events
    blue = numpy.transpose(positive_histogram > negative_histogram)

    if background is None:
        height, width = red.shape
        background = torch.full((3, height, width), 255).byte()
    if len(background.shape) == 2:
        background = background.unsqueeze(0)
    else:
        if min(background.size()) == 1:
            background = grayscale_to_rgb(background)
        else:
            if not isinstance(background, torch.Tensor):
                background = torch.from_numpy(background)
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(red.astype(numpy.uint8))), background,
        [255, 0, 0])
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(blue.astype(numpy.uint8))),
        points_on_background, [0, 0, 255])
    return points_on_background


def grayscale_to_rgb(tensor, permute=False):
    # Tensor [height, width, 3], or
    # Tensor [height, width, 1], or
    # Tensor [1, height, width], or
    # Tensor [3, height, width]

    # if permute -> Convert to [height, width, 3]
    if permute:
        if tensor.size()[0] < 4:
            tensor = tensor.permute(1, 2, 0)
        if tensor.size()[2] == 1:
            return torch.stack([tensor[:, :, 0]] * 3, dim=2)
        else:
            return tensor
    else:
        if tensor.size()[0] == 1:
            return torch.stack([tensor[0, :, :]] * 3, dim=0)
        else:
            return tensor





def visualize_events(src:Path):
    output_dir = src.parent / src.name + " [VIS]"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = src.glob("*.npy")

    for event_file in tqdm(files):
        events = numpy.load(event_file)
        event_img = events_to_event_image(events, 480, 640)

        save_file = output_dir / (event_file.stem + ".png")
        
        imageio.imwrite(save_file, event_img.numpy().transpose(1, 2, 0))