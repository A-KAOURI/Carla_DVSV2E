import carla
import numpy as np

import imageio.v2 as imageio
from flow_vis import flow_uv_to_colors
from visualization import events_to_event_image
from carla import ColorConverter as cc

import weakref
import time

"""
===============================================================================================================
DEFINING CALLBACKS
===============================================================================================================
Each callback function has three positional inputs : data, recorder, sensor
    - data : The sensor data captured by Carla server
    - recorder : The DataRecorder instance
    - sensor : The corresponding sensor's name

The callback definition depends on the type of sensor and data.
Note that, for data synchrony, a buffer is used to wait for RGB captures
===============================================================================================================
"""

def dummy_callback(data, recorder, sensor):
    """
        A dummy callback to be loaded when the actual callback is not defined.
        This allows the simulation to run, but the sensor for which the callback is
        not defined will not work.
    """
    return

def rgb_callback(image, recorder, sensor):
    if isinstance(recorder, weakref.ReferenceType):
        recorder = recorder()
    save_dir = recorder.data_save_dirs[sensor]
    
    frame = recorder.get_relative_frame(image.frame)
    frame_file_name = '{:06d}.png'.format(frame)
    image.save_to_disk(str(save_dir / frame_file_name))

    if not recorder.v2e_enabled:
        recorder.bgr_buffer.put(frame)


def rgb_hfps_callback(image, recorder, sensor):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    if isinstance(recorder, weakref.ReferenceType):
        recorder = recorder()
    recorder.bgr_buffer.put(array)


def segmentation_callback(segmentation, recorder, sensor):
    if isinstance(recorder, weakref.ReferenceType):
        recorder = recorder()
    save_dir = recorder.data_save_dirs[sensor]

    frame = recorder.get_relative_frame(segmentation.frame)
    frame_file_name = '{:06d}.png'.format(frame)
    segmentation.save_to_disk(str(save_dir / frame_file_name))

    vis_dir = recorder.visual_dirs[sensor]
    if vis_dir is not None:
        segmentation.save_to_disk(str(vis_dir / frame_file_name), carla.ColorConverter.CityScapesPalette)


def flow_callback(data, recorder, sensor):
    if isinstance(recorder, weakref.ReferenceType):
        recorder = recorder()
    save_dir = recorder.data_save_dirs[sensor]

    frame = recorder.get_relative_frame(data.frame)
    frame_file_name = '{:06d}.png'.format(frame)

    width = data.width
    height = data.height

    raw_flow = np.frombuffer(data.raw_data, dtype=np.float32)
    raw_flow = raw_flow.reshape((height, width, 2))

    flow = raw_flow.copy()
    flow[:, :, 0] *= width * 0.5
    flow[:, :, 1] *= height * -0.5
    
    # Visualize
    vis_dir = recorder.visual_dirs[sensor]
    if vis_dir is not None:
        BUILTIN  = False
        if BUILTIN:
            image = flow.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            imageio.imwrite(str(vis_dir / frame_file_name), array)
        else:
            vis = flow_uv_to_colors(u = flow[:,:,0], v = flow[:,:,1])
            imageio.imwrite(str(vis_dir / frame_file_name), vis.astype('uint8'))

    # Save flow (Encoded in a 16-bit RGB image)
    flow_uv = np.ndarray((height, width, 3))
    flow_uv [:,:,:2] = flow
    flow_uv = flow_uv * 128.0 + 2**15 
    flow_uv[:,:,2] = 1
    imageio.imwrite(str(save_dir / frame_file_name), flow_uv.astype(np.uint16), format='PNG-FI')


def dvs_callback(events, recorder, sensor):
    if isinstance(recorder, weakref.ReferenceType):
        recorder = recorder()
    save_dir = recorder.data_save_dirs[sensor]

    frame = recorder.get_relative_frame(events.frame)
    events_file_name = '{:06d}.npy'.format(frame)

    # Record the time of the callback
    recorder.dvs_curr_ts = time.time()
    with open(recorder.save_dir / "Events_DVS_times.txt", 'a') as f:
        f.write("{},{}\n".format(frame, recorder.dvs_curr_ts - recorder.dvs_prev_ts))
        recorder.dvs_prev_ts = recorder.dvs_curr_ts

    dvs_events = np.frombuffer(events.raw_data, dtype=np.dtype([
        ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', bool)]))
    
    t = dvs_events[:]['t']
    x = dvs_events[:]['x']
    y = dvs_events[:]['y']
    p = dvs_events[:]['pol'] * 2.0 - 1.0
    event_array = np.vstack([t, x, y, p]).transpose()
    np.save(save_dir / events_file_name, event_array)
    

    vis_dir = recorder.visual_dirs[sensor]
    if vis_dir is not None:
        dvs_image = events_to_event_image(event_array, events.height, events.width)
        dvs_image = dvs_image.numpy().transpose(1, 2, 0)
        imageio.imwrite(str(vis_dir / '{:06d}.png'.format(frame)), dvs_image)


def depth_callback(image, recorder, sensor):
    if isinstance(recorder, weakref.ReferenceType):
        recorder = recorder()
    save_dir = recorder.data_save_dirs[sensor]
    frame = recorder.get_relative_frame(image.frame)

    # Read Raw Depth from RGB encoding
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    RGB_encoded = array[:,:,:3][:,:,::-1] # From BGRA to RGB

    filename = '{:06d}.png'.format(frame)
    imageio.imwrite(save_dir / filename, RGB_encoded.astype('uint8'))
    
    # # Depth saved in the RGB encoded format
    # # To calculate the depth from the RGB channels :
    # R = RGB_encoded[:,:,0]
    # G = RGB_encoded[:,:,1]
    # B = RGB_encoded[:,:,2]
    # depth = (R + G * 256.0 + B * 256.0 ** 2) / (256.0 ** 3 - 1)     # 0 to 1 scale
    # depth = depth * 1000.0                                          # multiply by max distance in meters

    vis_dir = recorder.visual_dirs[sensor]
    if vis_dir is not None:
        frame_filename = '{:06d}.png'.format(frame)
        image.save_to_disk(str(vis_dir / frame_filename), carla.ColorConverter.LogarithmicDepth)