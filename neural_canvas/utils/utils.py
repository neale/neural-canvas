import os
import glob
import shutil
import logging
import warnings

import torch
import tifffile
import numpy as np
import networkx
import matplotlib.pyplot as plt


logging.getLogger().setLevel(logging.ERROR)


def lerp(z1, z2, n):
    delta = (z2 - z1) / (n + 1)
    total_frames = n + 2
    states = []
    for i in range(total_frames):
        z = z1 + delta * float(i)
        states.append(z)
    states = torch.stack(states)
    return states


def load_image_as_tensor(path, output_dir='/tmp', device='cpu'):
    import cv2
    target = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    target = target / 127.5 - 1 
    target = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float().to(device)
    target_fn = f'{output_dir}/target'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    write_image(path=target_fn,
        img=(target.permute(0, 2, 3, 1)[0].cpu().numpy()+1)*127.5, suffix='jpg')
    return target

def unnormalize_and_numpy(x, output_activation='tanh'):
    x = x.detach().cpu().numpy()
    if output_activation == 'tanh':
        x = ((x + 1.) * 127.5).astype(np.uint8)
    elif output_activation == 'sigmoid':
        x = (x * 255.).astype(np.uint8)
    else:
        x = (x * (255./x.max())).astype(np.uint8)
    return x

def write_image(path, img, suffix='jpg', metadata=None, colormaps=None):
    import cv2
    assert suffix in ['jpg', 'png', 'bmp', 'jpeg', 'tif'], f'Invalid suffix for file, got {suffix}'
    if suffix in ['jpg', 'png', 'bmp', 'jpeg']:
        if colormaps is not None:
            colormapped_imgs = image_colormaps(img, colormaps)
            for cmap, cmap_img in colormapped_imgs.items():
                cmap_path = path + f'_{cmap}.{suffix}'
                cv2.imwrite(cmap_path, cmap_img)
                assert os.path.isfile(cmap_path)
        else:
            path = path + f'.{suffix}'
            cv2.imwrite(path, img)
            assert os.path.isfile(path)
    elif suffix == 'tif':
        if metadata is None:
            warnings.warn('No metadata provided for tiff file, data will not be reproducible.')
        path = path + '.tif'
        tifffile.imwrite(path, img, metadata=metadata)
    else:
        raise NotImplementedError


def image_colormaps(img, colormaps):
    import cv2
    colormaps = {c.lower(): None for c in colormaps}
    if img.shape[-1] == 3:
        colormaps['rgb'] = img
        if 'hsv' in colormaps:
            colormaps['hsv'] = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if 'gray' in colormaps:
            colormaps['gray'] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if 'lab' in colormaps:
            colormaps['lab'] = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        if 'hls' in colormaps:
            colormaps['hls'] = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if 'luv' in colormaps:
            colormaps['luv'] = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    else:
        img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        colormaps['gray'] = img
        if 'rgb' in colormaps:
            colormaps['rgb'] = img2
        if 'hsv' in colormaps:
            colormaps['hsv'] = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
        if 'lab' in colormaps:
            colormaps['lab'] = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
        if 'hls' in colormaps:
            colormaps['hls'] = cv2.cvtColor(img2, cv2.COLOR_RGB2HLS)
        if 'luv' in colormaps:
            colormaps['luv'] = cv2.cvtColor(img2, cv2.COLOR_RGB2LUV) 
    return colormaps


def save_repository(search_dir, output_dir):
    assert os.path.isdir(output_dir), f'{output_dir} not found'
    assert os.path.isdir(search_dir), f'{search_dir} not found'
    py_files = glob.glob('*.py')
    assert len(py_files) > 0
    for fn in py_files:
        shutil.copy(fn, os.path.join(output_dir, fn))


def load_tif_metadata(path):
    assert os.path.isfile(path), f'{path} not found'
    try:
        with tifffile.TiffFile(path) as tif:
            img = tif.asarray()
            data = tif.shaped_metadata[0]
    except:
        warnings.warn(f'Could not load metadata from {path}')
        return None, None
    metadata = {  # these are the keys that are always present
        'seed': int(data['seed']),
        'latent_dim': int(data['latent_dim']),
        'latent_scale': float(data['latent_scale']),
        'x_dim': int(data['x_dim']),
        'y_dim': int(data['y_dim']),
        'c_dim': int(data['c_dim']),
        'device': data['device'],
    }
    if 'z_dim' in data:
        metadata['z_dim'] = int(data['z_dim'])
    else:
        metadata['z_dim'] = int(data['x_dim'])
    for int_key in ['mlp_layer_width', 'conv_feature_map_size', 'input_encoding_dim', 'num_graph_nodes']:
        try:
            metadata[int_key] = int(data[int_key])
        except KeyError:
            metadata[int_key] = None
            warnings.warn(f'Key {int_key} not found in metadata, setting to None.')
    for float_key in ['weight_init_mean', 'weight_init_std', 'weight_init_max', 'weight_init_min']:
        try:
            metadata[float_key] = float(data[float_key])
        except KeyError:
            metadata[float_key] = None
            warnings.warn(f'Key {float_key} not found in metadata, setting to None.')
    for str_key in ['activations', 'graph', 'final_activation', 'weight_init', 'graph_topology']:
        try:
            metadata[str_key] = data[str_key]
        except KeyError:
            metadata[str_key] = None
            warnings.warn(f'Key {str_key} not found in metadata, setting to None.')
    for tensor_key in ['latents']:
        try:
            metadata[tensor_key] = torch.Tensor(data[tensor_key])
        except KeyError:
            metadata[tensor_key] = None
            warnings.warn(f'Key {tensor_key} not found in metadata, setting to None.')
    return img, metadata


def draw_graph(num_nodes, random_graph, graph, c_dim=3, img=None):
    import cv2
    graph.dpi = 1000
    options = {
        'label': '',
        "font_size": 36,
        "node_size": 3000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 3,
        "width": 2,
        "with_labels": False,
    }
    if random_graph:
        if num_nodes > 40:
            plot_size = 30
        elif num_nodes > 20:
            plot_size = 90
        elif num_nodes > 10:
            plot_size = 200
        else:
            plot_size = 250
        options['node_size'] = plot_size

    H_layout = networkx.nx_pydot.pydot_layout(graph, prog='dot')
    networkx.draw_networkx(graph, H_layout, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.savefig('temp_net.png', dpi=700)
    x = cv2.imread('temp_net.png')

    if c_dim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    else:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.bitwise_not(x)
    x_s = cv2.resize(x, (100, 100), interpolation=cv2.INTER_AREA)
    if c_dim == 1:
        x_s = x_s.reshape((x_s.shape[0], x_s.shape[1], 1))
    img_trans = np.zeros_like(img)
    img_trans[-x_s.shape[0]:, -x_s.shape[1]:, :] = x_s
    plt.close('all')
    return img_trans


def write_video(frames, save_dir, save_name='video'):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    width = frames[0].shape[0]
    height = frames[0].shape[1]
    path = os.path.join(save_dir, save_name)
    video = cv2.VideoWriter(f'{path}.mp4', fourcc, 10., (width, height))
    for frame in frames: 
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()