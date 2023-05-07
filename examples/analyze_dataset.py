import os
import yaml
import argparse
import torch
import matplotlib.pyplot as plt
# import tsne
from sklearn.manifold import TSNE
from umap import UMAP
from neural_canvas.utils import utils
from neural_canvas.runners import runner2d


def load_data(dataset_path):
    # loads metadata from dataset of tif files
    data = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.tif'):
                path = os.path.join(root, file)
                _, metadata = utils.load_tif_metadata(path)
                data.append(metadata)
    return data


def collate_metadata(data):
    # collates metadata from dataset of tif files
    metadata = {}
    for key in data[0].keys():
        if key == 'latent':
            metadata[key] = [d[key].reshape(-1).norm() for d in data]
        elif key not in ['graph', 'seed', 'device', 'x_dim', 'y_dim', 'z_dim', 'c_dim']:
            metadata[key] = [d[key] for d in data]
    return metadata


def plot_metadata_histograms(metadata, output_dir):
    # plots histograms of metadata from dataset of tif files
    # all keys are in subplots arranged in a grid in a single large figure

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
    for i, key in enumerate(metadata.keys()):
        ax = axes[i // 4, i % 4]
        print ('key', key)
        ax.hist(metadata[key], bins=20)
        ax.set_title(key)
        ax.set_ylabel('Count')
        ax.set_xlabel(key)
        ax.set_ylim(0, max([len(metadata[key]) for key in metadata.keys()]))
        ax.set_xlim(min(metadata[key]), max(metadata[key]))
        ax.grid(True)
        ax.legend([f'Count: {len(metadata[key])}', f'Min: {min(metadata[key])}'], title=key)
    plt.savefig(os.path.join(output_dir, 'metadata_histograms.png'))


def tsne_metadata(metadata, runner):
    # performs t-SNE on metadata from dataset of tif files
    # individual points are a concatenation of latent points combined with network weights
    # weights are given by reinitializing network with metadata
    points = []
    for i, data in enumerate(metadata):
        latent = runner.reinit_model_from_metadata(
            output_shape=(256,256,3),
            metadata=data)
        weights = torch.cat(
            list(map(lambda x: torch.tensor(x).reshape(-1),
                     runner.model.map_fn.state_dict().values())))
        vals = torch.cat([latent.reshape(-1), weights.reshape(-1)])
        points.append(vals)
    points = torch.stack(points)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(points)
    # plot tsne
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1])
    plt.savefig(os.path.join(runner.output_dir, 'tsne.png'))
    return tsne_embedding
    

def umap_metadata(metadata, runner):
    # performs UMAP on metadata from dataset of tif files
    # individual points are a concatenation of latent points combined with network weights
    # weights are given by reinitializing network with metadata
    points = []
    for i, data in enumerate(metadata):
        latent = runner.reinit_model_from_metadata(
            output_shape=(256,256,3),
            metadata=data)
        weights = torch.cat(runner.model.map_fn.state_dict().values())
        vals = torch.cat([latent.reshape(-1), weights.reshape(-1)])
        points.append(vals)
    points = torch.stack(points)
    umap = UMAP(n_components=2, random_state=42)
    umap_embedding = umap.fit_transform(points)
    # plot umap
    plt.figure(figsize=(10, 10))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1])
    plt.savefig(os.path.join(runner.output_dir, 'umap.png'))
    return umap_embedding
    
def load_args(argv=None):
    parser = argparse.ArgumentParser(description='analysis-config')
    parser.add_argument('--conf', default=None, type=str, help='args config file')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to image to regenerate')
    parser.add_argument('--output_dir', type=str, default=None, help='path to output directory')
    args, _ = parser.parse_known_args(argv)
    if args.conf is not None:
        if os.path.isfile(args.conf):
            with open(args.conf, 'r') as f:
                defaults = yaml.safe_load(f)
            
            defaults = {k: v for k, v in defaults.items() if v is not None}
            parser.set_defaults(**defaults)
            args, _ = parser.parse_known_args(argv)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args


if __name__ == '__main__':
    args = load_args()

    if args.dataset_path is not None:
        data = load_data(args.dataset_path)
        metadata = collate_metadata(data)

    plot_metadata_histograms(metadata, args.output_dir)
    runner = runner2d.RunnerINRF2D(output_dir=args.output_dir)
    tsne_embedding = tsne_metadata(data, runner)
    umap_embedding = umap_metadata(data, runner)