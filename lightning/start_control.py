import os
import cv2
import tqdm
import yaml
import glob
import argparse
import torch
import numpy as np
import time
from lightning_runner import RunnerLightning2D
from lightning_models import LightningModel2D
from lightning_generators import INRLinearMap, INRRandomGraph
from neural_canvas.utils import utils

import sys
import termios
import atexit
from select import select

"""
elif c == 119: # w # pan up
    print ('[w] Panning up...')
    pan_x_curr += kb_pan_delta
    if pan_x_curr == 0:
        pan_x_curr += kb_pan_delta
    fields = runner.model.construct_fields(
        output_shape=(args.x_dim, args.y_dim), 
        zoom=(zoom_x_curr, zoom_y_curr), 
        pan=(pan_x_curr, pan_y_curr))
elif c == 97: # a # pan left
    print ('[a] Panning left...')
    pan_y_curr -= kb_pan_delta
    if pan_y_curr == 0:
        pan_y_curr -= kb_pan_delta
    fields = runner.model.construct_fields(
        output_shape=(args.x_dim, args.y_dim), 
        zoom=(zoom_x_curr, zoom_y_curr), 
        pan=(pan_x_curr, pan_y_curr))
elif c == 115: # s # pan down
    print ('[s] Panning down...')
    pan_x_curr -= kb_pan_delta
    if pan_x_curr == 0:
        pan_x_curr -= kb_pan_delta
    fields = runner.model.construct_fields(
        output_shape=(args.x_dim, args.y_dim), 
        zoom=(zoom_x_curr, zoom_y_curr), 
        pan=(pan_x_curr, pan_y_curr))
elif c == 100: # d # pan right
    print ('[d] Panning right...')  
    pan_y_curr += kb_pan_delta
    if pan_y_curr == 0:
        pan_y_curr += kb_pan_delta  
    fields = runner.model.construct_fields(
        output_shape=(args.x_dim, args.y_dim), 
        zoom=(zoom_x_curr, zoom_y_curr), 
        pan=(pan_x_curr, pan_y_curr))
"""  
class KBHit:
    def __init__(self):
        """Creates a KBHit object that you can call to do various keyboard things."""

        if os.name == 'nt':
            pass
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)

    def set_normal_term(self):
        """ Resets to normal terminal.  On Windows this is a no-op."""
        if os.name == 'nt':
            pass

        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def getch(self):
        """ Returns a keyboard character after kbhit() has been called.
            Should not be called in the same program as getarrow().
        """

        s = ''

        if os.name == 'nt':
            return msvcrt.getch().decode('utf-8')

        else:
            return sys.stdin.read(1)


    def getarrow(self):
        ''' Returns an arrow-key code after kbhit() has been called. Codes are
        0 : up
        1 : right
        2 : down
        3 : left
        Should not be called in the same program as getch().
        '''
        c = sys.stdin.read(3)[2]
        vals = [65, 67, 66, 68]

        return vals.index(ord(c.decode('utf-8')))


    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.
        '''
        dr,dw,de = select([sys.stdin], [], [], 0)
        return dr != []
    
def load_args(argv=None):
    parser = argparse.ArgumentParser(description='INRF2D-edit-config')
    parser.add_argument('--conf', default=None, type=str, help='args config file')
    parser.add_argument('--latent_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--latent_scale', default=1.0, type=float, help='mutiplier on z')
    parser.add_argument('--num_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=256, type=int, help='out image width')
    parser.add_argument('--y_dim', default=256, type=int, help='out image height')
    parser.add_argument('--c_dim', default=3, type=int, help='channels')
    parser.add_argument('--mlp_layer_width', default=32, type=int, help='net width')    
    parser.add_argument('--activations', default='fixed', type=str,
        help='activation set for generator')
    parser.add_argument('--final_activation', default='sigmoid', type=str, help='last activation')
    parser.add_argument('--backbone', default='straight', type=str, choices=['straight', 'resnet', 'densenet'],)
    parser.add_argument('--graph_topology', default='mlp_fixed', type=str,
        help='graph style to use for generator', choices=['mlp_fixed', 'WS'])
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--ws_graph_nodes', default=10, type=int, help='number of nodes in ws graph')
    parser.add_argument('--weight_init', default='normal', type=str, help='weight init scheme')
    parser.add_argument('--weight_init_mean', default=0.0, type=float, help='weight init mean')
    parser.add_argument('--weight_init_std', default=1.0, type=float, help='weight init std')
    parser.add_argument('--weight_init_max', default=2.0, type=float, help='weight init max')
    parser.add_argument('--weight_init_min', default=-2.0, type=float, help='weight init min')

    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--ignore_seed', action='store_true', help='ignore random seed, ' \
        'useful for running the same config multiple times without changing seed each run')

    parser.add_argument('--output_dir', default='trial', type=str, help='output fn')
    parser.add_argument('--tmp_dir', default='trial', type=str, help='output fn')

    parser.add_argument('--colormaps', type=str, default=None, help='colormaps to save out',
        choices=['gray', 'hsv', 'lab', 'hls', 'luv'])

    # For Zoom videos
    parser.add_argument('--zoom_bounds', default=(.5, .5), type=tuple, help='zoom in/out boundaries')
    parser.add_argument('--zoom_scheduler', default=None, type=str, help='zoom in/out pacing',
        choices=['linear', 'geometric', 'cosine', 'sigmoid', 'exp', 'log', 'sqrt'])
    # For Pan videos
    parser.add_argument('--pan_bounds', default=(2, 2), type=tuple, help='pan boundaries')
    parser.add_argument('--pan_scheduler', default=None, type=str, help='pan pacing',
        choices=['linear', 'geometric', 'cosine', 'sigmoid', 'exp', 'log', 'sqrt'])
    # For regenerating and augmenting images
    parser.add_argument('--init_image_path', type=str, default=None, help='path to image to regenerate')
    parser.add_argument('--lerp_iters', type=int, default=1, help='number of keyframes to interpolate between')


    args, _ = parser.parse_known_args(argv)
    if os.path.isfile(args.conf):
        with open(args.conf, 'r') as f:
            defaults = yaml.safe_load(f)
        
        defaults = {k: v for k, v in defaults.items() if v is not None}
        parser.set_defaults(**defaults)
        args, _ = parser.parse_known_args(argv)
    
    return args


def precompute_latent_table(model, fields):
    x_dim, y_dim = fields['shape'][1:]
    latents = model.sample_latents(
        reuse_latents=None, 
        output_shape=(1, x_dim, y_dim))
    table = []
    for _ in tqdm.tqdm(range(20)):
        latents = model.sample_latents(
            reuse_latents=None, 
            output_shape=(1, x_dim, y_dim))
        table.append(latents)
    return table


def sample_from_table(table, sampler, n):
    samples = sampler(table)
    sample_start = samples[0]
    z_schedule = []
    for sample in samples[1:]:
        lerp_schedule = utils.lerp(sample_start, sample, n)
        z_schedule.extend(lerp_schedule)
        sample_start = sample  
    z_schedule.extend(utils.lerp(sample, samples[0], n))
    return z_schedule


if __name__ == '__main__':
    args = load_args()
    if args.use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = LightningModel2D(
        latent_dim=args.latent_dim,
        latent_scale=args.latent_scale,
        output_shape=(args.x_dim, args.y_dim, args.c_dim),
        output_dir=args.output_dir,
        tmp_dir=args.tmp_dir,
        seed=args.seed,
        device=device,
    )
    files = glob.glob('/home/neale/repos/neural_canvas_datasets/datasets/rlhf_dataset2/good/*.tif')
    runner = RunnerLightning2D(output_dir=args.output_dir, skip_blank_generations=True)
    print ('==> Initialized runner')
    runner.reinit_model_from_metadata(output_shape=(args.x_dim, args.y_dim, args.c_dim),
                                      path=args.init_image_path)
    print ('==> Loaded Model from init path')
    fields = runner.model.construct_fields(output_shape=(args.x_dim, args.y_dim), zoom=(.5, .5), pan=(2, 2))

    table = precompute_latent_table(runner.model, fields)
    print ('==> Computed Latent Table')
    sample_table = lambda table: [table[i] for i in np.random.randint(0, len(table), args.lerp_iters)]
    z_schedule = sample_from_table(table, sample_table, args.num_samples)

    # store map parameters
    z_dim = runner.model.map_fn.latent_dim
    layer_width = runner.model.map_fn.layer_width
    scale = args.latent_scale  
    reset_mlp = lambda: INRLinearMap(
        runner.model.latent_dim, args.c_dim, runner.model.mlp_layer_width, runner.model.input_encoding_dim,
        runner.model.activations, runner.model.final_activation, runner.model.backbone, runner.model.num_layers, runner.model.device
    )  
    reset_ws = lambda: INRRandomGraph(
        runner.model.latent_dim, args.c_dim, runner.model.mlp_layer_width, runner.model.input_encoding_dim,
        runner.model.activations, runner.model.final_activation, runner.model.backbone, runner.model.num_layers, runner.model.device
    )  
    #opt_generate = torch.compile(runner.model.generate)
    zoom_x_curr = .5
    zoom_y_curr = .5
    pan_x_curr = 2
    pan_y_curr = 2
    kb_zoom_alpha = 1.1
    kb_pan_delta = .1

    total_iters = 0
    fps_queue = []
    kb = KBHit()
    print('Hit any key, or ESC to exit')
    control_loop_start = time.time()
    while True:
            if kb.kbhit():
                c = ord(kb.getch())
                if c == 27: # ESC
                    print (f'Exiting...')
                    print (f'Total iters: {total_iters}, FPS: {np.array(fps_queue).mean()}')
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif c == 43: # + # zoom in
                    print ('[CW Knob1] Zooming in...')
                    zoom_x_curr *= kb_zoom_alpha
                    zoom_y_curr *= kb_zoom_alpha
                    fields = runner.model.construct_fields(
                        output_shape=(args.x_dim, args.y_dim), 
                        zoom=(zoom_x_curr, zoom_y_curr), 
                        pan=(pan_x_curr, pan_y_curr))
                elif c == 45: # - # zoom out
                    print ('[CCW Knob1] Zooming out...')
                    zoom_x_curr /= kb_zoom_alpha
                    zoom_y_curr /= kb_zoom_alpha
                    fields = runner.model.construct_fields(
                        output_shape=(args.x_dim, args.y_dim), 
                        zoom=(zoom_x_curr, zoom_y_curr), 
                        pan=(pan_x_curr, pan_y_curr))
                elif c == 42: # * # reset zoom
                    print ('[Click Knob1] Resetting zoom...')
                    zoom_x_curr = .5
                    zoom_y_curr = .5
                    fields = runner.model.construct_fields(
                        output_shape=(args.x_dim, args.y_dim), 
                        zoom=(zoom_x_curr, zoom_y_curr), 
                        pan=(pan_x_curr, pan_y_curr))
                elif c == 32: # space # generate new latents table
                    print ('[Click Knob2] Generating new latents table...')
                    table = precompute_latent_table(runner.model, fields)
                    z_schedule = sample_from_table(table, sample_table, args.num_samples)
                elif c == 93: # ] # clockwise large knob
                    print ('[CW Knob3] Changing weights...')
                    runner.reinit_model_from_metadata(output_shape=(args.x_dim, args.y_dim, args.c_dim),
                                    path=files[np.random.randint(0, len(files))])
                    table = precompute_latent_table(runner.model, fields)
                    print ('==> Computed Latent Table')
                    z_schedule = sample_from_table(table, sample_table, args.num_samples)
                    #runner.model.init_map_weights()
                elif c == 91: # [ # counterclockwise large knob
                    print ('[CCW Knob3] Generate new activations...')
                    runner.model.map_fn.generate_new_acts()
                elif c == 59: # ; # click large knob
                    print ('[Click Knob3] Select new latents...')
                    z_schedule = sample_from_table(table, sample_table, args.num_samples)

                elif c == 97:
                    print ('[a] Changing latent dim to 8...')
                    runner.model.latent_dim = 8
                    runner.model.map_fn = reset_model()
                    table = precompute_latent_table(runner.model, fields)
                    z_schedule = sample_from_table(table, sample_table, args.num_samples)

                elif c == 98:
                    print ('[b] Changing latent dim to 16...')
                    runner.model.latent_dim = 16
                    runner.model.map_fn = reset_model()
                    table = precompute_latent_table(runner.model, fields)
                    z_schedule = sample_from_table(table, sample_table, args.num_samples)

                elif c == 99:
                    print ('[c] Changing latent dim to 32...')
                    runner.model.latent_dim = 32
                    runner.model.map_fn = reset_model()
                    table = precompute_latent_table(runner.model, fields)
                    z_schedule = sample_from_table(table, sample_table, args.num_samples)

                elif c == 100:
                    print ('[d] Changing latent dim to 64...')
                    runner.model.latent_dim = 64
                    runner.model.map_fn = reset_model()
                    table = precompute_latent_table(runner.model, fields)
                    z_schedule = sample_from_table(table, sample_table, args.num_samples)

                elif c == 101:
                    print ('[e] Changing mlp layer width to 8...')
                    runner.model.mlp_layer_width = 8
                    runner.model.map_fn = reset_model()
                elif c == 102:
                    print ('[f] Changing mlp layer width to 16...')
                    runner.model.mlp_layer_width = 16
                    runner.model.map_fn = reset_model()
                elif c == 103:
                    print ('[g] Changing mlp layer width to 32...')
                    runner.model.mlp_layer_width = 32
                    runner.model.map_fn = reset_model()
                elif c == 104:
                    print ('[h] Changing mlp layer width to 64...')
                    runner.model.mlp_layer_width = 64
                    runner.model.map_fn = reset_model()
                ### ROW 4
                elif c == 105:
                    print ('[i] Change graph to MLP...')

                elif c == 106:
                    print ('[j] Change graph to WS...')
     
                elif c == 107:
                    print ('[k] Redo WS graph')

                elif c == 108:
                    print ('[l] Changing latent scale to 8.0...')

            # we will generate a video by interpolating between the start and end latents
            # but this will be done multiple times, with the final one lerping to the start
            latent_iter_curr = z_schedule[total_iters % len(z_schedule)]
            latent = runner.model.sample_latents(reuse_latents=latent_iter_curr, output_shape=(args.x_dim, args.y_dim))
            frame = runner.model.generate(fields, latent)[0]
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            cv2.namedWindow('frame',cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


            ind = str(total_iters).zfill(8)
            total_iters += 1
            if total_iters % 500 == 0:
                control_loop_ckpt = time.time()
                fps = 500 / (control_loop_ckpt-control_loop_start)
                fps_queue.append(fps)
                if len(fps_queue) >= 100:
                    fps_queue.pop(0)
                control_loop_start = control_loop_ckpt
                print(f'Generator 500-frame FPS: {fps}, total avg: {np.array(fps_queue).mean()}')