import os
import shutil
import dnnlib
import torch
import legacy

from PIL import Image
from argparser import get_args
from blend import blend_models
from optimization import find_noise, generate_images
from metrics.metric_utils import get_feature_detector


args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load all required models
with dnnlib.util.open_url(args.source_model) as fp:
    G_source = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).eval()

with dnnlib.util.open_url(args.source_model) as fp:
    G_target = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).eval()

G_target = blend_models(G_source, G_target).to(device)
G_source = G_source.to(device)
vgg16 = get_feature_detector(args.vgg16_model).eval().to(device).eval()

# prepare dirs
if args.save_trace:
    if os.path.isdir('trace'):
        shutil.rmtree('trace')
    os.mkdir('trace')

if args.save_optim:
    if os.path.isdir('optim'):
        shutil.rmtree('optim')
    os.mkdir('optim')

if os.path.isdir('result'):
    shutil.rmtree('result')
os.mkdir('result')

# process images
image_files = os.listdir(args.datadir)
for image_file in image_files:
    pil_image = Image.open(image_file)
    image_name = image_file[:image_file.rfind('.')]
    noise_trace = find_noise(G_source, vgg16, pil_image, device, num_steps=args.num_steps)

    if args.save_trace:
        trace_images = generate_images(G_source, noise_trace[::10])
        trace_path = os.path.join('trace', image_name)
        os.mkdir(trace_path)

        for i, image in enumerate(trace_images):
            image = Image.fromarray(image, 'RGB')
            image_path = image_name + f'-trace-{i * 10:03d}.png'
            image.save(os.path.join(trace_path, image_path))

    if args.save_optim:
        image = generate_images(G_source, noise_trace[-1:])[0]
        image = Image.fromarray(image, 'RGB')
        image_path = image_name + '-optim.png'
        image.save(os.path.join('optim', image_path))

    image = generate_images(G_target, noise_trace[-1:])[0]
    image = Image.fromarray(image, 'RGB')
    image_path = image_name + '.png'
    image.save(os.path.join('result', image_path))
