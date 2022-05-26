import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Style transfer images using StyleGANv3 blending'
    )

    parser.add_argument('--source_model', type=str, required=True,
                        help='Source model pickle for blending')
    parser.add_argument('--target_model', type=str, required=True,
                        help='Target model pickle for blending')
    parser.add_argument('--vgg_model', type=str, default='models/vgg16.pkl',
                        help='Target model pickle for blending')
    parser.add_argument('--datadir', type=str, default='images',
                        help='Directory storing images to process')
    parser.add_argument('--num_steps', type=int, default=300,
                        help='Number of optimization steps')
    parser.add_argument('--save_optim', action='store_true', default=False,
                        help='Whether to save optimized images')
    parser.add_argument('--save_trace', action='store_true', default=False,
                        help='Whether to save intermediate optimization images')

    return parser.parse_args()
