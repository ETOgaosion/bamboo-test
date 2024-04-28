import os
import argparse
from test.bambootest.lab_res_parser import *

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-each', action='store_true')
    parser.add_argument('--plot-avg', action='store_true')
    parser.add_argument('--plot-dec', action='store_true')
    parser.add_argument('--plot-layer', action='store_true')
    parser.add_argument('--calculate-rdzv', action='store_true')
    parser.add_argument('--calculate-fallback', action='store_true')
    parser.add_argument('--calculate-pipeline-delta', action='store_true')
    parser.add_argument('--base-dir', type=str, default='test/bambootest/')
    parser.add_argument('--dir', type=str, default='res')
    parser.add_argument('--target', type=str, default='test/bambootest/fig')
    return parser.parse_args(args)

def main(args):
    options = parse(args)
    print(options)

    global base_dir
    base_dir = options.base_dir
    dirs = sorted(os.listdir(base_dir + options.dir))
    while '.DS_Store' in dirs: dirs.remove('.DS_Store')
    while '.gitignore' in dirs: dirs.remove('.gitignore')

    if options.plot_each:
        plot_each_main(dirs, options.dir, options.target)

    if options.plot_avg:
        plot_avg_main(dirs, options.dir, options.target)
    
    if options.plot_dec:
        plot_dec_main(dirs, options.dir, options.target)

    if options.plot_layer:
        plot_layer_main(dirs, options.dir, options.target)

    if options.calculate_rdzv:
        calculate_preparation_main()

    if options.calculate_fallback:
        calculate_fallback_main(dirs)

    if options.calculate_pipeline_delta:
        calculate_pipeline_delta()
    