""" Network architecture visualizer using graphviz """
import os
import sys
from graphviz import Digraph
import genotypes as gt

from template_lib.utils import get_attr_kwargs


def plot(genotype, cfg, **kwargs):
    """ make DAG plot and save to file_path as .png """

    format               = get_attr_kwargs(cfg, 'format', **kwargs)
    file_path            = get_attr_kwargs(cfg, 'file_path', **kwargs)
    caption              = get_attr_kwargs(cfg, 'caption', default=None, **kwargs)

    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format=format,
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor='lightblue')

    for i, edges in enumerate(genotype):
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)
    pass


def main(args, myargs):

    genotype_str = args.genotype
    try:
        genotype = gt.from_str(genotype_str)
    except AttributeError:
        raise ValueError("Cannot parse {}".format(genotype_str))

    plot(genotype.normal, file_path=os.path.join(args.outdir, "normal"), cfg=args)
    plot(genotype.reduce, file_path=os.path.join(args.outdir, "reduction"), cfg=args)
    pass


def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  args = config2args(myargs.config, args1)

  main(args=args, myargs=myargs)

if __name__ == '__main__':
  run()

