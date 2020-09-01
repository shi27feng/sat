import sys

from graphviz import Graph

from cnf import CNF
from util import adj

f = CNF.from_file(sys.argv[1])
a_pos, a_neg = adj(f)
a_pos = a_pos.todense()
a_neg = a_neg.todense()

n, m = a_pos.shape

g = Graph('G', filename=sys.argv[2], format='pdf', engine='neato')

for x in range(1, n + 1):
    g.node(f'x{x}', label='x', color='blue', fillcolor='blue', shape='point')

for c in range(1, m + 1):
    g.node(f'c{c}', label='y', color='red', fillcolor='red', shape='point')

pw = sys.argv[3]

for x in range(n):
    for c in range(m):
        var = f'x{x + 1}'
        cl = f'c{c + 1}'
        if a_pos[x, c] == 1:
            # g.edge(var, cl, color='#ff0000', penwidth='0.001')
            g.edge(var, cl, color='#ff0000', penwidth=pw)
        if a_neg[x, c] == 1:
            # g.edge(var, cl, color='#0000ff', penwidth='0.001')
            g.edge(var, cl, color='#0000ff', penwidth=pw)

g.view()
