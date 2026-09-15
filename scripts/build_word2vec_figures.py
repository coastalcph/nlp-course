#!/usr/bin/env python3
"""Rebuild the word-embedding lecture's exact SVG diagrams and measured plots.

Run from any directory: python scripts/build_word2vec_figures.py
Requires NumPy and Matplotlib. The experiment lives in the optional-notes notebook cell tagged
word2vec-demo; this script executes that cell rather than duplicating the trainer.
No network, pretrained model, GPU, or image-generation service is needed.
"""
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'img' / 'word2vec_2027'
OUT.mkdir(parents=True, exist_ok=True)
NOTEBOOK = ROOT / 'chapters' / 'dl-representations_notes.ipynb'
notebook = json.loads(NOTEBOOK.read_text())
demo_cells = [c for c in notebook['cells']
              if 'word2vec-demo' in c['metadata'].get('tags', [])]
if len(demo_cells) != 1:
    raise ValueError('Expected exactly one word2vec-demo cell')
demo = {}
exec(compile(''.join(demo_cells[0]['source']), str(NOTEBOOK) + ':word2vec-demo', 'exec'), demo)
C, E, O = (demo[key] for key in ('C', 'E', 'O'))
words = demo['vocabulary']
BLUE, TEAL, ORANGE = '#245da8', '#087c73', '#b45309'
INK, MUTED, LINE, PALE = '#172b45', '#52637a', '#cbd5e1', '#f1f5f9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK,
                     'svg.fonttype': 'none', 'svg.hashsalt': 'word2vec-2027'})


class Diagram:
    def __init__(self, height=400, title=''):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}" role="img">',
                      f'<title>{escape(title)}</title>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L7,3 z" fill="context-stroke"/></marker></defs>',
                      '<rect width="1100" height="100%" fill="white"/>']

    def text(self, x, y, value, size=24, fill=INK, weight=400, anchor='start'):
        # Explicit baselines keep labels stable in Reveal, browsers and SVG viewers.
        self.parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def lines(self, x, y, values, size=24, fill=INK, gap=34, anchor='start', weight=400):
        for i, value in enumerate(values):
            self.text(x, y + gap * i, value, size, fill, weight, anchor)

    def rect(self, x, y, w, h, fill=PALE, stroke=LINE, radius=10, sw=1.5):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, color=LINE, sw=2, arrow=False, dash=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ''
        dashattr = ' stroke-dasharray="7 5"' if dash else ''
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{marker}{dashattr}/>')

    def box(self, x, y, w, h, lines, color=BLUE, size=24):
        self.rect(x, y, w, h, fill='white', stroke=color, sw=2)
        gap = 33
        baseline = y + h / 2 - (len(lines)-1)*gap/2 + size*.34
        self.lines(x+w/2, baseline, lines, size=size, fill=color, anchor='middle', gap=gap)

    def matrix(self, x, y, values, row_labels=None, col_labels=None, cell_w=75, cell_h=40,
               color=BLUE, fmt='.2f', highlights=()):
        values = np.asarray(values)
        for i in range(values.shape[0]):
            if row_labels is not None:
                self.text(x-12, y+i*cell_h+cell_h*.68, row_labels[i], 20, anchor='end',
                          weight=700 if i in highlights else 400)
            for j in range(values.shape[1]):
                fill = '#e5eefb' if i in highlights else PALE
                self.rect(x+j*cell_w, y+i*cell_h, cell_w-2, cell_h-2, fill=fill, stroke='white', radius=2)
                v = format(values[i, j], fmt)
                self.text(x+(j+.5)*cell_w-1, y+i*cell_h+cell_h*.67, v, 19, color, anchor='middle')
        if col_labels is not None:
            for j, label in enumerate(col_labels):
                self.text(x+(j+.5)*cell_w, y-14, label, 18, anchor='middle')

    def save(self, name):
        (OUT / (name+'.svg')).write_text('\n'.join(self.parts+['</svg>'])+'\n')


def save_plot(fig, name):
    target = OUT/(name+'.svg')
    fig.savefig(target, metadata={'Date': None}, facecolor='white')
    # Matplotlib emits trailing spaces in SVG path data; keep generated diffs clean.
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


s = Diagram(390, 'A tiny corpus and its context pairs')
s.text(30, 40, '12 sentences', 27, weight=700)
for i, (phrase, count) in enumerate([('apples are tasty', 4), ('oranges are tasty', 4), ('rabbits are furry', 2), ('hamsters are furry', 2)]):
    s.text(35, 100+i*55, phrase, 26)
    s.text(340, 100+i*55, f'× {count}', 25, MUTED)
s.line(415, 55, 415, 330)
s.text(470, 40, 'Zoom into one sentence', 27, weight=700)
for x, text, color in [(470, 'apples', BLUE), (665, 'are', TEAL), (850, 'tasty', TEAL)]:
    s.box(x, 85, 160, 65, [text], color)
s.text(550, 181, 'centre', 21, BLUE, anchor='middle')
s.text(837, 181, 'context words', 21, TEAL, anchor='middle')
s.line(555, 202, 555, 240, BLUE, arrow=True)
s.line(555, 220, 905, 220, BLUE)
s.line(905, 220, 905, 245, BLUE, arrow=True)
s.box(465, 254, 260, 70, ['(apples, are)'], BLUE)
s.box(765, 254, 280, 70, ['(apples, tasty)'], BLUE)
s.text(550, 375, '12 sentences × 6 ordered pairs = 72 training pairs', 24, anchor='middle')
s.save('corpus_pairs')

s = Diagram(420, 'Observed centre-context counts')
s.text(470, 35, 'context word', 25, weight=700, anchor='middle')
s.matrix(140, 85, C, words, words, cell_w=90, cell_h=43, fmt='.0f', highlights=(0, 1))
s.line(798, 92, 798, 165, TEAL, sw=4)
s.lines(830, 112, ['Same contexts', 'for apples', 'and oranges'], 23, TEAL)
s.lines(825, 280, ['No apples–oranges', 'pair is needed.'], 22, MUTED)
s.text(36, 409, 'Rows are word representations; columns are context features.', 23)
s.save('counts')

with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log(C*C.sum()/(C.sum(1)[:, None]*C.sum(0)[None, :]))
ppmi = np.maximum(pmi, 0)
U, singular_values, Vt = np.linalg.svd(ppmi, full_matrices=False)
dense_counts = U[:, :2] * singular_values[:2]
# Canonicalize arbitrary SVD signs for stable diagrams across LAPACK versions.
for j in range(2):
    if dense_counts[np.argmax(abs(dense_counts[:, j])), j] < 0:
        dense_counts[:, j] *= -1
s = Diagram(385, 'From co-occurrence counts to dense count-based vectors')
s.text(150, 40, 'Count pairs', 27, BLUE, 700, 'middle')
s.text(528, 40, 'Weight association', 27, TEAL, 700, 'middle')
s.text(929, 40, 'Compress to d = 2', 27, BLUE, 700, 'middle')
s.box(25, 90, 250, 122, ['apples–are: 4', 'apples–tasty: 4'], BLUE, 23)
s.line(290, 150, 350, 150, MUTED, arrow=True)
s.box(370, 90, 310, 122, ['PPMI: 0.405', 'PPMI: 0.811'], TEAL, 23)
s.line(695, 150, 725, 150, MUTED, arrow=True)
s.matrix(843, 80, dense_counts[:4], words[:4], ['axis 1', 'axis 2'], cell_w=99, cell_h=39)
s.text(920, 278, 'U₂ Σ₂', 30, BLUE, anchor='middle')
s.lines(30, 275, ['Equal counts, but “tasty”', 'is a less frequent context.'], 23)
s.lines(378, 275, ['Count → PPMI → truncated SVD', 'Dense vectors, still count-based.'], 23, TEAL)
s.save('count_to_dense')

s = Diagram(400, 'Original word2vec initialization')
s.text(260, 38, 'Input E: small random values', 27, BLUE, 700, 'middle')
s.text(805, 38, 'Output O: all zeros', 27, TEAL, 700, 'middle')
s.matrix(158, 93, demo['initial_E'][:4, :3], words[:4], ['1', '2', '3'], cell_w=91, cell_h=48, fmt='.3f')
s.matrix(700, 93, np.zeros((4, 3)), words[:4], ['1', '2', '3'], cell_w=91, cell_h=48, fmt='.0f', color=TEAL)
s.text(303, 318, 'Uniform: −0.5/d to +0.5/d', 23, BLUE, anchor='middle')
s.text(828, 318, 'Every initial dot product is 0', 23, TEAL, anchor='middle')
s.text(550, 375, 'Shown: four words, first three coordinates of d = 8', 22, MUTED, anchor='middle')
s.save('initialization')

fig, ax = plt.subplots(figsize=(11, 3.4), layout='constrained')
colors = [TEAL if word == 'tasty' else '#b6cae5' for word in words]
ax.bar(words, np.full(7, 1/7), color=colors, width=.65)
ax.set_ylim(0, .23); ax.set_ylabel('p(context | apples)')
ax.set_yticks([0, 1/7], ['0', '1/7'])
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='x', labelsize=16)
ax.annotate('Observed context: tasty', xy=(5, 1/7), xytext=(3.4, .205),
            color=TEAL, fontsize=18, arrowprops={'arrowstyle':'->', 'color':TEAL, 'lw':2})
ax.set_title('At initialization: loss = log(7) ≈ 1.946', loc='left', fontsize=19, pad=16)
save_plot(fig, 'softmax')

s = Diagram(360, 'Three independent binary predictions in negative sampling')
s.box(25, 134, 220, 85, ['E[apples]'], BLUE, 28)
for y, label, score_label, color in [(38, 'O[tasty]', 'Observed pair: y = 1', TEAL),
                                      (142, 'O[furry]', 'Noise draw: y = 0', ORANGE),
                                      (246, 'O[rabbits]', 'Noise draw: y = 0', ORANGE)]:
    s.line(255, 176, 390, y+33, color, arrow=True)
    s.box(405, y, 220, 66, [label], color, 25)
    s.line(638, y+33, 693, y+33, color, arrow=True)
    s.text(710, y+42, score_label, 25, color)
s.text(550, 350, 'k = 2 noise samples • q(c) ∝ frequency(c)⁰·⁷⁵ • scores are independent sigmoids', 22, MUTED, anchor='middle')
s.save('negative_sampling')

# A separate worked 2D step. Use copies to make simultaneous gradients explicit.
e0 = np.array([1., 0.]); o0 = np.array([[0., 1.], [0., -1.]])
y = np.array([1., 0.]); eta = .2
before_scores = o0 @ e0
errors = 1/(1+np.exp(-before_scores))-y
e1 = e0-eta*(errors@o0)
o1 = o0-eta*errors[:, None]*e0
after_scores = o1@e1
before_loss = np.logaddexp(0, -before_scores[0])+np.logaddexp(0, before_scores[1])
after_loss = np.logaddexp(0, -after_scores[0])+np.logaddexp(0, after_scores[1])
s = Diagram(430, 'A simultaneous SGD update with one negative')
s.text(262, 36, 'Before', 28, weight=700, anchor='middle')
s.text(821, 36, 'After: learning rate 0.2', 28, weight=700, anchor='middle')
for y0, label, before, after, color in [
        (91, 'Input apples', e0, e1, BLUE),
        (169, 'Output tasty (+)', o0[0], o1[0], TEAL),
        (247, 'Output furry (−)', o0[1], o1[1], ORANGE)]:
    s.text(25, y0+7, label, 24, color)
    s.box(267, y0-30, 230, 60, [f'({before[0]:.1f}, {before[1]:.1f})'], color, 24)
    s.line(518, y0, 611, y0, color, arrow=True)
    s.box(633, y0-30, 230, 60, [f'({after[0]:.1f}, {after[1]:.1f})'], color, 24)
s.text(972, 98, '∇ₑL: (0, −1)', 23, BLUE, anchor='middle')
s.text(972, 176, 'error: −0.5', 23, TEAL, anchor='middle')
s.text(972, 254, 'error: +0.5', 23, ORANGE, anchor='middle')
s.line(30, 305, 1065, 305)
s.text(30, 348, 'Pair scores', 25, weight=700)
s.text(337, 348, '(0.0, 0.0)', 25, anchor='middle')
s.line(510, 338, 610, 338, INK, arrow=True)
s.text(748, 348, f'({after_scores[0]:+.1f}, {after_scores[1]:+.1f})', 25, anchor='middle')
s.text(30, 398, 'Total loss', 25, weight=700)
s.text(337, 398, f'{before_loss:.3f}', 25, anchor='middle')
s.line(510, 388, 610, 388, INK, arrow=True)
s.text(748, 398, f'{after_loss:.3f}', 25, TEAL, weight=700, anchor='middle')
s.save('gradient_step')

fig = plt.figure(figsize=(11, 4.3), layout='constrained')
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
labels = ['apples', 'oranges', 'rabbits', 'hamsters']
for i,(title, values) in enumerate([('Before training', demo['initial_cosines']),
                                    ('After training', demo['trained_cosines'])]):
    ax = fig.add_subplot(grid[0, i])
    ax.imshow(values, vmin=-1, vmax=1, cmap='BrBG')
    ax.set_xticks(range(4), labels, rotation=55, ha='right', fontsize=12)
    ax.set_yticks(range(4), labels if i == 0 else [], fontsize=12)
    if i == 1:
        ax.set_yticks([])
    ax.set_title(title, fontsize=19, pad=15)
    for row in range(4):
        for col in range(4):
            v=values[row,col]
            ax.text(col, row, f'{v:.2f}', ha='center', va='center', fontsize=13,
                    color='white' if abs(v)>.55 else INK)
ax = fig.add_subplot(grid[0, 2])
ax.plot(demo['loss_history'], color=BLUE, lw=2)
ax.set_title('Expected SGNS loss', fontsize=18, pad=15)
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylim(1.35, 2.15)
ax.set_yticks([1.4, 1.7, 2.0]); ax.tick_params(labelsize=13)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color=PALE)
ax.annotate(f"{demo['loss_history'][-1]:.3f}", xy=(400,demo['loss_history'][-1]),
            xytext=(-44,16), textcoords='offset points', color=BLUE, fontsize=16)
fig.supxlabel('Input–input cosine similarities (same scale: −1 to +1)', fontsize=16)
save_plot(fig, 'training_results')

s = Diagram(320, 'Low-rank score matrix and the shifted-PMI connection')
s.box(25, 50, 180, 150, ['E', '|V| × d'], BLUE, 29)
s.text(235, 139, '×', 40, anchor='middle')
s.box(268, 50, 205, 150, ['Oᵀ', 'd × |V|'], TEAL, 29)
s.text(515, 139, '=', 38, anchor='middle')
s.box(560, 50, 225, 150, ['Score matrix', '|V| × |V|', 'rank ≤ d'], INK, 25)
s.line(804, 125, 867, 125, MUTED, arrow=True)
s.box(889, 50, 190, 150, ['Related to', 'PMI − log k'], ORANGE, 25)
s.text(550, 252, 'The connection is about the objective’s preferred dot products.', 26, anchor='middle')
s.text(550, 294, 'It does not initialize word2vec from a count matrix.', 24, MUTED, anchor='middle')
s.save('implicit_matrix')

s = Diagram(380, 'Continuous bag-of-words and skip-gram')
s.text(255, 40, 'CBOW', 30, BLUE, 700, 'middle')
s.text(825, 40, 'Skip-gram', 30, TEAL, 700, 'middle')
s.line(550, 55, 550, 340)
s.box(25, 80, 170, 60, ['peel'], BLUE)
s.box(300, 80, 170, 60, ['the'], BLUE)
s.line(110, 146, 197, 199, BLUE, arrow=True)
s.line(385, 146, 292, 199, BLUE, arrow=True)
s.box(133, 205, 240, 60, ['mean of vectors'], BLUE, 23)
s.line(253, 272, 253, 303, BLUE, arrow=True)
s.text(255, 341, 'predict apple', 27, BLUE, anchor='middle')
s.box(705, 80, 240, 60, ['word vector: apple'], TEAL)
s.line(825, 147, 825, 200, TEAL, arrow=True)
s.text(825, 236, 'separate predictions', 25, TEAL, anchor='middle')
s.line(785, 253, 686, 297, TEAL, arrow=True)
s.line(865, 253, 963, 297, TEAL, arrow=True)
s.box(600, 310, 170, 55, ['peel'], TEAL)
s.box(875, 310, 170, 55, ['the'], TEAL)
s.save('cbow_skipgram')

# The lecture keeps the original cosine question, with a larger plotting area.
query = np.array([1., 1.])
candidates = np.array([[3., 3.], [1., .4], [5., -1.], [-1., -1.]])
fig, ax = plt.subplots(figsize=(8.2, 4.8), layout='constrained')
for vector, label, color, offset in zip(
        candidates, ['A = (3, 3)', 'B = (1, 0.4)', 'C = (5, −1)', 'D = (−1, −1)'],
        [TEAL, BLUE, ORANGE, '#7c3c88'], [(8, 0), (13, -10), (-95, -29), (-75, -28)]):
    ax.annotate('', xy=vector, xytext=(0, 0), arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 2.8})
    ax.annotate(label, xy=vector, xytext=offset, textcoords='offset points', color=color, fontsize=22, weight='bold')
ax.annotate('', xy=query, xytext=(0, 0), arrowprops={'arrowstyle': '-|>', 'color': INK, 'lw': 3.2})
ax.annotate('q = (1, 1)', xy=query, xytext=(-105, 15), textcoords='offset points', color=INK, fontsize=22, weight='bold')
ax.set(xlim=(-2.3, 6.2), ylim=(-2, 3.7), aspect='equal')
ax.spines[['top','right','bottom','left']].set_visible(False)
ax.axhline(0,color=LINE,lw=1,zorder=0);ax.axvline(0,color=LINE,lw=1,zorder=0)
ax.set_xticks(range(-1,6));ax.set_yticks(range(-1,4))
ax.tick_params(labelsize=13,length=0);ax.grid(color=PALE,zorder=0)
save_plot(fig,'cosine_quiz')

# A feed-forward network diagram connects to the lecture's opening MLP figure.
s=Diagram(430, 'The skip-gram network and the word vector inside it')
s.text(200,35,'One-hot input',28,INK,700,'middle')
s.text(550,35,'Word embedding',28,BLUE,700,'middle')
s.text(925,35,'Context predictions',28,TEAL,700,'middle')
labels=['apple','orange','rabbit','⋮','peel','slice']
ys=[88,140,192,244,296,348]
hidden_ys=[130,218,306]
for i,y in enumerate(ys):
    if labels[i]=='⋮': continue
    for hy in hidden_ys:
        s.line(258,y,530,hy,LINE,1.4)
        s.line(570,hy,882,y,LINE,1.4)
# The active one-hot coordinate selects the corresponding learned weights.
for hy in hidden_ys: s.line(258,ys[0],530,hy,BLUE,2.7)
for i,y in enumerate(ys):
    if labels[i]=='⋮':
        s.text(240,y+7,'⋮',31,MUTED,anchor='middle')
        s.text(900,y+7,'⋮',31,MUTED,anchor='middle')
        continue
    s.text(197,y+8,labels[i],26,INK,anchor='end')
    fill=BLUE if i==0 else 'white'
    s.parts.append(f'<circle cx="240" cy="{y}" r="18" fill="{fill}" stroke="{BLUE}" stroke-width="2"/>')
    s.text(240,y+7,'1' if i==0 else '0',21,'white' if i==0 else MUTED,anchor='middle')
    s.parts.append(f'<circle cx="900" cy="{y}" r="18" fill="white" stroke="{TEAL}" stroke-width="2"/>')
    s.text(934,y+8,labels[i],26,TEAL if labels[i]=='peel' else INK,weight=700 if labels[i]=='peel' else 400)
for i,y in enumerate(hidden_ys):
    s.parts.append(f'<circle cx="550" cy="{y}" r="20" fill="white" stroke="{BLUE}" stroke-width="2.5"/>')
    s.text(550,y+7,f'h{i+1}',19,BLUE,anchor='middle')
s.text(382,393,'input weights',25,BLUE,anchor='middle')
s.text(550,423,'linear projection',23,MUTED,anchor='middle')
s.text(725,393,'output weights',25,TEAL,anchor='middle')
s.text(925,423,'softmax',23,MUTED,anchor='middle')
s.save('network')

# Training overview: a forward prediction and a backward weight update.
s=Diagram(370, 'Initialize random word vectors, predict context and learn from the error')
s.text(355,37,'Small random values',26,BLUE,700,'middle')
s.text(355,68,'at initialization',24,BLUE,anchor='middle')
s.text(950,37,'Observed context',25,TEAL,700,'middle')
s.text(950,73,'peel',30,TEAL,anchor='middle')
s.text(90,177,'apple',32,INK,700,'middle')
s.line(160,167,258,167,INK,arrow=True)
for y,val in [(103,'0.02'),(148,'−0.01'),(193,'0.03')]:
    s.rect(280,y,150,42,fill='#eaf0f9',stroke='white',radius=0)
    s.text(355,y+29,val,27,BLUE,anchor='middle')
s.text(355,272,'word vector',26,BLUE,anchor='middle')
s.line(445,167,535,167,BLUE,arrow=True)
s.box(550,123,245,87,['Context prediction'],TEAL,25)
s.line(810,167,865,167,TEAL,arrow=True)
s.box(880,123,175,87,['Prediction','error'],INK,24)
s.line(950,87,950,115,TEAL,arrow=True)
# Error is propagated to both sets of weights before an update.
s.line(968,221,968,306,ORANGE,2.5)
s.line(968,306,355,306,ORANGE,2.5)
s.line(355,306,355,281,ORANGE,2.5,arrow=True)
s.line(672,306,672,223,ORANGE,2.5,arrow=True)
s.text(683,355,'Backpropagate error and update the weights',26,ORANGE,anchor='middle')
s.save('training')

print(json.dumps({'figures':len(list(OUT.glob('*.svg'))),
                  'notes_final_loss':demo['loss_history'][-1]},indent=2))
