"""Assemble the manuscript, with optional polished figures and embedded fallbacks."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = r'''% Self-contained English first draft, 2026-10-23.
% CVPR-like two-column layout; not a submission-compliance build.
% All section text, editable TikZ figure fallbacks and references are embedded.
% When present, the reviewed PNGs in figures/nature/ replace the TikZ diagrams.
% Copying this file alone preserves a compilable version with the original diagrams.
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage[letterpaper,textwidth=6.875in,textheight=8.875in,columnsep=0.3125in,top=0.75in]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,tabularx,multirow}
\usepackage{graphicx,xcolor,tikz}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{placeins}
\usepackage[breaklinks,hidelinks]{hyperref}
\urlstyle{same}
\setlength{\emergencystretch}{1.5em}
\setlist{nosep,leftmargin=*}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\renewcommand{\topfraction}{0.95}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.8}
\renewcommand{\dblfloatpagefraction}{0.8}
% Support compiling from either paper/ or the repository root. A missing PNG
% intentionally selects the embedded editable diagram, not an empty figure.
\newcommand{\RoboRAGFigure}[2]{%
  \IfFileExists{figures/nature/#1}{%
    \includegraphics[width=\textwidth]{figures/nature/#1}%
  }{%
    \IfFileExists{paper/figures/nature/#1}{%
      \includegraphics[width=\textwidth]{paper/figures/nature/#1}%
    }{#2}%
  }%
}
\title{RoboRAG: Interaction-Aware Retrieval-Augmented Manipulation}
\author{Research design draft}
\date{October 23, 2026}
\begin{document}
\maketitle
'''

PARTS = [
    'sections/00-abstract.tex',
    'sections/01-introduction.tex',
    'figures/01-overview.tex',
    'sections/02-related.tex',
    'sections/03-formulation.tex',
    'METHOD_HEADER',
    'sections/04a-construction.tex',
    'figures/02-graph-semantics.tex',
    'sections/04b-use.tex',
    'figures/03-plan-adaptation.tex',
    'sections/05-system.tex',
    'figures/04-platform-comparison.tex',
    'sections/06a-experiments.tex',
    'sections/06b-evidence.tex',
    'sections/07-discussion.tex',
    'references.tex',
]

FIGURE_ASSETS = {
    'figures/01-overview.tex': '01-overview-nature.png',
    'figures/02-graph-semantics.tex': '02-graph-semantics-nature.png',
    'figures/03-plan-adaptation.tex': '03-plan-adaptation-nature.png',
    'figures/04-platform-comparison.tex': '04-platform-comparison-nature.png',
}

def with_polished_figure(source, filename):
    """Wrap only the drawing; keep the source caption and label unchanged."""
    start = source.index(r'\centering') + len(r'\centering')
    end = source.index(r'\caption{', start)
    drawing = source[start:end].strip()
    return (source[:start] + '\n' + r'\RoboRAGFigure{' + filename + '}{%\n'
            + drawing + '\n}\n' + source[end:])

def build():
    parts=[PREAMBLE]
    missing=[]
    for rel in PARTS:
        if rel=='METHOD_HEADER':
            parts.append(r'\section{Method}'+'\n'+r'\label{sec:method}'+'\n')
            continue
        p=ROOT/rel
        if not p.exists():
            missing.append(rel);continue
        source = p.read_text()
        if rel in FIGURE_ASSETS:
            source = with_polished_figure(source, FIGURE_ASSETS[rel])
        parts.append('\n% BEGIN '+rel+'\n'+source+'\n% END '+rel+'\n')
    parts.append('\n'+r'\end{document}'+'\n')
    output=''.join(parts)
    (ROOT/'roborag-draft.tex').write_text(output)
    report=[]
    for p in sorted((ROOT/'sections').glob('*.tex')):
        s=re.sub(r'%[^\n]*','',p.read_text())
        words=re.findall(r"\b[A-Za-z][A-Za-z'-]*\b",s)
        report.append(f'{p.name}: {len(words)} lexical tokens (includes LaTeX commands).')
    (ROOT/'word-counts.txt').write_text('\n'.join(report)+'\n')
    print('Built standalone manuscript:',ROOT/'roborag-draft.tex')
    print('Missing sections:',', '.join(missing) if missing else 'none')
    print('Characters:',len(output))

if __name__=='__main__':build()
