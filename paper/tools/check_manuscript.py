"""Check source consistency and offline reader output without a TeX install."""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
import argparse,base64,hashlib,json,re

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--require-polished-figures',action='store_true',
                    help='Require all four reviewed PNGs in the published reader.')
args=parser.parse_args()

ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'roborag-draft.tex').read_text()
s=re.sub(r'(?m)(?<!\\)%.*$','',s)
labels=re.findall(r'\\label\{([^}]+)\}',s)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',s)
keys=set(re.findall(r'\\bibitem\{([^}]+)\}',s))
cites=[k.strip() for x in re.findall(r'\\cite\{([^}]+)\}',s) for k in x.split(',')]
stack=[];env_errors=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',s):
    side,env=m.groups()
    if side=='begin':stack.append(env)
    elif not stack or stack[-1]!=env:env_errors.append((m.start(),'mismatched end',env,stack[-1:]))
    else:stack.pop()
depth=0;braces=[]
for m in re.finditer(r'(?<!\\)[{}]',s):
    depth+=1 if m.group()=='{' else -1
    if depth<0:braces.append(m.start())
reader=json.loads((ROOT/'reader-validation.json').read_text())
h=(ROOT/'roborag-draft.html').read_text()

# The only external graphics are optional replacements in a two-path wrapper.
# Their editable TikZ alternatives remain embedded in the manuscript source.
expected_assets={
 'fig:overview':'01-overview-nature.png',
 'fig:semantics':'02-graph-semantics-nature.png',
 'fig:adaptation':'03-plan-adaptation-nature.png',
 'fig:platform':'04-platform-comparison-nature.png',
}
figure_asset_errors=[]
graphic_paths=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s)
allowed_graphic_paths=['figures/nature/#1','paper/figures/nature/#1']
if graphic_paths!=allowed_graphic_paths:
 figure_asset_errors.append('Unexpected includegraphics paths: '+repr(graphic_paths))
for raw in re.findall(r'\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}',s):
 label=re.search(r'\\label\{([^}]+)\}',raw)
 asset=re.search(r'\\RoboRAGFigure\{([^}]+)\}',raw)
 if not label or label.group(1) not in expected_assets:
  figure_asset_errors.append('Unknown or missing figure label in source')
 elif not asset or asset.group(1)!=expected_assets[label.group(1)]:
  figure_asset_errors.append('Missing or mismatched image wrapper: '+label.group(1))
 if r'\begin{tikzpicture}' not in raw:
  figure_asset_errors.append('A figure is missing its embedded editable fallback')

class ImageCollector(HTMLParser):
 def __init__(self):
  super().__init__();self.images=[]
 def handle_starttag(self,tag,attributes):
  if tag=='img':self.images.append(dict(attributes))

collector=ImageCollector();collector.feed(h)
image_records=reader.get('figure_assets',[])
record_labels=[x.get('label') for x in image_records]
if Counter(record_labels)!=Counter(expected_assets.keys()):
 figure_asset_errors.append('Reader figure asset records do not cover the four figure labels')
embedded_images={}
for item in collector.images:
 path=item.get('data-asset')
 if not path:
  figure_asset_errors.append('Reader contains an untracked image');continue
 if path in embedded_images:
  figure_asset_errors.append('Duplicate embedded image: '+path)
 embedded_images[path]=item

generated_count=0
for record in image_records:
 label=record.get('label');kind=record.get('kind')
 if kind=='embedded_svg':
  if args.require_polished_figures:
   figure_asset_errors.append('Polished PNG required but fallback rendered: '+str(label))
  continue
 if kind!='generated_png':
  figure_asset_errors.append('Unknown figure rendering kind: '+str(kind));continue
 generated_count+=1
 expected='figures/nature/'+expected_assets.get(label,'UNKNOWN')
 path=record.get('path')
 if path!=expected:
  figure_asset_errors.append('Unexpected generated image path: '+str(path));continue
 asset=ROOT/path
 if not asset.is_file():
  figure_asset_errors.append('Published PNG no longer exists: '+path);continue
 digest=hashlib.sha256(asset.read_bytes()).hexdigest()
 if digest!=record.get('sha256'):
  figure_asset_errors.append('Published PNG changed since reader build: '+path)
 image=embedded_images.get(path,{})
 if image.get('data-sha256')!=digest:
  figure_asset_errors.append('Reader image digest mismatch: '+path)
 prefix='data:image/png;base64,'
 uri=image.get('src','')
 if not uri.startswith(prefix):
  figure_asset_errors.append('Reader image is not embedded offline: '+path);continue
 try:
  data=base64.b64decode(uri[len(prefix):],validate=True)
  if hashlib.sha256(data).hexdigest()!=digest:
   figure_asset_errors.append('Embedded image bytes differ from the PNG: '+path)
 except ValueError:
  figure_asset_errors.append('Invalid embedded image data: '+path)
 if (image.get('width'),image.get('height'))!=(str(record.get('width')),str(record.get('height'))):
  figure_asset_errors.append('Reader image dimensions differ from the asset record: '+path)
if len(embedded_images)!=generated_count:
 figure_asset_errors.append('Embedded image count differs from generated figure records')

report={
 'missing_sections':[],
 'duplicate_labels':[k for k,n in Counter(labels).items() if n>1],
 'unresolved_references':sorted(set(refs)-set(labels)),
 'unresolved_citations':sorted(set(cites)-keys),
 'environment_errors':env_errors,
 'unclosed_environments':stack,
 'brace_balance':depth,
 'premature_closing_braces':braces,
 'external_input_commands':re.findall(r'\\(?:input|include)\b',s),
 'figure_asset_errors':figure_asset_errors,
 'polished_figures_required':args.require_polished_figures,
 'polished_figure_count':generated_count,
 'embedded_tikz_fallbacks':s.count(r'\begin{tikzpicture}'),
 'unrendered_placeholders':re.findall(r'ZZZ(?:MATH|BLOCK)\d+ZZZ',h),
 'reader_math_errors':reader['math_errors'],
 'figures':reader['figures'],'tables':reader['tables'],'references':len(keys),
 'native_latex_compilation':'unverified: app compiler cannot start because its sandbox executable is missing',
 'readable_pdf':'rendered locally from the same source text via offline HTML/KaTeX and Chromium'
}
for p in ['00-abstract','01-introduction','02-related','03-formulation','04a-construction','04b-use','05-system','06a-experiments','06b-evidence','07-discussion']:
 if not (ROOT/'sections'/(p+'.tex')).exists():report['missing_sections'].append(p)
(ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
bad=[k for k in ['missing_sections','duplicate_labels','unresolved_references','unresolved_citations','environment_errors','unclosed_environments','brace_balance','premature_closing_braces','external_input_commands','figure_asset_errors','unrendered_placeholders','reader_math_errors'] if report[k]]
print('Consistency errors:',bad or 'none')
print('Figures / tables / verified reference entries:',report['figures'],report['tables'],report['references'])
if bad:raise SystemExit(1)
