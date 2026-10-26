"""Produce an offline reading/print view from the same manuscript sources.

This is a small converter for the LaTeX subset used in this manuscript, not a
general LaTeX compiler. Equations are rendered by the pinned local KaTeX build.
"""
from pathlib import Path
import base64, hashlib, html, json, re, struct, subprocess, importlib.util

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diagram_spec',ROOT/'tools/render_figures.py')
figmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(figmod)
FIGURES={f['label']:f for f in figmod.FIGURES}
SOURCE=(ROOT/'roborag-draft.tex').read_text()
BODY=SOURCE.split(r'\maketitle',1)[1].rsplit(r'\end{document}',1)[0]
BODY=re.sub(r'(?m)(?<!\\)%.*$','',BODY)
maths=[]; protected=[]; labels={}; cites={}; captions=[]; display_equation_number=0
figure_assets=[]

def balanced(s,pos):
    assert s[pos]=='{',s[pos:pos+40]
    depth=1;i=pos+1
    while i<len(s) and depth:
        if s[i]=='{' and (i==0 or s[i-1]!='\\'):depth+=1
        if s[i]=='}' and (i==0 or s[i-1]!='\\'):depth-=1
        i+=1
    return s[pos+1:i-1],i

def command_arg(s,name):
    m=re.search(r'\\'+re.escape(name)+r'\s*\{',s)
    return balanced(s,m.end()-1)[0] if m else ''

def keep(s):
    protected.append(s);return f'ZZZBLOCK{len(protected)-1}ZZZ'

def mask_math(s):
    pat=r'\\begin\{equation\*?\}[\s\S]*?\\end\{equation\*?\}|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<!\\)\$[^$]+?(?<!\\)\$'
    def repl(m):
        global display_equation_number
        raw=m.group();display=raw.startswith(r'\begin') or raw.startswith(r'\[')
        if raw.startswith(r'\begin'):
            tex=re.sub(r'\\(?:begin|end)\{equation\*?\}','',raw)
        elif raw.startswith('$'):tex=raw[1:-1]
        else:tex=raw[2:-2]
        tex=re.sub(r'\\label\{[^}]+\}','',tex).strip().replace(r'\begin{split}',r'\begin{aligned}').replace(r'\end{split}',r'\end{aligned}')
        if raw.startswith(r'\begin{equation}'):
            display_equation_number+=1
            tex+=r'\tag{'+str(display_equation_number)+'}'
        maths.append({'tex':tex,'display':display});return f'ZZZMATH{len(maths)-1}ZZZ'
    return re.sub(pat,repl,s)

sec=0;sub=0;eq=0;fig=0;tab=0
for m in re.finditer(r'\\(section|subsection|label)\{([^}]+)\}|\\begin\{(equation\*?|figure\*?|table\*?)\}',BODY):
    cmd,arg,env=m.groups()
    if cmd=='section':sec+=1;sub=0;current=str(sec)
    elif cmd=='subsection':sub+=1;current=f'{sec}.{sub}'
    elif env and env.startswith('equation'):eq+=1;current=str(eq)
    elif env and env.startswith('figure'):fig+=1;current=str(fig)
    elif env and env.startswith('table'):tab+=1;current=str(tab)
    elif cmd=='label':labels[arg]=current
for i,key in enumerate(re.findall(r'\\bibitem\{([^}]+)\}',BODY),1):cites[key]=i

def inline(s):
    s=mask_math(s)
    out=[];i=0
    tags={'textbf':'strong','emph':'em','textit':'em','texttt':'code','underline':'u'}
    while i<len(s):
        if s[i]=='\\':
            m=re.match(r'\\([A-Za-z]+\*?|.)',s[i:])
            if not m:out.append('\\');i+=1;continue
            cmd=m.group(1);i+=len(m.group());j=i
            while j<len(s) and s[j].isspace():j+=1
            arg=None
            if j<len(s) and s[j]=='{':arg,end=balanced(s,j)
            if cmd in tags and arg is not None:
                out.append(f'<{tags[cmd]}>'+inline(arg)+f'</{tags[cmd]}>');i=end
            elif cmd in ('ref','eqref') and arg is not None:
                number=labels.get(arg,'?');text='('+number+')' if cmd=='eqref' else number
                out.append(f'<a href="#{html.escape(arg)}">{text}</a>');i=end
            elif cmd=='cite' and arg is not None:
                out.append('['+', '.join(f'<a href="#bib-{html.escape(k.strip())}">{cites.get(k.strip(),"?")}</a>' for k in arg.split(','))+']');i=end
            elif cmd=='url' and arg is not None:
                out.append(f'<a href="{html.escape(arg)}">{html.escape(arg)}</a>');i=end
            elif cmd=='label' and arg is not None:i=end
            elif cmd in ('small','footnotesize','scriptsize','normalsize','centering','raggedright','noindent','arraybackslash'):pass
            elif cmd in ('%','&','_','#','$','{','}'):
                out.append(html.escape(cmd))
            elif cmd=='\\':out.append('<br>')
            elif cmd in ('quad','qquad',',',';',':',' '):out.append(' ')
            elif cmd=='LaTeX':out.append('LaTeX')
            elif cmd in ('textendash','textemdash'):out.append('—')
            else:
                out.append(html.escape('\\'+cmd))
        elif s[i] in '{}':i+=1
        else:
            ch=s[i];out.append('&nbsp;' if ch=='~' else html.escape(ch));i+=1
    return ''.join(out).replace('``','“').replace("''",'”').replace('---','—').replace('--','–')

def svg_diagram(f):
    W,H=f['width'],f['height']
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(f["name"])}">',
         '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#40505c"/></marker></defs>']
    for name,x,y,w,h,label,style in f['nodes']:
        fill='#eaf1f6' if style=='accent' else '#f7f7f7'
        out.append(f'<rect x="{x}" y="{H-y-h}" width="{w}" height="{h}" fill="{fill}" stroke="#40505c" stroke-width="1.3"/>')
        lines=label.split('\n');size=15 if w>=250 else 14
        for i,line in enumerate(lines):
            yy=H-(y+h/2+(len(lines)-1)*18/2-i*18-5)
            out.append(f'<text x="{x+w/2}" y="{yy}" text-anchor="middle" font-family="Arial,sans-serif" font-size="{size}" fill="#17202a">{html.escape(line)}</text>')
    for points,label,labelpos,dash in f['edges']:
        points=' '.join(f'{x},{H-y}' for x,y in points)
        ds=' stroke-dasharray="3 4"' if dash=='dotted' else (' stroke-dasharray="7 4"' if dash else '')
        out.append(f'<polyline points="{points}" fill="none" stroke="#40505c" stroke-width="1.7" marker-end="url(#arrow)"{ds}/>')
        if label:
            x,y=labelpos
            out.append(f'<text x="{x}" y="{H-y}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#17202a">{html.escape(label)}</text>')
    for x,y,t,size in f.get('texts',[]):
        out.append(f'<text x="{x}" y="{H-y}" font-family="Arial,sans-serif" font-size="{size}" fill="#17202a">{html.escape(t)}</text>')
    out.append('</svg>')
    result=''.join(out);(ROOT/'figures'/(f['name']+'.svg')).write_text(result)
    return result

def figure_html(raw):
    label=command_arg(raw,'label');f=FIGURES[label]
    caption=command_arg(raw,'caption')
    number=labels[label]
    filename=command_arg(raw,'RoboRAGFigure')
    expected=f['name']+'-nature.png'
    if filename and filename!=expected:
        raise ValueError('Unexpected polished figure asset for '+label+': '+filename)
    asset=ROOT/'figures/nature'/filename if filename else None
    if asset and asset.exists():
        data=asset.read_bytes()
        if data[:8]!=b'\x89PNG\r\n\x1a\n' or data[12:16]!=b'IHDR':
            raise ValueError('Polished figure is not a PNG: '+str(asset))
        width,height=struct.unpack('>II',data[16:24])
        if width<=0 or height<=0:
            raise ValueError('Invalid polished figure dimensions: '+str(asset))
        digest=hashlib.sha256(data).hexdigest()
        encoded=base64.b64encode(data).decode('ascii')
        drawing=(f'<img class="polished-figure" src="data:image/png;base64,{encoded}" '
                 f'width="{width}" height="{height}" '
                 f'alt="RoboRAG Figure {number}" aria-describedby="{label}-caption" '
                 f'data-asset="{html.escape(str(asset.relative_to(ROOT)))}" '
                 f'data-sha256="{digest}">')
        figure_assets.append({'label':label,'kind':'generated_png',
                              'path':str(asset.relative_to(ROOT)),'sha256':digest,
                              'width':width,'height':height})
    else:
        drawing=svg_diagram(f)
        figure_assets.append({'label':label,'kind':'embedded_svg',
                              'requested_asset':filename or None})
    result=f'<figure id="{label}">{drawing}<figcaption id="{label}-caption"><b>Figure {number}.</b> {inline(caption)}</figcaption></figure>'
    captions.append(('figure',number,result))
    return keep(result)

def table_html(raw):
    label=command_arg(raw,'label');caption=command_arg(raw,'caption')
    m=re.search(r'\\begin\{(tabularx|tabular)\}',raw)
    if not m:raise ValueError('No tabular in '+label)
    start=m.end()
    for _ in range(2 if m.group(1)=='tabularx' else 1):
        while raw[start].isspace():start+=1
        _,start=balanced(raw,start)
    body=raw[start:raw.index(r'\end{'+m.group(1)+'}',start)]
    body=mask_math(body)
    body=re.sub(r'\\(?:toprule|midrule|bottomrule|hline|addlinespace)(?:\[[^]]*\])?','',body)
    rows=[]
    for row in re.split(r'\\\\(?:\[[^]]*\])?',body):
        if not row.strip():continue
        cells=re.split(r'(?<!\\)&',row)
        rows.append('<tr>'+''.join('<td>'+inline(cell.strip())+'</td>' for cell in cells)+'</tr>')
    number=labels.get(label,str(len(captions)+1))
    result=f'<figure class="table-figure" id="{label}"><figcaption><b>Table {number}.</b> {inline(caption)}</figcaption><table>'+''.join(rows)+'</table></figure>'
    captions.append(('table',number,result));return keep(result)

def bib_html(raw):
    raw=re.sub(r'\\begin\{thebibliography\}\{[^}]*\}|\\end\{thebibliography\}','',raw)
    chunks=re.split(r'\\bibitem\{([^}]+)\}',raw)
    rows=[]
    for i in range(1,len(chunks),2):
        key=chunks[i];content=' '.join(chunks[i+1].split())
        rows.append(f'<li id="bib-{key}" value="{cites[key]}">{inline(content)}</li>')
    return keep('<section class="references"><h2>References</h2><ol>'+''.join(rows)+'</ol></section>')

BODY=re.sub(r'\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}',lambda m:figure_html(m.group()),BODY)
BODY=re.sub(r'\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}',lambda m:table_html(m.group()),BODY)
BODY=re.sub(r'\\begin\{thebibliography\}[\s\S]*?\\end\{thebibliography\}',lambda m:bib_html(m.group()),BODY)

def convert(s):
    s=mask_math(s)
    sec=0;sub=0
    def heading(m):
        nonlocal sec,sub
        kind,title=m.groups()
        if kind=='section':sec+=1;sub=0;num=str(sec);tag='h2'
        elif kind=='subsection':sub+=1;num=f'{sec}.{sub}';tag='h3'
        else:num='';tag='h4'
        return '\n\n'+keep(f'<{tag}>{num+". " if num else ""}{inline(title)}</{tag}>')+'\n\n'
    s=re.sub(r'\\(section|subsection|paragraph)\{([^}]+)\}',heading,s)
    s=re.sub(r'\\label\{([^}]+)\}',lambda m:keep(f'<span id="{m.group(1)}"></span>'),s)
    s=s.replace(r'\begin{abstract}',keep('<section class="abstract"><h2>Abstract</h2>')).replace(r'\end{abstract}',keep('</section>'))
    for env,tag in [('enumerate','ol'),('itemize','ul')]:
        pat=r'\\begin\{'+env+r'\}(?:\[[^]]*\])?([\s\S]*?)\\end\{'+env+r'\}'
        def list_repl(m,tag=tag):
            items=re.split(r'\\item(?:\[[^]]*\])?',m.group(1))[1:]
            return '\n\n'+keep('<'+tag+'>'+''.join('<li>'+inline(x.strip())+'</li>' for x in items)+'</'+tag+'>')+'\n\n'
        s=re.sub(pat,list_repl,s)
    s=re.sub(r'\\(?:FloatBarrier|clearpage|newpage)\b','',s)
    blocks=[]
    for para in re.split(r'\n\s*\n',s):
        para=' '.join(para.split())
        if not para:continue
        if re.fullmatch(r'(?:ZZZBLOCK\d+ZZZ\s*)+',para):blocks.append(para)
        else:blocks.append('<p>'+inline(para)+'</p>')
    result='\n'.join(blocks)
    for _ in range(5):
        result=re.sub(r'ZZZBLOCK(\d+)ZZZ',lambda m:protected[int(m.group(1))],result)
    return result

content=convert(BODY)
# Keep section anchors on headings so an empty inline anchor cannot satisfy
# break-after:avoid while the actual opening paragraph moves to another page.
content=re.sub(r'<(h[234])>(.*?)</\1>\s*<span id="([^"]+)"></span>',
               lambda m:f'<{m[1]} id="{m[3]}">{m[2]}</{m[1]}>',content)
review_content='\n'.join(x[2] for x in captions)
result=subprocess.run(['node',str(ROOT/'tools/render_math.cjs')],input=json.dumps(maths),text=True,capture_output=True,check=True)
rendered=json.loads(result.stdout)
errors=[]
def math_replace(m):
    n=int(m.group(1));r=rendered[n]
    if r['error']:errors.append((n,r['error']));return '<code>'+html.escape(maths[n]['tex'])+'</code>'
    return r['html']
content=re.sub(r'ZZZMATH(\d+)ZZZ',math_replace,content)
review_content=re.sub(r'ZZZMATH(\d+)ZZZ',math_replace,review_content)

CSS='''
@page { size:letter; margin:.62in .7in .62in; @bottom-center { content:counter(page); font-size:9pt; } }
*{box-sizing:border-box}body{margin:0;background:#e9ecef;color:#111;font-family:"Times New Roman",Times,serif;font-size:10pt;line-height:1.22}
.sheet{max-width:1000px;margin:28px auto;background:white;padding:52px 64px}header{text-align:center;margin-bottom:24px}h1{font-size:22pt;line-height:1.1;margin:0 0 12px}header p{font-size:10pt;margin:5px 0;text-align:center}
article{column-count:2;column-gap:.32in}p{margin:0 0 .6em;text-align:justify;orphans:3;widows:3}h2,h3,h4{break-inside:avoid}h2{font-size:13pt;margin:1em 0 .45em;break-after:avoid}h3{font-size:11pt;margin:.8em 0 .35em;break-after:avoid}h4{font-size:10pt;margin:.7em 0 .25em;break-after:avoid}
.abstract{font-style:italic}.abstract h2{text-align:center;font-size:12pt;margin-top:0}figure{column-span:all;break-inside:avoid;margin:16px 0 20px}figure svg,figure img.polished-figure{display:block;width:100%;height:auto}figcaption{font-size:8.8pt;line-height:1.2;margin-top:6px;text-align:justify}figure.table-figure figcaption{margin:0 0 7px}
table{border-collapse:collapse;width:100%;font-size:9pt;line-height:1.18;border-top:1.5px solid;border-bottom:1.5px solid}td{padding:6px 5px;vertical-align:top}tr:first-child{font-weight:bold;border-bottom:1px solid}tr+tr{border-top:.4px solid #ddd}
code{font-family:"Courier New",monospace;font-size:.88em;overflow-wrap:anywhere}a{color:inherit;text-decoration:none}.references{font-size:8.5pt}.references ol{padding-left:22px}.references li{margin:0 0 8px;overflow-wrap:anywhere}ol,ul{padding-left:19px;margin:6px 0 9px}li{margin-bottom:5px}
.katex{font-size:1em}.katex-display{margin:.65em 0;overflow:visible;font-size:9.2pt}.review article{column-count:1}.review figure{break-after:page;margin-top:0}.review header{margin-bottom:18px}
@media print{body{background:white}.sheet{margin:0;padding:0;max-width:none}header{margin-bottom:18px}h1{font-size:20pt}a{color:inherit}figure{margin:12px 0 16px}.review figure{break-after:page}}
@media screen and (max-width:750px){.sheet{padding:24px 20px;margin:0}article{column-count:1}}
'''
def wrap(body,title,review=False):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><link rel="stylesheet" href="assets/katex/katex.min.css"><style>'+CSS+'</style></head><body'+(' class="review"' if review else '')+'><main class="sheet"><header><h1>'+html.escape(title)+'</h1><p>English research design draft · October 23, 2026</p></header><article>'+body+'</article></main></body></html>'
(ROOT/'roborag-draft.html').write_text(wrap(content,'RoboRAG: Interaction-Aware Retrieval-Augmented Manipulation'))
(ROOT/'figures-and-captions.html').write_text(wrap(review_content,'RoboRAG: Figures and Captions',True))
(ROOT/'reader-validation.json').write_text(json.dumps({'math_fragments':len(maths),'math_errors':errors,'references':len(cites),'figures':len(FIGURES),'tables':sum(x[0]=='table' for x in captions),'figure_assets':figure_assets},indent=2))
print('Reader built:',len(maths),'math fragments;',len(errors),'math errors;',len(captions),'figures/tables.')
if errors:print(errors)
