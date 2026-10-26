"""Render manuscript diagrams from one specification to standalone PDF and TikZ.

The self-contained manuscript embeds TikZ; PDFs serve isolated visual review.
"""
from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

FIGURES = [
 dict(name='01-overview', label='fig:overview', width=1000, height=420,
 caption=r'Proposed RoboRAG workflow. Training episodes are segmented and checked to form a versioned bank $\mathcal{B}$ of experience graphs $G$. A new goal, current observation and backend capabilities produce a draft plan $P$. Retrieved graphs or subgraphs supply phase structure, conditions and recovery evidence; alignment and validation produce an adapted plan. A supervisor follows approved phases through a local executor, while the planner handles structural revisions. Actual receipts and observations extend the execution graph $X$. The three graphs share a schema, but history, intent and observed execution retain distinct status. The training bank stays fixed during held-out evaluation; the dashed return path is used only by a separately evaluated curation process. This is a method diagram, not an experimental result.',
 nodes=[
 ('episodes',20,305,160,68,'TRAIN episodes\nobservations, actions\nand feedback','plain'),
 ('annotate',220,305,170,68,'Segment + annotate\ncheck evidence\nand graph schema','accent'),
 ('bank',430,305,170,68,'Graph bank B\nwhole graphs +\nphase subgraphs','accent'),
 ('input',20,170,160,75,'NEW task\ngoal + observation\n+ capabilities','plain'),
 ('draft',220,170,170,75,'Draft graph plan\nroles, conditions\nunknowns explicit','plain'),
 ('adapt',430,170,170,75,'Retrieve + align\nedit + validate\nadapted plan P','accent'),
 ('follow',650,170,155,75,'Follow plan\nsupervisor +\nlocal executor','accent'),
 ('world',850,170,130,75,'Backend\nrobot /\nsimulator','plain'),
 ('history',650,35,155,75,'Execution graph X\nactual events\n+ evidence','plain'),
 ('revise',430,35,170,75,'Planner revision\nonly when needed\nnew plan version','plain'),
 ('curate',850,305,130,68,'Separate curation\nnew bank version\nafter evaluation','plain')],
 edges=[
 ([(180,339),(220,339)],'',None,False),
 ([(390,339),(430,339)],'',None,False),
 ([(180,207),(220,207)],'',None,False),
 ([(390,207),(430,207)],'',None,False),
 ([(515,305),(515,245)],'retrieve',(532,275),False),
 ([(600,207),(650,207)],'',None,False),
 ([(805,207),(850,207)],'',None,False),
 ([(915,170),(915,72),(805,72)],'receipts + observations',(919,121),False),
 ([(650,72),(600,72)],'',None,False),
 ([(515,110),(515,170)],'replan',(535,140),False),
 ([(728,110),(728,170)],'status',(745,140),False),
 ([(915,245),(915,305)],'',None,True),
 ([(850,339),(600,339)],'reviewed experience',(724,355),True)],
 texts=[(20,393,'A. BUILD THE EXPERIENCE BANK',15),(20,270,'B. PLAN AND EXECUTE AN UNSEEN TASK',15)]),

 dict(name='02-graph-semantics',label='fig:semantics',width=1000,height=420,
 caption=r'An interaction graph records phase instances and the evidence connecting them. In this illustrative recorded episode, a grasp attempt fails; reobservation and a second attempt precede placement. Each box is a phase instance, not an object. Solid arrows show the observed temporal order. The dashed recovery link connects the failed attempt to reobservation; the second grasp uses the newly grounded target. The dotted dependency states that placement requires verified holding evidence from the second grasp. Alternative success and failure guards are transition choices (OR), whereas prerequisites must all hold (AND). The node record separates the requested effect from its observed outcome. A plan may contain bounded retry logic; recorded attempts have distinct IDs and times. Times and evidence IDs in this example illustrate the schema and are not measured experimental data.',
 nodes=[
 ('g1',20,250,165,80,'v1: grasp attempt\n[0, 2] s\noutcome: failed','plain'),
 ('observe',240,250,165,80,'v2: reobserve\n[2, 3] s\nnew object pose','plain'),
 ('g2',460,250,165,80,'v3: grasp attempt\n[3, 5] s\nholding verified','accent'),
 ('place',760,250,205,80,'v4: place\n[5, 7] s\nplacement verified','plain'),
 ('record',20,15,600,155,'Example record for v1\nrole: grasp(item)     binding: item = cup_7\nprecondition: reachable(item)\nintended effect: holding(item)\nobserved outcome: failed; evidence = frames_18_21\nsource: TRAIN episode_12; interval = [0, 2] s','plain'),
 ('guard',680,15,285,155,'Ready to start a phase?\nAn eligible branch is selected (OR)\nAND every prerequisite is true\nAND resources are available\nUnknown condition: observe first\nRequested effect is not completion','plain')],
 edges=[
 ([(185,290),(240,290)],'',None,False),
 ([(405,290),(460,290)],'',None,False),
 ([(625,290),(760,290)],'temporal',(665,305),False),
 ([(103,330),(103,375),(322,375),(322,330)],'recovery: failure to reobservation',(280,389),True),
 ([(543,250),(543,215),(862,215),(862,250)],'dependency: holding(item) = true',(701,199),'dotted')],
 texts=[(20,408,'ILLUSTRATIVE OBSERVED GRAPH: FOUR DISTINCT PHASE INSTANCES',14)]),

 dict(name='03-plan-adaptation',label='fig:adaptation',width=1000,height=460,
 caption=r'Illustrative retrieval and adaptation for a new task: place a cup in a closed drawer. Two training subgraphs provide object transfer and drawer opening; the target composition is held out from the experience bank. Object roles are rebound to current observations, the open-drawer phase is inserted as a prerequisite, and current geometry supplies the action parameters. The adapted plan also incorporates an authorized recovery branch from historical failure evidence. The supervisor checks predicates and follows the approved branch; structural changes go to the planner. A failed grasp appends a failed event to $X$ and creates a distinct retry instance, rather than relabeling the failed action as successful. The example explains graph composition and revision; it makes no claim that this task has been implemented or solved.',
 nodes=[
 ('source1',20,335,280,82,'TRAIN subgraph A: transfer\ngrasp -> move -> place\nitem / target roles\nfailure -> observe -> retry','plain'),
 ('source2',360,335,280,82,'TRAIN subgraph B: open\napproach(handle) -> pull(drawer)\nverify open(drawer)','plain'),
 ('query',720,335,260,82,'UNSEEN composition\nput cup in closed drawer\ncurrent cup / handle / drawer','plain'),
 ('align',200,210,600,65,'ALIGN + EDIT + VALIDATE\nitem = current cup; target = current drawer\ninsert opening prerequisite; ground geometry; check capabilities','accent'),
 ('open',20,90,185,66,'Open drawer\nverify open(drawer)\nrelease handle','plain'),
 ('grasp',270,90,185,66,'Grasp cup\nverify holding(cup)','accent'),
 ('move',535,90,185,66,'Move cup\ncheck live geometry','plain'),
 ('place',795,90,185,66,'Place cup\nverify in(cup, drawer)','plain'),
 ('retry',270,0,185,58,'Reobserve + retry\napproved budget only\nnew attempt ID','plain')],
 edges=[
 ([(160,335),(160,243),(200,243)],'',None,False),
 ([(500,335),(500,275)],'',None,False),
 ([(850,335),(850,243),(800,243)],'',None,False),
 ([(500,210),(500,182),(113,182),(113,156)],'validated plan',(265,194),False),
 ([(205,123),(270,123)],'',None,False),
 ([(455,123),(535,123)],'holding',(491,139),False),
 ([(720,123),(795,123)],'',None,False),
 ([(362,90),(362,58)],'failed',(402,73),True),
 ([(455,29),(490,29),(490,103),(455,103)],'retry',(520,60),True)],
 texts=[(20,440,'RETRIEVE COMPATIBLE SUBGRAPHS; ADAPT THEIR CONDITIONS AND STRUCTURE',14)]),

 dict(name='04-platform-comparison',label='fig:platform',width=1000,height=420,
 caption=r'Proposed interface comparison on RoboDojo, LIBERO and a real robot. Within each backend, all three modes use the same planner, graph bank, observations, task instances and local controller. Mode A uses a backend-specific native driver. Mode B routes calls through the proposed common capability, observation, action and receipt contract. Mode C exposes exactly the same contract and adapter through MCP. B versus A tests semantic normalization and reuse; C versus B tests the incremental effect of protocol discovery and transport, including its overhead. The three backend names denote planned validation targets, not completed integrations. A shared contract does not make backend physics or capabilities identical.',
 nodes=[
 ('shared',210,320,580,66,'FIXED WITHIN EACH BACKEND\nplanner + graph bank + observations + task instances + local controller','plain'),
 ('a',20,185,280,83,'A. Backend-specific native driver\nbackend-specific tool signatures\nlocal/native calls','plain'),
 ('b',360,185,280,83,'B. Shared contract, direct calls\ncommon semantics + adapter\nlocal/native calls','accent'),
 ('c',700,185,280,83,'C. Shared contract, MCP\nsame semantics + same adapter\nMCP discovery + transport','accent'),
 ('backends',210,65,580,66,'EVALUATE EACH BACKEND SEPARATELY\nRoboDojo  |  LIBERO  |  real robot\nbackend-specific capabilities and physical execution','plain')],
 edges=[
 ([(500,320),(500,294),(160,294),(160,268)],'',None,False),
 ([(500,320),(500,268)],'',None,False),
 ([(500,294),(840,294),(840,268)],'',None,False),
 ([(160,185),(160,98),(210,98)],'',None,False),
 ([(500,185),(500,131)],'',None,False),
 ([(840,185),(840,98),(790,98)],'',None,False)],
 texts=[(20,405,'SEPARATE INTERFACE REUSE FROM THE EFFECT OF MCP',15),
 (20,32,'B - A: reuse / semantic consistency     C - B: discovery / transport overhead',15),
 (20,10,'Report task success, conformance, application edits and latency for each backend.',13)])
]

def escape(s):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:
        s=s.replace(a,b)
    return s.replace('->',r'$\rightarrow$')

def render(f):
    W,H=f['width'],f['height']
    c=canvas.Canvas(str(OUT/(f['name']+'.pdf')),pagesize=(W,H))
    c.setTitle(f['name'])
    tex=[r'\begin{figure*}[t]',r'\centering',r'\resizebox{\textwidth}{!}{%',r'\begin{tikzpicture}[x=1pt,y=1pt,>=stealth]',f'\\path[use as bounding box] (0,0) rectangle ({W},{H});']
    for name,x,y,w,h,label,style in f['nodes']:
        fill='#EAF1F6' if style=='accent' else '#F7F7F7'
        c.setStrokeColor(HexColor('#40505C'));c.setLineWidth(1.3)
        c.setFillColor(HexColor(fill));c.rect(x,y,w,h,fill=1,stroke=1)
        lines=label.split('\n');size=15 if w>=250 else 14
        leading=18
        c.setFillColor(HexColor('#17202A'));c.setFont('Helvetica',size)
        for i,line in enumerate(lines):
            c.drawCentredString(x+w/2,y+h/2+(len(lines)-1)*leading/2-i*leading-5,line)
        tlabel=r'\\'.join(escape(s) for s in lines)
        tfill='blue!7' if style=='accent' else 'black!3'
        tex.append(f'\\node[draw=black!65,fill={tfill},line width=1.3pt,minimum width={w}pt,minimum height={h}pt,text width={w-12}pt,inner sep=4pt,align=center,font=\\sffamily\\fontsize{{{size}}}{{18}}\\selectfont] ({name}) at ({x+w/2},{y+h/2}) {{{tlabel}}};')
    for points,label,labelpos,dash in f['edges']:
        c.setStrokeColor(HexColor('#40505C'));c.setFillColor(HexColor('#40505C'));c.setLineWidth(1.7)
        c.setDash([3,4] if dash=='dotted' else ([7,4] if dash else []))
        p=c.beginPath();p.moveTo(*points[0])
        for pt in points[1:]:p.lineTo(*pt)
        c.drawPath(p)
        x,y=points[-1];px,py=points[-2];ang=math.atan2(y-py,x-px)
        p=c.beginPath();p.moveTo(x,y)
        p.lineTo(x-9*math.cos(ang)+4*math.sin(ang),y-9*math.sin(ang)-4*math.cos(ang))
        p.lineTo(x-9*math.cos(ang)-4*math.sin(ang),y-9*math.sin(ang)+4*math.cos(ang));p.close()
        c.setDash([]);c.drawPath(p,fill=1,stroke=0)
        tstyle=',dotted' if dash=='dotted' else (',dashed' if dash else '')
        tex.append(r'\draw[->,line width=1.7pt'+tstyle+'] '+' -- '.join(f'({a},{b})' for a,b in points)+';')
        if label:
            lx,ly=labelpos;c.setFillColor(HexColor('#17202A'));c.setFont('Helvetica',12)
            c.drawCentredString(lx,ly,label)
            tex.append(f'\\node[fill=white,inner sep=2pt,font=\\sffamily\\fontsize{{12}}{{14}}\\selectfont] at ({lx},{ly+4}) {{{escape(label)}}};')
    for x,y,t,size in f.get('texts',[]):
        c.setFillColor(HexColor('#17202A'));c.setFont('Helvetica',size);c.drawString(x,y,t)
        tex.append(f'\\node[anchor=base west,font=\\sffamily\\fontsize{{{size}}}{{18}}\\selectfont] at ({x},{y}) {{{escape(t)}}};')
    c.showPage();c.save()
    tex.extend([r'\end{tikzpicture}}',r'\caption{'+f['caption']+'}',r'\label{'+f['label']+'}',r'\end{figure*}'])
    (OUT/(f['name']+'.tex')).write_text('\n'.join(tex)+'\n')
    (OUT/(f['name']+'.caption.txt')).write_text(f['caption']+'\n')

if __name__=='__main__':
    for f in FIGURES:render(f)
    print('Rendered',len(FIGURES),'diagram PDFs and matching embedded TikZ sources.')
