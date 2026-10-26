/* Local graph-bank browser: library -> graph -> phase subtree. No build step. */
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {bank:null, task:'', graph:null, phase:null, selected:null, panel:'content', scale:1, playing:null, load:0, searchLoad:0};
const titleFor = task => state.bank?.graphs.find(g => g.task === task)?.task_name || task;
const frameURL = (id, frame, camera='cam_head') => `/api/frame?id=${encodeURIComponent(id)}&frame=${frame}&camera=${encodeURIComponent(camera)}`;
const childrenOf = phase => state.graph.nodes.filter(n => n.parent === phase.id);
const phasesOf = graph => graph.nodes.filter(n => n.kind === 'phase');
const time = (frame, frequency) => (frame/frequency).toFixed(2);

async function api(path){const response = await fetch(path);if(!response.ok){let error;try{error=await response.json()}catch{}throw new Error(error?.error || `请求失败 (${response.status})`)}return response.json()}
function showError(error){$('errorNotice').hidden=false;$('errorNotice').textContent=error.message || String(error)}
function clearError(){$('errorNotice').hidden=true}
function stopPlayback(){if(state.playing)clearInterval(state.playing);state.playing=null;$('play').textContent='▶';$('play').setAttribute('aria-label','播放片段')}
function showView(view){stopPlayback();for(const name of ['library','graph','search','retrieve'])$(name+'View').hidden=name!==view;window.scrollTo(0,0);clearError()}

async function loadBank(){
  state.bank=await api('/api/bank');const {counts,graphs}=state.bank;
  $('navCount').textContent=counts.graphs;
  const groups=new Map();for(const graph of graphs){if(!groups.has(graph.task))groups.set(graph.task,{name:graph.task_name,count:0});groups.get(graph.task).count++}
  $('taskNav').innerHTML=[...groups].map(([task,g])=>`<button class="nav-item ${state.task===task?'active':''}" data-task="${escapeHTML(task)}"><span class="task-dot"></span>${escapeHTML(g.name)}<span class="nav-count">${g.count}</span></button>`).join('');
  const current=$('searchTask').value;$('searchTask').innerHTML='<option value="">全部任务</option>'+[...groups].map(([task,g])=>`<option value="${escapeHTML(task)}">${escapeHTML(g.name)}</option>`).join('');$('searchTask').value=current;
  $('mobileTask').innerHTML=$('searchTask').innerHTML;$('mobileTask').value=state.task;
  $('collectionStamp').textContent=`${counts.graphs} 张经验图已入库`;
  renderLibrary();
}
function miniature(count){
  const shown=Math.min(10,count), width=240, step=shown>1?220/(shown-1):0;
  const line=shown>1?`<path d="M10 12H230" stroke="#b4c9b9" fill="none"/>`:'';
  const dots=Array.from({length:shown},(_,i)=>{const x=10+i*step;return `<path d="M${x} 15V25h6" stroke="#c6d5c8" fill="none"/><circle cx="${x}" cy="12" r="${i===0?4:3}" fill="${i===0?'#337d56':'#91b39a'}"/><circle cx="${x+7}" cy="25" r="2" fill="#d3ded1"/>`}).join('');
  return `<svg viewBox="0 0 ${width} 36" aria-hidden="true">${line}${dots}</svg>`;
}
function renderLibrary(){
  if(!state.bank)return;const all=state.bank.graphs;let graphs=all.filter(g=>!state.task||g.task===state.task);
  graphs.sort($('sort').value==='phases'?(a,b)=>b.phase_count-a.phase_count:(a,b)=>a.task.localeCompare(b.task)||a.episode-b.episode);
  $('allGraphs').classList.toggle('active',!state.task);document.querySelectorAll('[data-task]').forEach(b=>b.classList.toggle('active',b.dataset.task===state.task));
  $('mobileTask').value=state.task;
  $('libraryTitle').innerHTML=escapeHTML(state.task?titleFor(state.task):'交互经验图库')+'<span class="heading-dot">.</span>';
  $('libraryDescription').textContent=state.task?'同一任务的多条示范，各自保留完整的交互阶段与局部经验。':'浏览示范中提取的任务图，沿着阶段查看经验如何连接。';
  const stats=[[graphs.length,'经验图'],[graphs.reduce((v,g)=>v+g.phase_count,0),'交互阶段'],[graphs.reduce((v,g)=>v+g.node_count,0),'图节点'],[new Set(graphs.map(g=>g.task)).size,'任务集合']];
  $('stats').innerHTML=stats.map(([v,label])=>`<div class="stat"><strong>${v}</strong><span>${label}</span></div>`).join('');
  $('graphCount').textContent=`${graphs.length} GRAPHS`;
  $('graphGrid').innerHTML=graphs.map(g=>`<a class="graph-card" href="#/graph/${encodeURIComponent(g.id)}" aria-label="查看 ${escapeHTML(g.task_name)} 示范 ${g.episode} 的经验图"><div class="card-image"><img loading="lazy" src="${frameURL(g.id,0)}" alt="${escapeHTML(g.task_name)}原始示范初始画面"><span class="episode-badge">EPISODE ${String(g.episode).padStart(2,'0')}</span></div><div class="card-content"><div class="card-task"><h3>${escapeHTML(g.task_name)}</h3><span aria-hidden="true">↗</span></div><p class="card-title">${escapeHTML(g.title)}</p><div class="mini-graph">${miniature(g.phase_count)}</div><div class="card-footer"><span><strong>${g.phase_count}</strong> 阶段 · <strong>${g.node_count}</strong> 节点</span><span>${time(g.frames,g.frequency)} s 示范</span></div></div></a>`).join('') || '<div class="empty-state"><strong>正在等待第一张经验图</strong>构图完成后，点击右上角刷新即可查看。</div>';
}
function library(task=''){state.task=task;showView('library');$('crumb').textContent=task?' / '+titleFor(task):'';renderLibrary()}
async function openGraph(id,nodeId){
  showView('graph');const request=++state.load;
  $('graphTitle').textContent='读取经验图…';$('graphSummary').textContent='';$('graphCanvas').innerHTML='';
  try{const graph=await api('/api/graph?id='+encodeURIComponent(id));if(request!==state.load)return;state.graph=graph;state.panel='content';
    $('crumb').textContent=' / '+graph.task_name+' / '+String(graph.episode).padStart(2,'0');
    $('graphTitle').textContent=graph.title;$('graphSummary').textContent=graph.summary;
    $('exportGraph').href='/api/export?id='+encodeURIComponent(graph.id);
    const phases=phasesOf(graph);
    $('graphMeta').innerHTML=`<span class="badge">${escapeHTML(graph.task_name)}</span><span>Episode ${String(graph.episode).padStart(2,'0')}</span><span>${phases.length} 阶段 / ${graph.nodes.length} 节点 / ${graph.edges.length} 条边</span><span>经验记录 · 时间顺序</span><span>Codex 自动提取</span>`;
    $('graphSize').textContent=`${graph.nodes.length} nodes · ${graph.edges.length} edges`;
    state.scale=1;
    const wanted=graph.nodes.find(n=>n.id===nodeId)||phases[0];state.phase=wanted.kind==='phase'?wanted:graph.nodes.find(n=>n.id===wanted.parent);state.selected=wanted.id;
    drawGraph();selectNode(wanted.id);$('graphViewport').scrollTop=0;
    if(nodeId)revealSelected();
  }catch(error){if(request===state.load)showError(error)}
}
function drawGraph(){
  if(!state.graph)return;const phases=phasesOf(state.graph), expanded=$('showDetails').checked;
  const width=expanded?Math.max(620,Math.min(900,$('graphViewport').clientWidth-4)):430, row=expanded?166:125, height=phases.length*row+54;
  const coordinates=new Map();let buttons='';
  for(const phase of phases){const x=expanded?24:68,y=34+phase.order*row,w=expanded?Math.min(278,width*.4):278,h=78;coordinates.set(phase.id,{x,y,w,h});
    buttons+=`<button class="phase-node ${phase.id===state.phase?.id?'selected':''}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px" data-node="${phase.id}" aria-label="阶段 ${phase.order+1}：${escapeHTML(phase.title)}"><span class="phase-number">${String(phase.order+1).padStart(2,'0')}</span><span class="phase-copy"><strong>${escapeHTML(phase.title)}</strong><small>${time(phase.start_frame,state.graph.source.frequency)}–${time(phase.end_frame,state.graph.source.frequency)} s · ${phase.end_frame-phase.start_frame} 帧</small></span></button>`;
    if(expanded)childrenOf(phase).forEach((child,i)=>{const box={x:w+80,y:y-13+i*47,w:width-w-104,h:37};coordinates.set(child.id,box);buttons+=`<button class="detail-node kind-${child.kind} ${child.id===state.selected?'selected':''}" data-node="${child.id}" style="left:${box.x}px;top:${box.y}px;width:${box.w}px;height:${box.h}px" aria-label="${escapeHTML(phase.title)}：${escapeHTML(child.title)}"><span class="detail-kind">${escapeHTML(child.title)}</span><span class="detail-text">${escapeHTML(child.text)}</span></button>`});
  }
  let paths='';for(const edge of state.graph.edges){const a=coordinates.get(edge.source),b=coordinates.get(edge.target);if(!a||!b)continue;
    if(edge.kind==='next'){const x=a.x+a.w/2;paths+=`<path d="M${x} ${a.y+a.h}V${b.y-5}" fill="none" stroke="#9fbda7" stroke-width="1.4" marker-end="url(#arrow)"/>`;}
    else{const x=a.x+a.w,y=a.y+a.h/2,end=b.y+b.h/2;paths+=`<path d="M${x} ${y}C${x+32} ${y},${b.x-32} ${end},${b.x} ${end}" fill="none" stroke="#d1ded1" stroke-width="1"/>`;}
  }
  $('graphCanvas').style.width=width+'px';$('graphCanvas').style.height=height+'px';
  $('graphCanvas').innerHTML=`<svg width="${width}" height="${height}" aria-hidden="true"><defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#89a991" stroke-width="1"/></marker></defs>${paths}</svg>${buttons}`;
  state.graphWidth=width;state.graphHeight=height;applyZoom();
}
function applyZoom(){if(!state.graphWidth)return;$('graphCanvas').style.transform=`scale(${state.scale})`;$('graphExtent').style.width=state.graphWidth*state.scale+'px';$('graphExtent').style.height=state.graphHeight*state.scale+'px';$('zoomValue').textContent=Math.round(state.scale*100)+'%'}
function revealSelected(){const selected=$('graphCanvas').querySelector(`[data-node="${state.phase?.id}"]`);if(selected){const row=$('showDetails').checked?166:125;$('graphViewport').scrollTop=Math.max(0,(state.phase.order*row+18)*state.scale-80)}}
function selectNode(id){
  const node=state.graph.nodes.find(n=>n.id===id);if(!node)return;stopPlayback();state.selected=id;
  state.phase=node.kind==='phase'?node:state.graph.nodes.find(n=>n.id===node.parent);
  const phase=state.phase,children=childrenOf(phase),frequency=state.graph.source.frequency;
  $('nodeTitle').textContent=phase.title;$('nodeSummary').textContent=phase.summary;$('nodeOrder').textContent=`${String(phase.order+1).padStart(2,'0')} / ${String(phasesOf(state.graph).length).padStart(2,'0')}`;
  $('nodeObjects').innerHTML=phase.objects.map(o=>`<span>${escapeHTML(o)}</span>`).join('');
  $('guidanceContent').innerHTML=children.map(child=>`<section class="guidance-block kind-${child.kind} ${child.id===id?'focused':''}"><div class="guidance-label"><i></i>${escapeHTML(child.title)}</div><p>${escapeHTML(child.text)}</p></section>`).join('');
  $('nodeSource').textContent=`来源：${state.graph.task_name} · Episode ${state.graph.episode}\n帧 ${phase.start_frame}–${phase.end_frame-1} · ${frequency} Hz · 阶段边界为自动估计`;
  $('nodeJson').textContent=JSON.stringify({graph_id:state.graph.id,phase,guidance:children,edges:state.graph.edges.filter(e=>e.source===phase.id)},null,2);
  document.querySelectorAll('.phase-node').forEach(button=>button.classList.toggle('selected',button.dataset.node===phase.id));
  document.querySelectorAll('.detail-node').forEach(button=>button.classList.toggle('selected',button.dataset.node===id));
  $('scrub').min=phase.start_frame;$('scrub').max=phase.end_frame-1;$('scrub').value=phase.start_frame;
  $('traceDescription').textContent=`${phase.end_frame-phase.start_frame} 个采样点 · ${time(phase.end_frame-phase.start_frame,frequency)} 秒。包含双臂动作目标、末端位姿、关节与夹爪状态。`;
  $('downloadTrace').href=`/api/trace?id=${encodeURIComponent(state.graph.id)}&node=${phase.id}&download=1`;
  const cameraNames={cam_head:'头部视角',cam_left_wrist:'左腕视角',cam_right_wrist:'右腕视角'},previousCamera=$('camera').value;
  $('camera').innerHTML=state.graph.source.cameras.map(camera=>`<option value="${camera}">${cameraNames[camera]||camera}</option>`).join('');
  if(state.graph.source.cameras.includes(previousCamera))$('camera').value=previousCamera;
  updateKeyframes();if(node.kind!=='phase')setPanel('content');else setPanel(state.panel);
  if(state.panel==='media')updateFrame();
}
function setPanel(panel){state.panel=panel;for(const name of ['content','media','json'])$(name+'Panel').hidden=name!==panel;document.querySelectorAll('[data-panel]').forEach(b=>b.setAttribute('aria-selected',b.dataset.panel===panel));if(panel==='media')updateFrame();else stopPlayback()}
function updateFrame(){if(!state.graph||!state.phase)return;const frame=Number($('scrub').value);$('frameError').hidden=true;$('frame').src=frameURL(state.graph.id,frame,$('camera').value);$('frameLabel').textContent=`FRAME ${frame}`;$('timeLabel').textContent=time(frame,state.graph.source.frequency)+' s'}
function updateKeyframes(){if(!state.phase)return;const labels=['开始','中间','结束'];$('keyframes').innerHTML=state.phase.keyframes.map((f,i)=>`<button data-frame="${f}" aria-label="查看第 ${f} 帧"><img loading="lazy" src="${frameURL(state.graph.id,f,$('camera').value)}" alt="第 ${f} 帧"><span>${labels[i]||'关键帧'} · ${f}</span></button>`).join('')}
async function search(query){
  if(!query.trim()){location.hash='#/';return;}showView('search');$('searchInput').value=query;$('crumb').textContent=' / 经验检索';
  const request=++state.searchLoad;$('searchSummary').textContent=`正在库内检索「${query}」…`;
  try{const data=await api('/api/search?q='+encodeURIComponent(query)+'&task='+encodeURIComponent($('searchTask').value));if(request!==state.searchLoad)return;
    $('searchSummary').textContent=`「${query}」展示 ${data.results.length} 个相关阶段。点击结果，在原图中查看上下文。`;
    $('searchResults').innerHTML=data.results.map(result=>`<a class="search-result" href="#/graph/${encodeURIComponent(result.graph_id)}/${result.node_id}"><img loading="lazy" src="${frameURL(result.graph_id,result.start_frame)}" alt="检索阶段的开始画面"><div class="result-body"><div class="result-path">${escapeHTML(result.task_name)} / ${escapeHTML(result.graph_id)} / ${result.node_id}</div><h3>${escapeHTML(result.title)}</h3><p>${escapeHTML(result.summary)}</p></div><div class="result-aside"><small>BM25 ${result.score.toFixed(2)}</small>在图中查看 ↗</div></a>`).join('')||'<div class="empty-state"><strong>没有找到相关阶段</strong>试试库中常见的操作词：<br><button class="search-chip" data-query="抓取">抓取</button><button class="search-chip" data-query="对齐">对齐</button><button class="search-chip" data-query="释放">释放</button></div>';
  }catch(error){if(request===state.searchLoad)showError(error)}
}
function route(){
  const raw=location.hash.slice(1)||'/';const parts=raw.split('/').filter(Boolean).map(decodeURIComponent);
  ++state.load;++state.searchLoad;
  if(parts[0]==='graph'&&parts[1])return openGraph(parts[1],parts[2]);
  if(parts[0]==='retrieve')return openRetrieval();
  if(parts[0]==='search')return search(parts.slice(1).join('/'));
  if(parts[0]==='task')return library(parts[1]||'');
  library();
}
$('allGraphs').onclick=()=>{location.hash='#/'};
$('taskNav').onclick=e=>{const button=e.target.closest('[data-task]');if(button)location.hash='#/task/'+encodeURIComponent(button.dataset.task)};
$('sort').onchange=renderLibrary;
$('mobileTask').onchange=()=>{location.hash=$('mobileTask').value?'#/task/'+encodeURIComponent($('mobileTask').value):'#/'};
$('refresh').onclick=async()=>{try{$('refresh').classList.add('loading');await loadBank();await route()}catch(error){showError(error)}finally{$('refresh').classList.remove('loading')}};
$('searchForm').onsubmit=e=>{e.preventDefault();const query=$('searchInput').value.trim();const hash='#/search/'+encodeURIComponent(query);if(location.hash===hash)search(query);else location.hash=hash};
$('searchTask').onchange=()=>search($('searchInput').value);
$('backToLibrary').onclick=()=>{location.hash=state.task?'#/task/'+encodeURIComponent(state.task):'#/'};
$('backFromSearch').onclick=()=>{location.hash='#/'};
$('graphCanvas').onclick=e=>{const button=e.target.closest('[data-node]');if(button){selectNode(button.dataset.node);if(window.matchMedia('(max-width:980px)').matches)document.querySelector('.inspector').scrollIntoView({block:'start'})}};
$('backToGraph').onclick=()=>document.querySelector('.graph-pane').scrollIntoView({block:'start'});
$('showDetails').onchange=()=>{drawGraph();revealSelected()};
$('zoomIn').onclick=()=>{state.scale=Math.min(1.6,state.scale+.15);applyZoom()};
$('zoomOut').onclick=()=>{state.scale=Math.max(.25,state.scale-.15);applyZoom()};
$('fitGraph').onclick=()=>{state.scale=Math.min(1,($('graphViewport').clientWidth-16)/state.graphWidth,($('graphViewport').clientHeight-20)/state.graphHeight);applyZoom();$('graphViewport').scrollTop=0;$('graphViewport').scrollLeft=0};
document.querySelectorAll('[data-panel]').forEach(button=>button.onclick=()=>setPanel(button.dataset.panel));
$('camera').onchange=()=>{updateFrame();updateKeyframes()};$('scrub').oninput=()=>{stopPlayback();updateFrame()};
$('keyframes').onclick=e=>{const button=e.target.closest('[data-frame]');if(button){stopPlayback();$('scrub').value=button.dataset.frame;updateFrame()}};
$('play').onclick=()=>{if(state.playing){stopPlayback();return;}$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','暂停片段');state.playing=setInterval(()=>{let frame=Number($('scrub').value)+Math.max(1,Math.round(state.graph.source.frequency/8));if(frame>Number($('scrub').max))frame=Number($('scrub').min);$('scrub').value=frame;updateFrame()},125)};
$('frame').onerror=()=>{$('frameError').hidden=false};
$('copyJson').onclick=async()=>{try{await navigator.clipboard.writeText($('nodeJson').textContent);$('copyJson').textContent='已复制';setTimeout(()=>{$('copyJson').textContent='复制 JSON'},1500)}catch{showError(new Error('当前浏览器不支持复制，请选中 JSON 内容复制。'))}};
$('searchResults').onclick=e=>{const button=e.target.closest('[data-query]');if(button)location.hash='#/search/'+encodeURIComponent(button.dataset.query)};
document.addEventListener('keydown',e=>{if(e.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){e.preventDefault();$('searchInput').focus()}});
window.addEventListener('hashchange',route);
loadBank().then(route).catch(showError);
