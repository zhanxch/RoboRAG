import {groupKey, summarize, totalTokens} from './metrics.js';

const $ = s => document.querySelector(s);
const h = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, digits=0) => v == null ? '—' : Number(v).toLocaleString('zh-CN', {maximumFractionDigits:digits});
const pct = v => v == null ? '—' : `${fmt(v,1)}%`;
const compact = v => v == null ? '—' : v >= 1e6 ? `${fmt(v/1e6,2)}M` : v >= 1e3 ? `${fmt(v/1e3,1)}k` : fmt(v);
const seconds = v => v == null ? '—' : v >= 60 ? `${fmt(v/60,1)} min` : `${fmt(v,1)} s`;
const localDate = () => new Date().toLocaleDateString('sv-SE');
const tracks = {rag:'经验 RAG 与 Graph',plan:'Plan Graph 调度',mcp:'MCP 与实验验证'};
const statusNames = {todo:'待开始',active:'进行中',review:'待验证',done:'已完成',blocked:'受阻'};
const runNames = {planned:'待执行',running:'执行中',completed:'已完成',interrupted:'已中断',error:'运行异常'};
const splits = {development:'开发集',validation:'验证集',test:'测试集',unknown:'未注明'};
const transports = {mcp:'MCP',legacy:'历史 HTTP / CLI',unknown:'未确认'};
const goals = {ai:'AI-friendly',cost:'Cost-efficient',sota:'SOTA'};
const noteKinds = {update:'进展',decision:'决策',finding:'发现',blocker:'阻塞'};
const pages = {overview:'研究总览',tasks:'研究路线',experiments:'实验台账',notes:'研究日志'};
let state, page='overview', filters={tasks:{query:'',track:'',status:'',view:'table'},experiments:{query:'',transport:'',split:'',group:''},notes:{query:'',kind:''}};
let editing=null, pendingDelete=null, busy=false, toastTimer;
const allRuns = () => [...state.experiments,...state.historical];
const badge = (value, names=statusNames) => `<span class="badge ${h(value)}"><i></i>${h(names[value] || value)}</span>`;
const goalTags = values => (values || []).map(g => `<span class="goal-tag ${h(g)}">${h(goals[g])}</span>`).join('');
const empty = (title,text) => `<div class="empty-state"><span>⌕</span><h3>${h(title)}</h3><p>${h(text)}</p></div>`;
const btn = (action,label,cls='button',extra='') => `<button class="${cls}" data-action="${action}" ${extra}>${label}</button>`;
const opts = (values,current,blank='全部') => (blank === null ? '' : `<option value="">${h(blank)}</option>`) + Object.entries(values).map(([v,n])=>`<option value="${h(v)}" ${v === current ? 'selected':''}>${h(n)}</option>`).join('');
function titleBlock(eyebrow,title,description,actions='') {return `<section class="page-heading"><div><div class="eyebrow">${h(eyebrow)}</div><h1>${h(title)}</h1><p>${h(description)}</p></div><div class="heading-actions">${actions}</div></section>`;}
function evidenceLinks(text) {return String(text || '').split('\n').filter(Boolean).map(path => /^(docs|results|experiments|src\/roborag|tests)\//.test(path) && !path.includes('..') ? `<a class="evidence-link" href="/api/evidence?path=${encodeURIComponent(path)}" target="_blank" rel="noopener">↗ ${h(path)}</a>` : `<span class="evidence-path">${h(path)}</span>`).join('') || '<span class="muted">尚未添加证据</span>';}
function dependencies(task) {return task.dependencies.map(id=>state.tasks.find(t=>t.id === id)).filter(Boolean);}
function unresolved(task) {return dependencies(task).filter(t=>t.status !== 'done');}
function taskButton(task, cls='text-button') {return `<button class="${cls}" data-action="edit-task" data-id="${h(task.id)}">${h(task.title)}</button>`;}
function metric(label,value,note,cls='') {return `<article class="metric ${cls}"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${note}</div></article>`;}

async function api(path,options={}) {
  const response = await fetch(path,{cache:'no-store',...options,headers:{'Content-Type':'application/json','X-RoboRAG-Client':'research-progress',...options.headers}});
  let data;
  try {data=await response.json();} catch {throw new Error('服务返回了无法读取的响应');}
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function toast(message) {$('#toast').textContent=message;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,4500);}
function connection(text,error=false) {$('#save-status').innerHTML=`<i class="live-dot ${error?'error':''}"></i>${h(text)}`;}
async function refresh(notify=false) {
  try {state=await api('/api/state');connection('已连接 · 本地保存');render();if(notify)toast('已读取最新研究记录');}
  catch(error) {connection('连接失败',true);if(!state)$('#content').innerHTML=empty('无法连接本地服务',error.message)+btn('refresh','重新连接');else toast(error.message);}
}
function render() {
  page=location.hash.slice(1) in pages ? location.hash.slice(1) : 'overview';
  $('#breadcrumb').textContent=pages[page];
  document.querySelectorAll('nav a').forEach(a=>{a.classList.toggle('active',a.dataset.page === page);if(a.dataset.page === page)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  $('#content').innerHTML=({overview:overviewPage,tasks:tasksPage,experiments:experimentsPage,notes:notesPage}[page])();
}
function overviewPage() {
  const done=state.tasks.filter(t=>t.status==='done').length, review=state.tasks.filter(t=>t.status==='review').length;
  const blocked=state.tasks.filter(t=>t.status==='blocked'), mcp=allRuns().filter(r=>r.transport==='mcp' && r.status==='completed');
  const focus=state.tasks.filter(t=>t.status!=='done' && t.status!=='blocked').sort((a,b)=>Number(b.status==='active')-Number(a.status==='active') || a.priority.localeCompare(b.priority) || unresolved(a).length-unresolved(b).length || b.updated_at.localeCompare(a.updated_at)).slice(0,4);
  const current=focus.find(t=>t.status==='active');
  return titleBlock('PROJECT / OVERVIEW','研究总览','从交互经验出发，让每一步研究都有清晰的下一步与可复核的证据。',btn('new-note','＋ 记录进展','button')+btn('new-task','＋ 新建任务','button primary')) +
  (current?`<section class="panel current-focus"><div class="panel-heading"><h2>当前推进</h2>${badge(current.status)}</div>${taskButton(current)}<p>${h(current.next_step || current.description)}</p><span class="micro">${h(current.owner || tracks[current.track])} · ${h(current.priority)} · 点击任务查看验收标准</span></section>`:'')+
  `<section class="research-brief"><div class="brief-copy"><span class="brief-tag"><i class="live-dot"></i>RoboRAG · 研究进行时</span><h2>交互经验驱动 RAG<br><span>在 MCP 下完成检索、规划与执行。</span></h2><p>用可调用的经验图连接智能体与机器人，验证接口友好性、执行成本与任务表现。</p><div class="brief-goals">${goalTags(['ai','cost','sota'])}</div></div><div class="research-map" aria-label="研究路线依赖示意，并非实际运行中的计划图"><div class="map-label">RESEARCH LOOP <span>研究路线示意</span></div><div class="map-row"><div><b>01</b><strong>交互经验</strong><small>观察 · 动作 · 反馈</small></div><span>→</span><div><b>02</b><strong>Experience Graph</strong><small>构图 · 索引 · 检索</small></div></div><div class="map-connector">↓</div><div class="map-row"><div><b>04</b><strong>执行与评测</strong><small>回执 · 成功 · 消耗</small></div><span>←</span><div><b>03</b><strong>Plan Graph</strong><small>依赖 · 调度 · 恢复</small></div></div><div class="map-protocol">MCP · tools / resources / traces <span>待贯通</span></div></div></section>`+
  `<section class="metrics-grid">${metric('研究任务已完成',`${done}<small> / ${state.tasks.length}</small>`,`按已完成任务计数，不代表研究效果`, 'accent')}${metric('实现待验证',review,`已有基础，等待验收或实验`) }${metric('MCP 完整回合',mcp.length,`历史 HTTP / CLI 不计入`) }${metric('显式阻塞',blocked.length,blocked.length?'需要实验或前置条件解除':'暂无显式阻塞')}</section>`+
  `<section class="section-title"><div><h2>三条研究主线</h2><p>把实现、验证与结论逐一对齐。</p></div><a href="#tasks" class="text-link">查看全部任务 ↗</a></section><section class="track-grid">${Object.entries(tracks).map(([key,name],index)=>{const tasks=state.tasks.filter(t=>t.track===key), n=tasks.filter(t=>t.status==='done').length;return `<article class="track-card"><div class="track-top"><span class="track-number">0${index+1}</span><span>${n} / ${tasks.length} 已完成</span></div><h3>${h(name)}</h3><p>${['经验单元、图结构与检索策略','计划依赖、执行回执与失败恢复','工具契约、消耗计量与公平对照'][index]}</p><div class="progress-track"><span style="width:${tasks.length?n/tasks.length*100:0}%"></span></div><div class="track-bottom"><span>${tasks.filter(t=>t.status==='review').length} 待验证 · ${tasks.filter(t=>t.status==='todo').length} 待开始</span>${btn('filter-track','查看路线 →','text-button',`data-id="${key}"`)}</div></article>`;}).join('')}</section>`+
  `<div class="overview-columns"><section class="panel"><div class="panel-heading"><h2>下一步优先推进</h2><span class="subtle-label">进行中优先 · 优先级 × 依赖</span></div>${focus.map(t=>`<div class="focus-row"><span class="priority ${h(t.priority)}">${h(t.priority)}</span><div>${taskButton(t)}<p>${h(t.next_step || t.acceptance)}</p><span class="micro">${unresolved(t).length?`${unresolved(t).length} 项前置待完成`:'前置就绪'} · ${h(tracks[t.track])}</span></div>${badge(t.status)}</div>`).join('') || empty('暂无待推进任务','可以创建下一阶段的研究任务。')}</section><section class="panel evidence-panel"><div class="panel-heading"><h2>证据与研究目标</h2><span class="subtle-label">验收边界</span></div>${Object.entries(goals).map(([id,name])=>{const linked=state.tasks.filter(t=>t.goals.includes(id));return `<div class="goal-check"><span class="goal-dot ${id}"></span><div><strong>${name}</strong><p>${{ai:'MCP 契约与端到端工具调用',cost:'同条件延迟与 token 消耗验证',sota:'公平基线、消融与多次重复'}[id]}</p><small>${linked.filter(t=>t.status==='done').length} / ${linked.length} 关联任务完成 · 研究结论需独立验收</small></div></div>`;}).join('')}<div class="callout">已导入 ${state.historical.length} 条历史实验记录。历史复现与 pilot 是参考证据；当前未据此认定 MCP 全链路或 SOTA 已达成。</div></section></div>`+
  `<section class="panel activity-panel"><div class="panel-heading"><h2>最近维护</h2><a href="#notes" class="text-link">研究日志 ↗</a></div>${state.activity.length?state.activity.slice(0,5).map(a=>`<div class="activity-row"><span class="activity-dot"></span><div><strong>${h(a.title)}</strong><span>${{created:'新增',updated:'更新',deleted:'删除'}[a.action]}${{tasks:'研究任务',experiments:'实验记录',notes:'研究日志'}[a.collection]}</span></div><time>${h(new Date(a.at).toLocaleString('zh-CN',{hour12:false}))}</time></div>`).join(''):'<div class="activity-empty">工作台已从仓库初始化。新增或编辑后，维护记录会显示在这里。</div>'}</section>`;
}
function tasksPage() {
  const f=filters.tasks, selected=state.tasks.filter(t=>(!f.track || t.track===f.track)&&(!f.status || t.status===f.status)&&`${t.id} ${t.title} ${t.description} ${t.owner} ${t.next_step}`.toLowerCase().includes(f.query.toLowerCase()));
  return titleBlock('RESEARCH / ROADMAP','研究路线','围绕三条主线管理任务，用验收标准与证据区分“做了”和“验证了”。',btn('new-task','＋ 新建任务','button primary'))+
  `<div class="filters"><label class="search"><span>⌕</span><input data-filter="tasks.query" aria-label="搜索任务" placeholder="搜索任务、负责人或下一步…" value="${h(f.query)}"></label><select data-filter="tasks.track" aria-label="研究主线">${opts(tracks,f.track,'全部主线')}</select><select data-filter="tasks.status" aria-label="任务状态">${opts(statusNames,f.status,'全部状态')}</select><div class="view-toggle">${btn('task-view','列表',f.view==='table'?'selected':'', 'data-id="table"')}${btn('task-view','看板',f.view==='board'?'selected':'','data-id="board"')}</div></div><div class="list-meta"><span>${selected.length} 项任务</span><span>完成比例按任务计数；待验证项不算完成。</span></div>`+
  (!selected.length?empty('没有匹配的任务','调整筛选条件，或新建研究任务。'):f.view==='board'?`<div class="kanban">${Object.entries(statusNames).map(([key,name])=>`<section class="kanban-column"><header>${badge(key)}<span>${selected.filter(t=>t.status===key).length}</span></header>${selected.filter(t=>t.status===key).map(t=>`<article class="kanban-card"><div class="card-meta"><span>${h(t.id)}</span><span class="priority ${h(t.priority)}">${h(t.priority)}</span></div>${taskButton(t)}<p>${h(t.next_step || t.description)}</p><div class="tag-row">${goalTags(t.goals)}</div><footer><span>${h(t.owner || '未分配')}</span><span>${t.due?h(t.due):'未设日期'}</span></footer></article>`).join('') || '<div class="kanban-empty">暂无任务</div>'}</section>`).join('')}</div>`:
  `<div class="table-wrap"><table class="task-table"><thead><tr><th>研究任务</th><th>状态</th><th>优先级</th><th>依赖 / 验收</th><th>负责人 / 日期</th><th></th></tr></thead><tbody>${selected.sort((a,b)=>Object.keys(tracks).indexOf(a.track)-Object.keys(tracks).indexOf(b.track)||a.priority.localeCompare(b.priority)||a.id.localeCompare(b.id)).map(t=>`<tr><td><div class="row-kicker">${h(t.id)} <span>· ${h(tracks[t.track])}</span></div>${taskButton(t)}<div class="tag-row">${goalTags(t.goals)}</div></td><td>${badge(t.status)}</td><td><span class="priority ${h(t.priority)}">${h(t.priority)}</span></td><td><span class="dependency ${unresolved(t).length?'waiting':''}">${unresolved(t).length?`${unresolved(t).length} 项前置待完成`:'前置就绪'}</span><p class="cell-note">${h(t.next_step || t.acceptance)}</p></td><td><span>${h(t.owner || '未分配')}</span><small class="${t.due && t.due < localDate() && t.status!=='done'?'overdue':''}">${h(t.due || '未设截止日')}</small></td><td>${btn('edit-task','↗','icon-button',`data-id="${h(t.id)}" aria-label="编辑 ${h(t.title)}"`)}</td></tr>`).join('')}</tbody></table></div>`);
}
function experimentsPage() {
  const f=filters.experiments;
  const candidates=allRuns().filter(r=>(!f.transport || r.transport===f.transport)&&(!f.split || r.split===f.split)&&`${r.title} ${r.task} ${r.method} ${r.protocol} ${r.model}`.toLowerCase().includes(f.query.toLowerCase()));
  const groups=new Map();candidates.forEach(r=>{const key=groupKey(r);if(!groups.has(key))groups.set(key,`${r.task} · ${r.condition} · ${r.protocol} · ${splits[r.split]} · ${r.model}/${r.effort} · ${transports[r.transport]}${r.bank_version?` · 库 ${r.bank_version.slice(0,8)}`:''}${r.code_version?` · 代码 ${r.code_version.slice(0,8)}`:''}${r.mcp_server?` · ${r.mcp_server}/${r.mcp_version}`:''}`);});
  if(f.group && !groups.has(f.group)) f.group='';
  const runs=candidates.filter(r=>!f.group || groupKey(r)===f.group), s=summarize(runs), comparable=f.group!=='';
  const methods=[...new Set(runs.map(r=>r.method))];
  return titleBlock('EVALUATION / LEDGER','实验台账','一条记录对应一次任务执行。保留失败与中断，按实验条件查看表现和消耗。',`<a class="button" href="/api/export?format=csv" download>↓ 导出 CSV</a>${btn('new-experiment','＋ 记录实验','button primary')}`)+
  `<div class="filters"><label class="search"><span>⌕</span><input data-filter="experiments.query" aria-label="搜索实验" placeholder="搜索任务、方法或评测协议…" value="${h(f.query)}"></label><select data-filter="experiments.transport" aria-label="通信协议">${opts(transports,f.transport,'全部通信协议')}</select><select data-filter="experiments.split" aria-label="数据划分">${opts(splits,f.split,'全部数据划分')}</select></div>`+
  `<div class="comparison-picker"><div><strong>同条件分组</strong><span>任务 / 场景 / 协议 / 模型 / 划分 / 库与代码版本 / MCP 配置</span></div><select data-filter="experiments.group" aria-label="同条件实验组"><option value="">全部记录 · 不跨组汇总性能</option>${[...groups].map(([key,label])=>`<option value="${h(key)}" ${key===f.group?'selected':''}>${h(label)}</option>`).join('')}</select></div>`+
  (comparable && methods.length>1 ? `<section class="panel method-panel"><div class="panel-heading"><h2>同条件方法对照</h2><span class="subtle-label">各方法分别统计，不合并成绩</span></div><div class="table-wrap"><table><thead><tr><th>方法</th><th>成功率</th><th>平均完成度</th><th>平均耗时</th><th>已知总 token</th></tr></thead><tbody>${methods.map(method=>{const m=summarize(runs.filter(r=>r.method===method));return `<tr><td>${h(method)}</td><td>${pct(m.successRate)}<small>${m.successes} / ${m.scored} 回合</small></td><td>${pct(m.completion)}<small>n = ${m.completionN}</small></td><td>${seconds(m.wall)}<small>n = ${m.wallN}</small></td><td>${m.lowerBound?'≥ ':''}${compact(m.tokens)}<small>${m.tokenN} / ${m.count} 有总用量</small></td></tr>`;}).join('')}</tbody></table></div></section>` : comparable?`<section class="metrics-grid">${metric('任务成功率',pct(s.successRate),`${s.successes} / ${s.scored} 个已知最终结果`)}${metric('平均任务完成度',pct(s.completion),`${s.completionN} 个有评分的完整回合`)}${metric('平均端到端耗时',seconds(s.wall),`${s.wallN} 个完整回合 · 墙钟时间`)}${metric('已知 token 消耗',`${s.lowerBound?'≥ ':''}${compact(s.tokens)}`,`${s.tokenN} / ${s.count} 有总用量 · 含中断消耗`)}</section>`:
  `<div class="comparison-empty"><span>◎</span><div><strong>选择一个实验组，查看成功率、完成度与平均耗时</strong><p>不同任务和实验设置分别报告；不把历史方法复现与 MCP 方案合并为一个成绩。</p></div><div class="record-count">${runs.length}<small>条记录</small></div></div>`)+
  (comparable?`<div class="callout compact-callout">分组只保证已记录字段一致。布局 / 种子配对、冻结模型快照、观测权限、预算和并发条件仍需核验；小样本结果不自动证明 SOTA。缓存 token 已包含在输入中。</div>`:'')+
  `<div class="list-meta"><span>${runs.length} 条记录 · ${s.completed} 个完整回合 · ${runs.filter(r=>r.readonly).length} 条历史来源</span><span>“—” 表示未知，不是零</span></div>`+
  (!runs.length?empty('还没有符合条件的实验','创建 MCP 实验记录，或调整筛选查看已有历史结果。'):
  `<div class="table-wrap"><table class="experiment-table"><thead><tr><th>实验 / 条件</th><th>协议</th><th>结果</th><th>完成度</th><th>耗时</th><th>Token</th><th></th></tr></thead><tbody>${runs.sort((a,b)=>b.date.localeCompare(a.date)||a.id.localeCompare(b.id)).map(r=>`<tr><td><button class="text-button" data-action="view-experiment" data-id="${h(r.id)}">${h(r.title)}</button><small>${h(r.method)} · ${h(splits[r.split])} · L${h(r.layout || '?')} / seed ${h(r.seed || '?')}</small><div class="micro">${h(r.protocol)}</div></td><td><span class="protocol-tag ${r.transport==='mcp'?'mcp':''}">${h(transports[r.transport])}</span><small>${r.readonly?'↳ 历史来源':'手动记录'}</small></td><td>${r.status==='completed'?(r.success===true?'<span class="badge done"><i></i>成功</span>':r.success===false?'<span class="badge blocked"><i></i>未成功</span>':'<span class="badge review">待评分</span>'):badge(r.status,runNames)}</td><td><strong>${pct(r.completion)}</strong></td><td class="numeric">${seconds(r.wall_seconds)}</td><td class="numeric" title="${h(fmt(totalTokens(r)))}">${r.usage==='partial'?'≥ ':''}${compact(totalTokens(r))}<small>${{complete:'完整用量',partial:'已知下限',unknown:'用量未知'}[r.usage]}</small></td><td>${btn('view-experiment','↗','icon-button',`data-id="${h(r.id)}" aria-label="查看实验 ${h(r.title)}"`)}</td></tr>`).join('')}</tbody></table></div>`)+
  `<details class="source-details"><summary>来源与统计口径 · ${state.sources.filter(s=>s.status==='loaded').length} 个已读取文件</summary><p>历史结果只读，每次刷新从仓库重读；来源文件不被前端覆盖。成功率只统计已完成且有明确成功判定的回合；完成度按已有评分单独计数。已知消耗包含中断回合，不等于实际货币账单。不同方法的合并统计仅为本组描述。</p>${state.sources.map(s=>`<div class="source-row"><span>${s.status==='loaded'?'✓':'!'} ${h(s.path)}</span><code>${h(s.sha256?.slice(0,16) || s.error)}</code></div>`).join('')}</details>`;
}
function notesPage() {
  const f=filters.notes, notes=state.notes.filter(n=>(!f.kind || n.kind===f.kind)&&`${n.title} ${n.body}`.toLowerCase().includes(f.query.toLowerCase())).sort((a,b)=>b.date.localeCompare(a.date)||b.updated_at.localeCompare(a.updated_at));
  return titleBlock('RESEARCH / JOURNAL','研究日志','留下实验发现、设计决策与阻塞原因，方便下次继续。',btn('new-note','＋ 记录进展','button primary'))+
  `<div class="filters"><label class="search"><span>⌕</span><input data-filter="notes.query" aria-label="搜索日志" placeholder="搜索记录与研究发现…" value="${h(f.query)}"></label><select data-filter="notes.kind" aria-label="日志类型">${opts(noteKinds,f.kind,'全部类型')}</select></div>`+
  (notes.length?`<div class="notes-grid">${notes.map(n=>`<article class="note-card"><div class="note-top"><span class="note-kind ${h(n.kind)}">${h(noteKinds[n.kind])}</span><time>${h(n.date)}</time></div><h2><button class="text-button" data-action="edit-note" data-id="${h(n.id)}">${h(n.title)}</button></h2><p>${h(n.body)}</p><footer><span>${h(state.tasks.find(t=>t.id===n.task_id)?.title || '项目记录')}</span>${btn('edit-note','编辑 ↗','text-button',`data-id="${h(n.id)}"`)}</footer></article>`).join('')}</div>`:empty('暂无匹配的记录','记录今天的进展、发现或下一步。'));
}

function field(label,key,value='',type='text',options={}) {
  const {required=false,full=false,help='',values,min,max,step}=options;
  const attrs=`name="${key}" id="field-${key}" ${required?'required':''}`;
  const control=type==='textarea'?`<textarea ${attrs} rows="${key==='body'?9:3}">${h(value)}</textarea>`:type==='select'?`<select ${attrs}>${opts(values,value,null)}</select>`:`<input ${attrs} type="${type}" value="${h(value)}" ${min!=null?`min="${min}"`:''} ${max!=null?`max="${max}"`:''} ${step!=null?`step="${step}"`:''}>`;
  return `<div class="form-field ${full?'full':''}"><label for="field-${key}">${h(label)}${required?' <span>*</span>':''}</label>${control}${help?`<small>${h(help)}</small>`:''}</div>`;
}
const selectField=(label,key,value,values,full=false)=>field(label,key,value,'select',{values,full});
function dialogShell(title,subtitle,body,record=null,collection=null) {
  editing={record,collection};
  $('#editor').innerHTML=`<form id="edit-form"><header class="dialog-header"><div><div class="eyebrow">${record?.id?h(record.id):'NEW RECORD'}</div><h2 id="dialog-title">${h(title)}</h2><p>${h(subtitle)}</p></div>${btn('close-dialog','×','close-button','type="button" aria-label="关闭对话框"')}</header><div class="dialog-body">${body}<div class="form-error" id="form-error" role="alert" hidden></div></div><footer class="dialog-footer">${record && collection?btn('delete-record','删除记录','button danger','type="button"'):''}<span class="spacer"></span>${btn('close-dialog',collection?'取消':'关闭','button','type="button"')}${collection?'<button class="button primary" type="submit">保存记录</button>':''}</footer></form>`;
  $('#editor').showModal();
}
function openTask(id) {
  const t=state.tasks.find(t=>t.id===id), d=t||{title:'',track:'rag',status:'todo',priority:'P1',goals:['ai'],dependencies:[]};
  dialogShell(t?'编辑研究任务':'新建研究任务','任务状态反映研究进展；验收证据决定是否能够完成。',
    `<div class="form-grid">${field('任务名称','title',d.title,'text',{required:true,full:true})}${selectField('研究主线','track',d.track,tracks)}${selectField('任务状态','status',d.status,statusNames)}${selectField('优先级','priority',d.priority,{P0:'P0 · 当前重点',P1:'P1 · 后续推进',P2:'P2 · 探索拓展'})}${field('负责人','owner',d.owner)}${field('截止日期','due',d.due,'date')}<fieldset class="goal-checkboxes"><legend>关联研究目标</legend>${Object.entries(goals).map(([key,name])=>`<label><input type="checkbox" name="goals" value="${key}" ${d.goals.includes(key)?'checked':''}>${name}</label>`).join('')}</fieldset>${field('研究内容与现状','description',d.description,'textarea',{full:true})}${field('验收标准','acceptance',d.acceptance,'textarea',{full:true})}${field('下一步行动','next_step',d.next_step,'textarea',{full:true})}${field('阻塞原因','blocker',d.blocker,'textarea',{full:true})}<div class="form-field full"><label for="field-dependencies">前置依赖</label><select id="field-dependencies" name="dependencies" multiple size="5">${state.tasks.filter(x=>x.id!==id).map(x=>`<option value="${h(x.id)}" ${d.dependencies.includes(x.id)?'selected':''}>${h(x.id)} · ${h(x.title)} · ${h(statusNames[x.status])}</option>`).join('')}</select><small>按住 Ctrl / Command 多选；依赖图不允许环。</small></div>${field('证据 / 相关文件','evidence',d.evidence,'textarea',{full:true,help:'每行一个项目相对路径或说明。'})}</div>${t?`<div class="dialog-evidence">${evidenceLinks(t.evidence)}</div>`:''}`,t,'tasks');
}
function openNote(id) {
  const n=state.notes.find(n=>n.id===id), d=n||{date:localDate(),kind:'update'};
  dialogShell(n?'编辑研究日志':'记录研究进展','写下变化、结论、证据与下一步。',`<div class="form-grid">${field('标题','title',d.title,'text',{required:true,full:true})}${field('日期','date',d.date,'date',{required:true})}${selectField('记录类型','kind',d.kind,noteKinds)}${selectField('关联任务','task_id',d.task_id||'',{'':'项目级记录',...Object.fromEntries(state.tasks.map(t=>[t.id,t.title]))},true)}${field('记录内容','body',d.body,'textarea',{required:true,full:true})}</div>`,n,'notes');
}
function openExperiment(id) {
  const r=state.experiments.find(r=>r.id===id), d=r||{date:localDate(),status:'planned',split:'development',transport:'mcp',usage:'unknown',condition:'standard',effort:'',success:null};
  const numberField=(label,key,help='',max=null)=>field(label,key,d[key],'number',{min:0,max,step:key.endsWith('tokens')?1:'any',help});
  dialogShell(r?'编辑实验记录':'记录一次实验','一条记录对应一次执行；未知值留空，失败与中断也需要保留。',
  `<div class="form-grid">${field('实验名称','title',d.title,'text',{required:true,full:true})}${field('执行日期','date',d.date,'date',{required:true})}${selectField('运行状态','status',d.status,runNames)}${field('机器人任务','task',d.task,'text',{required:true,help:'如 stack_blocks / fold_clothes'})}${field('方法 / 对照条件','method',d.method,'text',{required:true,help:'如 Graph RAG + plan graph / no RAG'})}${field('评测协议 ID','protocol',d.protocol,'text',{required:true})}${field('场景变体','condition',d.condition,'text',{required:true})}${field('模型 / 固定快照','model',d.model,'text',{required:true})}${field('推理强度','effort',d.effort,'text',{required:true,help:'如 medium / xhigh / not-applicable'})}${selectField('数据划分','split',d.split,splits)}${selectField('通信协议','transport',d.transport,transports)}${field('布局 ID','layout',d.layout)}${field('随机种子','seed',d.seed)}${field('经验库版本','bank_version',d.bank_version)}${field('代码版本','code_version',d.code_version)}${selectField('关联研究任务','task_id',d.task_id||'',{'':'未关联',...Object.fromEntries(state.tasks.map(t=>[t.id,t.title]))},true)}
  <h3 class="form-section">MCP 调用证据</h3>${field('MCP server','mcp_server',d.mcp_server)}${field('MCP 协议版本','mcp_version',d.mcp_version)}${field('Trace / 请求链路路径','trace',d.trace,'text',{full:true,help:'MCP 完整回合需填写 server、协议版本、trace 与下方结果证据。'})}
  <h3 class="form-section">任务结果与速度</h3>${selectField('最终任务成功','success',d.success==null?'unknown':String(d.success),{unknown:'未知 / 尚未评分',true:'成功',false:'未成功'})}${numberField('最终任务完成度 (%)','completion','原生评分归一到 0–100；未完整结束请留空。',100)}${numberField('端到端墙钟时间 (s)','wall_seconds','单回合从开始到结束的实际时间。')}${numberField('检索耗时 (s)','retrieval_seconds')}${numberField('动作执行耗时 (s)','execution_seconds')}
  <h3 class="form-section">Token 消耗</h3>${selectField('用量完整性','usage',d.usage,{unknown:'未知',complete:'完整',partial:'部分记录 / 已知下限'})}${numberField('输入 tokens','input_tokens')}${numberField('输出 tokens','output_tokens')}${numberField('缓存输入 tokens','cached_tokens','属于输入子集，不额外累加。')}${numberField('推理输出 tokens','reasoning_tokens','属于输出子集，不额外累加。')}${numberField('总 tokens','total_tokens','输入和输出齐全时自动计算；不填 0 代替未知。')}
  <h3 class="form-section">证据与备注</h3>${field('结果证据 / 日志路径','evidence',d.evidence,'textarea',{full:true})}${field('实验备注与限制','notes',d.notes,'textarea',{full:true,help:'说明预算、布局配对、并发、重试、失败原因等比较条件。'})}</div>`,r,'experiments');
}
function viewExperiment(id) {
  const r=allRuns().find(r=>r.id===id);if(!r)return;
  const items={'方法':r.method,'任务 / 场景':`${r.task} / ${r.condition}`,'评测协议':r.protocol,'模型 / 推理强度':`${r.model} / ${r.effort}`,'数据划分':splits[r.split],'布局 / seed':`${r.layout || '未知'} / ${r.seed || '未知'}`,'通信协议':transports[r.transport],'运行状态':runNames[r.status],'成功判定':r.success==null?'未知':r.success?'成功':'未成功','完成度':pct(r.completion),'墙钟耗时':seconds(r.wall_seconds),'检索 / 执行耗时':`${seconds(r.retrieval_seconds)} / ${seconds(r.execution_seconds)}`,'输入 / 输出 token':`${fmt(r.input_tokens)} / ${fmt(r.output_tokens)}`,'缓存 / 推理 token':`${fmt(r.cached_tokens)} / ${fmt(r.reasoning_tokens)}`,'总 token':`${r.usage==='partial'?'≥ ':''}${fmt(totalTokens(r))}`,'用量完整性':{complete:'完整',partial:'已知下限',unknown:'未知'}[r.usage],'经验库版本':r.bank_version || '未注明','代码版本':r.code_version || '未注明','MCP server / version':`${r.mcp_server || '—'} / ${r.mcp_version || '—'}`,'Trace':r.trace || '—'};
  dialogShell(r.title,r.readonly?'历史记录 · 从仓库来源文件读取，刷新后更新。':'手动实验记录 · 保存在本地项目中。',`<div class="detail-grid">${Object.entries(items).map(([key,value])=>`<div><dt>${h(key)}</dt><dd>${h(value)}</dd></div>`).join('')}</div><h3>备注与限制</h3><p class="pre-wrap">${h(r.notes || '未添加')}</p><h3>原始证据</h3><div class="dialog-evidence">${evidenceLinks(r.evidence)}</div>${!r.readonly?`<div class="detail-actions">${btn('edit-experiment','编辑此记录','button primary',`data-id="${h(r.id)}" type="button"`)}</div>`:''}`);
}
async function saveForm(event) {
  event.preventDefault();if(!editing?.collection || busy)return;
  const form=event.target, data=Object.fromEntries(new FormData(form));
  if(editing.collection==='tasks'){data.goals=new FormData(form).getAll('goals');data.dependencies=new FormData(form).getAll('dependencies');}
  if(editing.collection==='experiments'){
    data.success=data.success==='unknown'?null:data.success==='true';
    ['completion','wall_seconds','retrieval_seconds','execution_seconds','input_tokens','output_tokens','cached_tokens','reasoning_tokens','total_tokens'].forEach(k=>data[k]=data[k]===''?null:Number(data[k]));
  }
  if(editing.record)data.revision=editing.record.revision;
  busy=true;form.querySelector('[type=submit]').disabled=true;$('#form-error').hidden=true;connection('正在保存…');
  try{
    await api(`/api/${editing.collection}${editing.record?'/'+encodeURIComponent(editing.record.id):''}`,{method:editing.record?'PUT':'POST',body:JSON.stringify(data)});
    $('#editor').close();await refresh();toast('记录已保存到本地项目');
  }catch(error){$('#form-error').textContent=error.message;$('#form-error').hidden=false;$('#form-error').scrollIntoView({block:'nearest'});connection('保存未完成',true);}
  finally{busy=false;form.querySelector('[type=submit]').disabled=false;}
}
function confirmDelete() {
  pendingDelete=editing;
  $('#confirm-dialog').innerHTML=`<div class="confirm-content"><h2 id="confirm-title">删除这条记录？</h2><p>${h(editing.record.title)}</p><p class="muted">删除后从当前列表移除。维护历史仍保留；有依赖或关联的任务需要先解除关联。</p><div class="form-error" id="delete-error" hidden></div><div class="heading-actions">${btn('cancel-delete','取消','button')}${btn('confirm-delete','确认删除','button danger')}</div></div>`;
  $('#confirm-dialog').showModal();
}
async function deleteRecord() {
  if(busy || !pendingDelete)return;busy=true;
  try {await api(`/api/${pendingDelete.collection}/${encodeURIComponent(pendingDelete.record.id)}`,{method:'DELETE',body:JSON.stringify({revision:pendingDelete.record.revision})});$('#confirm-dialog').close();$('#editor').close();await refresh();toast('记录已删除');}
  catch(error){$('#delete-error').textContent=error.message;$('#delete-error').hidden=false;}
  finally{busy=false;}
}
function changeFilter(element) {
  const [section,key]=element.dataset.filter.split('.'), value=element.value;
  const start=element.selectionStart,end=element.selectionEnd;
  filters[section][key]=value;render();
  const replacement=document.querySelector(`[data-filter="${section}.${key}"]`);
  replacement?.focus();if(start!=null)replacement?.setSelectionRange(start,end);
}
function syncTokenTotal() {
  if(editing?.collection !== 'experiments')return;
  const input=$('#field-input_tokens'), output=$('#field-output_tokens'), total=$('#field-total_tokens');
  if(input.value!=='' && output.value!=='') {
    total.value=Number(input.value)+Number(output.value);total.readOnly=true;total.dataset.derived='true';
  } else {
    if(total.dataset.derived==='true')total.value='';
    total.readOnly=false;delete total.dataset.derived;
  }
}
document.addEventListener('input',event=>{
  if(event.target.matches('input[data-filter]') && !event.isComposing)changeFilter(event.target);
  if(['field-input_tokens','field-output_tokens'].includes(event.target.id))syncTokenTotal();
});
document.addEventListener('compositionend',event=>{if(event.target.matches('input[data-filter]'))changeFilter(event.target);});
document.addEventListener('change',event=>{if(event.target.matches('select[data-filter]'))changeFilter(event.target);});
document.addEventListener('submit',event=>{if(event.target.id==='edit-form')saveForm(event);});
document.addEventListener('click',event=>{
  const target=event.target.closest('[data-action]');if(!target)return;
  const id=target.dataset.id, action=target.dataset.action;
  if(action==='refresh')return refresh(true);
  if(!state)return;
  if(action==='new-task')openTask();
  if(action==='edit-task')openTask(id);
  if(action==='new-note')openNote();
  if(action==='edit-note')openNote(id);
  if(action==='new-experiment'){openExperiment();syncTokenTotal();}
  if(action==='view-experiment')viewExperiment(id);
  if(action==='edit-experiment'){$('#editor').close();openExperiment(id);syncTokenTotal();}
  if(action==='close-dialog' && !busy)$('#editor').close();
  if(action==='delete-record')confirmDelete();
  if(action==='cancel-delete' && !busy)$('#confirm-dialog').close();
  if(action==='confirm-delete')deleteRecord();
  if(action==='task-view'){filters.tasks.view=id;render();}
  if(action==='filter-track'){filters.tasks.track=id;location.hash='tasks';if(page==='tasks')render();}
  if(action==='storage')dialogShell('本地存储与备份','记录保存在 RoboRAG 项目内，浏览器刷新不会清空。',`<p>数据库位置</p><code class="storage-path">${h(state.storage)}</code><p>项目数据使用 SQLite 保存。JSON 导出包含当前任务、手动实验、日志与历史来源快照；CSV 导出包含全部实验。恢复时可使用项目提供的恢复脚本。</p><div class="heading-actions"><a class="button primary" href="/api/export" download>↓ 导出 JSON</a><a class="button" href="/api/export?format=csv" download>↓ 导出实验 CSV</a></div>`);
});
$('#editor').addEventListener('cancel',event=>{if(busy)event.preventDefault();});
$('#confirm-dialog').addEventListener('cancel',event=>{if(busy)event.preventDefault();});
window.addEventListener('hashchange',()=>{if(state){render();window.scrollTo(0,0);}});
refresh();
