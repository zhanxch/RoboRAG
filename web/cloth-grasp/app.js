const $=id=>document.getElementById(id);
const names={left_sleeve:'左袖口',right_sleeve:'右袖口',left_hem:'左衣摆',right_hem:'右衣摆'};
let state={},mode=['demo','live','episode'].includes(new URLSearchParams(location.search).get('view'))?new URLSearchParams(location.search).get('view'):'demo',frame=0,playing=false,camera='cam_head',selectedEpisode=null;
const geometries=new Map();
const labels={starting:'启动物理仿真',prepared:'示范已准备',collecting:'采集物理交互',learning:'更新世界模型与策略',evaluating:'独立物理验证',complete:'本轮已完成',failed:'运行失败',stopped:'已停止'};
function setText(id,value){$(id).textContent=value}
function allEpisodes(){return [...(state.episodes||[]),...(state.focus?.episodes||[]),...(state.mimic?.episodes||[])]}
function showFrame(){
 const live=state.mimic?.live||state.focus?.live||state.live;
 let src,notice='',label='TRAIN DEMONSTRATION',number='';
 if(mode==='demo'&&state.demo){src=`/data/demo_frames/${camera}_${String(frame).padStart(4,'0')}.jpg`;number=`${frame} / ${state.demo.frames-1}`;$('scrub').max=state.demo.frames-3;$('scrub').step=3;$('scrub').value=frame;setText('time',(frame/state.demo.frequency_hz).toFixed(2)+' s');}
 else if(mode==='live'&&live){const age=Math.max(0,Math.floor(Date.now()/1000-live.updated));src=(state.mimic?.live?'/mimic/':state.focus?.live?'/focus/':'/data/')+live.frames[camera]+'?t='+live.updated;label=(age>5?'LAST PHYSICS FRAME / ':'LIVE PHYSICS / ')+(names[live.target]||live.target);number='step '+live.step+' · '+age+'s 前';}
 else if(mode==='episode'){
  const ep=allEpisodes().find(e=>e.id===selectedEpisode);
  if(ep?.frames?.length){let f=ep.frames[Math.min(frame,ep.frames.length-1)];src=(ep.mimic_path?'/mimic/'+ep.mimic_path:ep.focus_path?'/focus/'+ep.focus_path:'/data/')+f[camera];label=(ep.role==='test'?'FROZEN TEST / '+ep.policy.toUpperCase():ep.role==='seen_development'?'SEEN-CONDITION DEVELOPMENT':ep.role==='seen_confirmation'?'SEEN-CONDITION CONFIRMATION':ep.role==='validation'||ep.role==='world_holdout'?'HELD-OUT PHYSICS':'TRAIN PHYSICS')+' / '+(names[ep.target]||ep.target);number=`${Math.min(frame,ep.frames.length-1)+1} / ${ep.frames.length}`;$('scrub').max=ep.frames.length-1;$('scrub').step=1;$('scrub').value=frame;setText('time',(f.step/25).toFixed(2)+' s');}else notice='选择一个已完成的物理回合。';
 }else notice='仿真首次输出相机画面后，会自动显示在这里。';
 $('frame').hidden=!src;if(src&&$('frame').getAttribute('src')!==src)$('frame').src=src;
 $('sceneNotice').hidden=!notice;setText('sceneNotice',notice);setText('frameLabel',label);setText('frameNumber',number);
 $('scrub').disabled=mode==='live';$('play').disabled=mode==='live';$('episodeSelect').hidden=mode!=='episode';
 setText('evidence',mode==='demo'?'画面来自原始 TRAIN 示范。示范中的动作不代表训练后策略已经成功。':mode==='live'?'显示最近一次记录的物理画面；复位和预定位时会暂时保留上一帧，右上角显示画面年龄。布料没有被绑定到夹爪。':'画面来自原生 PhysX 布料与机械臂交互，没有把布料绑定到夹爪。局部抓取验证与完整折叠任务评分分开记录。');
 drawGeometry();
}
function drawGeometry(){
 const ep=allEpisodes().find(e=>e.id===(mode==='episode'?selectedEpisode:(state.mimic?.live||state.focus?.live||state.live)?.episode));
 const data=ep&&geometries.get(ep.id);$('geometryPanel').hidden=mode==='demo'||!data;
 if(!data)return;
 const step=mode==='episode'?ep.frames[Math.min(frame,ep.frames.length-1)].step:data.tcp.length-1;
 const i=Math.min(step,data.tcp.length-1),X=x=>280+x*1800,Y=z=>185-z*700;
 const path=arr=>arr.slice(0,i+1).map(p=>X(p[0])+','+Y(p[2])).join(' ');
 let svg='';for(const z of [0,.08,.16,.24])svg+=`<line x1="30" y1="${Y(z)}" x2="530" y2="${Y(z)}" stroke="#303a34"/><text x="8" y="${Y(z)+4}" fill="#91a198" font-size="10">${Math.round(z*100)}</text>`;
 svg+=`<polyline points="${path(data.tcp)}" fill="none" stroke="#91a198" stroke-dasharray="3 3"/><polyline points="${path(data.target)}" fill="none" stroke="#bfe792"/>`;
 for(const p of data.patch[i])svg+=`<circle cx="${X(p[0])}" cy="${Y(p[2])}" r="2" fill="#91a198"/>`;
 const p=data.target[i],t=data.tcp[i];svg+=`<circle cx="${X(p[0])}" cy="${Y(p[2])}" r="5" fill="#bfe792"/><circle cx="${X(t[0])}" cy="${Y(t[2])}" r="7" stroke="#e8eee9" fill="none"/><text x="300" y="205" fill="#91a198" font-size="10">X →</text><text x="30" y="16" fill="#91a198" font-size="10">Z ↑</text>`;$('geometryChart').innerHTML=svg;
}
async function loadGeometry(){for(const e of allEpisodes()){if(geometries.has(e.id))continue;try{const r=await fetch(`${e.mimic_path?'/mimic/'+e.mimic_path:e.focus_path?'/focus/'+e.focus_path:'/data/'}rollouts/${encodeURIComponent(e.id)}/geometry.json`);if(r.ok)geometries.set(e.id,await r.json())}catch{}}drawGeometry()}
function plot(id,points,threshold){
 const el=$(id),W=560,H=190,x0=46,y0=18,w=498,h=138;
 if(!points.length){el.innerHTML='<text x="46" y="90" fill="#91a198" font-size="12">等待真实训练记录</text>';return;}
 const maxX=Math.max(1,...points.map(p=>p[0])),minY=Math.min(0,...points.map(p=>p[1])),maxY=Math.max(threshold||0,...points.map(p=>p[1]),.001)*1.12;
 const X=x=>x0+x/maxX*w,Y=y=>y0+h-(y-minY)/(maxY-minY)*h;
 let s='';for(let i=0;i<4;i++){const y=minY+(maxY-minY)*i/3;s+=`<line x1="${x0}" y1="${Y(y)}" x2="${x0+w}" y2="${Y(y)}" stroke="#303a34"/><text x="${x0-8}" y="${Y(y)+4}" text-anchor="end" fill="#91a198" font-size="10">${y.toFixed(maxY<2?3:1)}</text>`;}
 if(threshold)s+=`<line x1="${x0}" y1="${Y(threshold)}" x2="${x0+w}" y2="${Y(threshold)}" stroke="#91a198" stroke-dasharray="4 4"/>`;
 s+=`<polyline points="${points.map(p=>X(p[0])+','+Y(p[1])).join(' ')}" fill="none" stroke="#bfe792" stroke-width="2"/>`;
 for(const p of points)s+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.5" fill="#bfe792"><title>${p[0]}: ${p[1].toFixed(4)}</title></circle>`;
 s+=`<text x="${x0}" y="180" fill="#91a198" font-size="10">0</text><text x="${x0+w}" y="180" text-anchor="end" fill="#91a198" font-size="10">${maxX}</text>`;el.innerHTML=s;
}
function render(){
 const s=state.status||{};setText('status',labels[s.status]||s.status);setText('stage',labels[s.status]||'准备实验');
 setText('stageDetail',s.status==='stopped_by_user_method_correction'?'已停止自定义 BC＋RL 训练，保留结果；待明确 SuperMimic 原法后重建协议。':s.error||s.detail||(state.results?'全部 48 次冻结模型测试已完成。':'TRAIN 动作重定向采样；测试案例已预先固定。'));setText('runName',state.run||'');
 if(state.demo){setText('demoMeta',`${state.demo.frames} 帧 · ${state.demo.frequency_hz} Hz · 4 个局部抓取窗口`);if(!$('targetButtons').children.length)for(const e of state.demo.events){let b=document.createElement('button');b.textContent=names[e.target];b.onclick=()=>{setMode('demo');frame=e.start-e.start%3;showFrame()};$('targetButtons').append(b);}}
 const eps=state.episodes||[],m=state.metrics||[],last=m.at(-1)||{};
 const training=eps.filter(e=>['train','world_holdout'].includes(e.role));
 setText('simSteps',eps.reduce((n,e)=>n+(e.steps||0),0).toLocaleString()+' 步');setText('episodes',`${training.length} / ${state.config?.episodes??'—'} 训练`);
 setText('params',state.config?.actor_parameters?.toLocaleString()||'待初始化');setText('observation',state.config?.observation_label||'仿真局部几何（待初始化）');
 $('progress').style.width=Math.min(100,training.length/(state.config?.episodes||1)*100)+'%';
 if(state.evaluation){const e=state.evaluation;setText('evalRate',`${e.successes} / ${e.total} 成功`);setText('evalDescription',e.description);$('targetResults').replaceChildren();for(const [k,v]of Object.entries(e.by_target||{})){const p=document.createElement('p');p.textContent=`${names[k]||k} · ${v.successes} / ${v.total}`;$('targetResults').append(p);}}
 plot('modelChart',m.filter(x=>Number.isFinite(x.wm_validation_rmse)).map(x=>[x.update,x.wm_validation_rmse]));plot('liftChart',eps.map((x,i)=>[i+1,100*x.max_lift_m]),8);
 $('episodeRows').replaceChildren();for(const e of eps.slice(-12).reverse()){const tr=document.createElement('tr');for(const v of [e.id,names[e.target]||e.target,e.controller,(100*e.max_lift_m).toFixed(1)+' cm',e.final_hold_steps,e.success?'成功':e.reason||'未成功']){let td=document.createElement('td');td.textContent=v;tr.append(td)}tr.lastChild.className=e.success?'ok':'fail';$('episodeRows').append(tr);}
 const options=allEpisodes().filter(e=>e.frames?.length);if($('episodeSelect').options.length!==options.length){$('episodeSelect').replaceChildren();for(const e of options){const o=document.createElement('option');o.value=e.id;o.textContent=`${e.id} · ${names[e.target]} · ${e.success?'成功':'未成功'}`;$('episodeSelect').append(o);}selectedEpisode=selectedEpisode??options.at(-1)?.id;if(selectedEpisode)$('episodeSelect').value=selectedEpisode;}
 renderBenchmark();renderCoverage();renderScale();renderFocus();renderMimic();setText('updated','最近读取 '+new Date().toLocaleTimeString('zh-CN')+' · 2 秒刷新');showFrame();
}
function renderBenchmark(){
 const bench=state.benchmark;$('benchmarkPanel').hidden=!bench;if(!bench)return;
 const tests=(state.episodes||[]).filter(e=>e.role==='test');
 setText('benchProgress',`${tests.length} / 48 次已测试`);
 for(const model of ['bc','rl']){const rows=tests.filter(e=>e.policy===model),n=rows.filter(e=>e.success).length;setText(model+'Score',rows.length?`${n} / ${rows.length} 成功 · ${(100*n/rows.length).toFixed(1)}%`:'等待测试');$(model+'Bar').style.width=rows.length?100*n/rows.length+'%':'0%';}
 if(state.results){const r=state.results,delta=r.models.rl.rate-r.models.bc.rate;setText('benchConclusion',`RL 相比 BC ${delta>0?'提高':delta<0?'下降':'持平'}${delta?' '+Math.abs(100*delta).toFixed(1)+' 个百分点':''}。配对案例：仅 RL 成功 ${r.rl_only} 个，仅 BC 成功 ${r.bc_only} 个。这是单训练种子的小型结构化测试，不能据此声称通用抓取能力。`);}
 setText('benchHash','冻结于 '+new Date(bench.frozen_at*1000).toLocaleString('zh-CN')+' · SHA-256 '+bench.sha256);
 const part=$('partFilter').value,garment=$('garmentFilter').value;$('benchRows').replaceChildren();
 for(const c of bench.cases.filter(c=>(part==='all'||part===c.subtask)&&(garment==='all'||Number(garment)===c.garment))){
  const tr=document.createElement('tr');for(const value of [c.id,String(c.garment).padStart(2,'0')+' / '+names[c.subtask],c.tcp_offset_m.map(v=>(100*v).toFixed(1)).join(' / ')+' cm']){const td=document.createElement('td');td.textContent=value;tr.append(td);}
  for(const model of ['bc','rl']){const td=document.createElement('td'),e=tests.find(e=>e.case_id===c.id&&e.policy===model);if(e){const b=document.createElement('button');b.textContent=e.success?'成功 ↗':'未成功 ↗';b.className=e.success?'ok':'fail';b.setAttribute('aria-label',`查看 ${model.toUpperCase()} ${c.id} 回放`);b.disabled=!e.frames?.length;b.onclick=()=>{selectedEpisode=e.id;$('episodeSelect').value=e.id;setMode('episode');document.querySelector('.viewer').scrollIntoView({behavior:'smooth',block:'start'})};td.append(b)}else td.textContent='待测试';tr.append(td);}$('benchRows').append(tr);
 }
 $('downloads').replaceChildren();const labels={'cloth-grasp-model-and-results.zip':'模型与结果打包','policy.ts':'RL TorchScript','bc-policy.ts':'BC TorchScript','benchmark.json':'测试清单','benchmark.csv':'逐案例 CSV','model-card.json':'模型说明'};
 for(const file of state.downloads||[]){if(!labels[file])continue;const a=document.createElement('a');a.href='/data/'+file;a.textContent=labels[file];a.download=file;$('downloads').append(a);}
}
$('partFilter').addEventListener('change',renderBenchmark);$('garmentFilter').addEventListener('change',renderBenchmark);
function renderCoverage(){
 const coverage=state.coverage;$('coveragePanel').hidden=!!state.focus||!coverage;if(state.focus||!coverage)return;
 const s=coverage.summary;setText('coverageSummary',`${s.base_tasks} 个基础任务 + ${s.randomized_variants} 个随机化变体 · ${s.eval_layout_files.toLocaleString()} 个布局文件 · 启动时已有 ${s.local_unique_train_episodes} 条独立 TRAIN 示范（新增数据见上方进度）。本策略已完成的完整任务测试：${s.full_tasks_evaluated_by_this_policy}。`);
 const families={cloth:'布料',rigid_pick_place:'刚体抓取放置',assembly:'装配',articulated_manipulation:'关节物体操作',dynamic_pickup:'动态物体抓取',contact_or_pushing:'接触与推移',fluid_transfer:'液体转移'};
 const selected=$('familyFilter').value;$('coverageRows').replaceChildren();
 for(const row of coverage.tasks.filter(r=>selected==='all'||r.family===selected)){
  const tr=document.createElement('tr');for(const value of [row.task,families[row.family]||row.family,row.eval_layouts_total,row.train_hdf5_count,row.task==='fold_clothes'?'仅局部抓取实验':'待适配与验证']){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('coverageRows').append(tr);
 }
}
$('familyFilter').addEventListener('change',renderCoverage);
function setMode(value){mode=value;frame=0;playing=false;setText('play','播放');document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));showFrame()}
document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('active',b.dataset.mode===mode);b.addEventListener('click',()=>setMode(b.dataset.mode))});$('camera').addEventListener('change',e=>{camera=e.target.value;showFrame()});$('episodeSelect').addEventListener('change',e=>{selectedEpisode=e.target.value;frame=0;showFrame()});$('scrub').addEventListener('input',e=>{frame=Number(e.target.value);showFrame()});$('play').addEventListener('click',()=>{playing=!playing;setText('play',playing?'暂停':'播放')});
let playbackTicks=0;
setInterval(()=>{if(!playing){playbackTicks=0;return;}if(++playbackTicks<(mode==='demo'?3:4))return;playbackTicks=0;frame+=mode==='demo'?3:1;if(frame>Number($('scrub').max))frame=0;showFrame()},40);
async function refresh(){try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw new Error(r.status);state=await r.json();render();loadGeometry()}catch(e){setText('status','连接中断 · 请检查本地服务')}}refresh();setInterval(refresh,2000);

function renderScale(){
 const d=state.scale;$('scalePanel').hidden=!!state.focus||!d?.plan;if(state.focus||!d?.plan)return;
 const status={waiting:state.results?'等待 TRAIN 数据':'等待数据与审计',running:'运行中',failed:'需要处理',development_complete:'开发集运行完成'};
 setText('scaleStatus',status[d.status?.status]||'准备数据');
 const end=new Date(d.plan.deadline*1000).toLocaleString('zh-CN');
 setText('scaleSummary',`RTX 4090 · 最长 72 小时 · 截止 ${end}。示范文件 ${d.data?.completed||0}/${d.data?.total||15}；已采集 ${d.physics_completed}/88 个物理回合。`);
 const stage=d.status?.stage||''; const activity=stage.startsWith('collect-')?'正在采集真实物理训练回合':stage.startsWith('fit-')?'正在训练模型':stage.startsWith('dev-')?'正在运行开发集配对验证':d.status?.status==='waiting'?'数据文件校验完成后自动启动采集与训练':'准备中'; setText('scaleActivity',d.status?.error||d.data?.error||activity);
 $('scaleRows').replaceChildren();for(const m of d.models){
  const tr=document.createElement('tr');const score=p=>`${m.policies[p].successes} / ${m.policies[p].total} 成功`;
  for(const v of [m.name,m.trained?'已保存模型':'待训练',score('bc'),score('rl')]){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('scaleRows').append(tr);
 }
}

function renderFocus(){
 const f=state.focus;$('focusPanel').hidden=!f;if(!f)return;
 const p=f.progress||{},s=f.status||{},stage=s.stage||'';
 const activity=s.status==='stopped_by_user_method_correction'?'已停止 · 校正 SuperMimic 方法':s.status==='effective'?'训练见过条件已通过复核':s.status==='failed'?'训练需要处理':s.status==='budget_exhausted'?'已到运行截止时间':stage.startsWith('collect-')?'采集物理训练轨迹':stage.startsWith('learn-')?'更新世界模型与 RL 策略':stage.startsWith('probe-')||stage.startsWith('baseline-')||stage.startsWith('confirm-')?'验证训练见过的条件':'持续 RL 训练';
 setText('focusStage',activity);setText('status',activity);setText('stage',activity);
 setText('stageDetail',s.status==='stopped_by_user_method_correction'?'已停止自定义 BC＋RL 训练，保留结果；待明确 SuperMimic 原法后重建协议。':s.error||`已完成 ${p.round||0} 轮 · ${p.rl_updates||0} 次 RL 更新`);
 setText('focusSummary',`10 条已校验 TRAIN 示范 · 已完成 ${p.round||0} 轮，累计 ${p.rl_updates||0} 次 RL 更新${p.learning_revision?' · '+p.learning_revision:''}${p.rl_updates_since_actor_reset!=null?' · actor 恢复后 '+p.rl_updates_since_actor_reset+' 次更新':''} · 新采集 ${f.train_episodes} 回合、${f.train_steps.toLocaleString()} 个物理控制步。`);
 setText('focusDeadline','原 72 小时预算截止：'+new Date(f.plan.deadline*1000).toLocaleString('zh-CN'));
 const rate=x=>x?`${x.successes} / ${x.total} · ${(100*x.rate).toFixed(1)}%`:'等待物理验证';
 setText('focusLast',rate(p.last_probe));setText('focusBest',rate(p.best_probe));
 setText('focusBaseline','同条件冻结 BC 对照：'+rate(p.baseline)+'。最新 RL 评测：'+(p.last_probe?.learning_revision||'v1')+'，第 '+(p.last_probe?.round||'—')+' 轮。');
 setText('evalRate',rate(p.last_probe));setText('evalDescription','当前训练见过条件的开发验证；不是完整折叠评分。');
 $('targetResults').replaceChildren();for(const [k,v] of Object.entries(p.last_probe?.by_target||{})){const x=document.createElement('p');x.textContent=`${names[k]} · ${v.successes} / ${v.total}`;$('targetResults').append(x)}
 setText('simSteps',f.train_steps.toLocaleString()+' 新采集步');setText('episodes',`${f.train_episodes} 新训练回合`);$('progress').style.width=100*(p.last_probe?.rate||0)+'%';
 plot('focusRateChart',f.probes.map(x=>[x.round,100*x.rate]),90);
 plot('modelChart',f.metrics.filter(x=>Number.isFinite(x.world_holdout_rmse)).map(x=>[x.round,x.world_holdout_rmse]));
 const eps=f.episodes.filter(x=>x.role==='train');plot('liftChart',eps.map((x,i)=>[i+1,100*(x.max_lift_m||0)]),8);
 $('episodeRows').replaceChildren();for(const e of eps.slice(-12).reverse()){const tr=document.createElement('tr');for(const v of [e.id,names[e.target],e.controller,((e.max_lift_m||0)*100).toFixed(1)+' cm',e.final_hold_steps||0,e.success?'成功':e.reason||'未成功']){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('episodeRows').append(tr)}
 $('focusDownloads').replaceChildren();const links={'policy.ts':'旧 actor','best-export/policy.ts':'旧最佳 RL actor','best.pt':'旧训练 checkpoint','training-protocol.json':'持续训练协议','progress.json':'当前进度'};for(const name of f.downloads){if(!links[name])continue;const a=document.createElement('a');a.href='/focus/'+name;a.textContent=links[name];a.download=name.split('/').pop();$('focusDownloads').append(a)}
}

function renderMimic(){
 const m=state.mimic;$('mimicPanel').hidden=!m;if(!m)return;
 const status=m.status||{},p=m.progress||{},stage=status.stage||'';
 const label=status.status==='effective'?'稳定抓取已通过保持复核':status.status==='reference_gate_failed'?'参考抓取需要诊断':status.status==='failed'?'新实验需要处理':status.status==='references_ready'?'参考动作已验证':stage.startsWith('reference')?'验证示范参考的物理抓取':stage.startsWith('learn')?'训练 SuperMimic 策略':stage.startsWith('collect')?'采集当前策略物理轨迹':stage.startsWith('probe')?'验证当前策略的稳定抓取':status.status==='deadline_reached'?'预算已结束':'准备 SuperMimic 复现';
 setText('mimicStage',label);setText('status',label);setText('stage',label);setText('stageDetail',status.error||status.detail||label);
 const activeIds=new Set(m.plan.candidates.map(c=>c.id));
 const refs=m.episodes.filter(e=>e.role==='reference_feasibility'&&activeIds.has(e.id)),train=m.episodes.filter(e=>e.role==='train');
 setText('mimicSummary',`从零初始化 · ${m.model?.updates||p.update||0} 次策略更新 · ${train.length} 个策略训练回合 · ${refs.length}/${m.plan.candidates.length} 个参考候选已完成。`);
 setText('mimicReferences',`${refs.filter(r=>r.success).length} / ${refs.length} 稳定抓取`);
 const last=m.probes.at(-1);const rate=last?`${last.successes} / ${last.n} · ${(last.rate*100).toFixed(1)}%`:'尚未完成策略物理验证';
 setText('mimicRate',rate);setText('mimicBest',`最佳开发验证：${m.best?`${m.best.successes} / ${m.best.n} · ${(m.best.rate*100).toFixed(1)}% · 第 ${m.best.round} 轮`:'尚未完成物理复测'}。当前 checkpoint：${m.model?.policy_sha256?.slice(0,12)||'尚未更新'}。`);setText('evalRate',rate);setText('evalDescription','新判据：离桌、跟随、稳定保持；参考动作成功不计为策略成功。');$('targetResults').replaceChildren();
 setText('mimicDeadline','原 72 小时预算截止：'+new Date(m.plan.deadline*1000).toLocaleString('zh-CN'));
 setText('simSteps',train.reduce((n,e)=>n+e.steps,0).toLocaleString()+' 策略物理步');setText('episodes',`${train.length} 策略 / ${refs.length} 参考`);
 setText('params',m.model?.actor_parameters?.toLocaleString()||'等待参考可行性门槛');setText('observation','10 帧关节状态历史 + 10 帧参考运动');
 $('progress').style.width=(m.model?100*(last?.rate||0):100*refs.length/m.plan.candidates.length)+'%';
 plot('modelChart',m.metrics.filter(x=>Number.isFinite(x.world_train_loss)).map(x=>[x.update,x.world_train_loss]));
 plot('liftChart',m.episodes.map((x,i)=>[i+1,100*(x.max_lift_m||0)]));
 $('episodeRows').replaceChildren();for(const e of m.episodes.slice(-12).reverse()){
  const tr=document.createElement('tr');for(const v of [e.id,names[e.target],e.role==='reference_feasibility'?'参考可行性':e.role==='train'?'当前策略 TRAIN':'策略开发验证',(100*e.max_lift_m).toFixed(1)+' cm',`${(e.final_hold_steps/25).toFixed(2)} s / 滑移 ${(1000*e.relative_slip_m).toFixed(1)} mm`,e.success?'稳定抓取':'未通过']){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('episodeRows').append(tr);
 }
 $('mimicDownloads').replaceChildren();const labels={'plan.json':'初始协议','effective-plan.json':'当前方法与判据','adapter-amendment.json':'适配修订记录','model-card.json':'模型说明','policy.ts':'当前策略（见验证状态）','last.pt':'完整恢复 checkpoint','best-policy.ts':'最佳已验证策略','references.json':'参考可行性结果','verification-protocol.json':'稳定保持复核协议','conclusion.json':'实验结论','supermimic-cloth-model.zip':'模型与审计包','confirmed-policy.ts':'已通过保持复核的策略'};
 for(const n of m.downloads){if(!labels[n])continue;const a=document.createElement('a');a.href='/mimic/'+n;a.textContent=labels[n];a.download=n;$('mimicDownloads').append(a)}
}
