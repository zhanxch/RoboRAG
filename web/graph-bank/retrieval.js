/* The library's execution-memory inspector. No generation calls in this UI. */
let retrievalSeed=null, retrievalInitialized=false, retrievalRequest=0, excludedGraph='', lastModelContext='';

async function openRetrieval(){
  showView('retrieve');$('crumb').textContent=' / 检索与动作参考';
  const previous=$('retrievalTask').value;
  const tasks=[...new Set(state.bank.graphs.map(g=>g.task))];
  $('retrievalTask').innerHTML=tasks.map(task=>`<option value="${escapeHTML(task)}">${escapeHTML(titleFor(task))}</option>`).join('');
  $('retrievalTask').value=previous||state.task||(tasks.includes('stack_blocks')?'stack_blocks':tasks[0]);
  if(!retrievalInitialized){
    const task=$('retrievalTask').value;
    if(task==='stack_blocks')applyPreset('white');
    else setRetrievalQuery({mode:'plan',task,goal:state.bank.graphs.find(g=>g.task===task)?.instruction||''});
    retrievalInitialized=true;
  }
  if(retrievalSeed){setRetrievalQuery(retrievalSeed);retrievalSeed=null;}
  try{const info=await api('/api/retrieval-info');$('retrievalInventory').textContent=`${info.counts.graphs} 张任务图 · ${info.counts.phases} 个交互阶段 · ${info.counts.chunks} 个动作窗口。GPT 生成新计划与动作，这里提供历史参考。`}
  catch(error){$('retrievalInventory').textContent='执行索引尚未就绪';showError(error)}
  updateRetrievalMode();
}

function setRetrievalQuery(query){
  for(const [field,id] of Object.entries({mode:'retrievalMode',task:'retrievalTask',goal:'retrievalGoal',observation:'retrievalObservation',stage:'retrievalStage',arm:'retrievalArm',held_object:'retrievalHeld',target_object:'retrievalTarget',action_space:'retrievalSpace'})){
    $(id).value=query[field]||'';
  }
  $('retrievalState').value=query.state?JSON.stringify(query.state,null,2):'';
  excludedGraph=query.exclude_graph_id||'';
  $('retrievalExampleNote').hidden=!excludedGraph;
  $('retrievalExampleNote').textContent=excludedGraph?`示范查询：使用 ${excludedGraph} 的阶段起始观测；检索时排除这张源图。`:'';
  updateRetrievalMode();
}

function applyPreset(name){
  const common={task:'stack_blocks',mode:'action'};
  const presets={
    plan:{...common,mode:'plan',goal:'把三个积木堆起来',observation:'三块积木分散在桌面上，双爪空闲'},
    white:{...common,goal:'把白色块放到黑色底块上',observation:'右爪夹着白色块，黑色底块已摆在中央',arm:'right',held_object:'白色块',target_object:'黑色底块'},
    failure:{...common,goal:'重新抓取积木',observation:'抓取失败，积木仍然在桌面上'}
  };
  setRetrievalQuery(presets[name]);
}

function updateRetrievalMode(){
  const plan=$('retrievalMode').value==='plan';
  document.querySelectorAll('.action-fields').forEach(element=>element.hidden=plan);
  $('retrievalIncludeActions').disabled=plan;
}

async function runRetrieval(){
  const request=++retrievalRequest;
  clearError();$('runRetrieval').disabled=true;$('runRetrieval').textContent='正在检索…';
  try{
    const query={mode:$('retrievalMode').value,task:$('retrievalTask').value,goal:$('retrievalGoal').value,
      observation:$('retrievalObservation').value,stage:$('retrievalStage').value,
      arm:$('retrievalArm').value,held_object:$('retrievalHeld').value,target_object:$('retrievalTarget').value,
      action_space:$('retrievalSpace').value,exclude_graph_id:excludedGraph,
      k:Number($('retrievalK').value),max_context_chars:Number($('retrievalBudget').value),
      include_actions:$('retrievalMode').value==='action'&&$('retrievalIncludeActions').checked,
      state:$('retrievalState').value.trim()?JSON.parse($('retrievalState').value):{}};
    if(query.mode==='plan'){delete query.arm;delete query.held_object;delete query.target_object;delete query.state;delete query.action_space;}
    const response=await fetch('/api/retrieve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(query)});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'检索失败');if(request!==retrievalRequest)return;
    lastModelContext=data.model_context;
    $('retrievalContext').textContent=JSON.stringify(JSON.parse(lastModelContext),null,2);
    $('retrievalWarnings').innerHTML=data.warnings.map(w=>`<p class="retrieval-notice">${escapeHTML(w)}</p>`).join('');
    const d=data.diagnostics;
    $('retrievalMetrics').innerHTML=`<span><strong>${d.retrieved}</strong> 条参考 / <strong>${d.packed}</strong> 条进入上下文</span><span><strong>${d.context_chars.toLocaleString()}</strong> 字符</span><span><strong>${d.elapsed_ms}</strong> ms 本地检索</span><span>模型调用 <strong>${d.online_model_calls}</strong> 次</span>`;
    $('retrievalResults').innerHTML=data.results.map(renderRetrievalResult).join('')||'<div class="empty-state"><strong>没有找到适用经验</strong>当前条件可能超出图库覆盖范围。可以返回图库检查已有示范。</div>';
    $('retrievalMetrics').scrollIntoView({block:'nearest',behavior:'smooth'});
  }catch(error){if(request===retrievalRequest)showError(error)}
  finally{if(request===retrievalRequest){$('runRetrieval').disabled=false;$('runRetrieval').textContent='检索经验 →'}}
}

function renderRetrievalResult(result){
  const url=`#/graph/${encodeURIComponent(result.graph_id)}${result.node_id?'/'+result.node_id:''}`;
  const common=`<div class="result-path">${escapeHTML(result.task_name)} / ${escapeHTML(result.graph_id)}${result.node_id?' / '+result.node_id:''}</div><div class="memory-title"><h2>${escapeHTML(result.title)}</h2><a href="${url}" class="text-button">查看 Graph ↗</a></div><div class="match-reasons">${result.reasons.map(r=>`<span>${escapeHTML(r)}</span>`).join('')}</div>`;
  const content=result.episode_trace?`<p class="memory-cue">${escapeHTML(result.initial_cue)}</p><p class="field-note">示范实际经过的阶段，供 GPT 参考，不是待执行计划。</p><ol class="plan-chain">${result.episode_trace.map(p=>`<li><a href="#/graph/${encodeURIComponent(result.graph_id)}/${p.id}">${escapeHTML(p.recorded_step)}</a></li>`).join('')}</ol>`:
    `<div class="memory-detail"><span>历史起点</span><p>${escapeHTML(result.cue)}</p><span>示范操作</span><p>${escapeHTML(result.action)}</p><span>阶段反馈</span><p>${escapeHTML(result.feedback)}</p></div><div class="chunk-preview"><div class="chunk-pair"><figure><img loading="lazy" src="${frameURL(result.graph_id,result.chunk.interval[0])}" alt="动作前的示范画面"><figcaption>动作前 · f${result.chunk.interval[0]}</figcaption></figure><figure><img loading="lazy" src="${frameURL(result.graph_id,result.chunk.interval[1])}" alt="动作后的示范画面"><figcaption>动作后 · f${result.chunk.interval[1]}</figcaption></figure></div><div class="chunk-controls"><span>${result.chunk.interval[1]-result.chunk.interval[0]} 步 · ${((result.chunk.interval[1]-result.chunk.interval[0])/result.chunk.frequency).toFixed(2)} s · ${result.chunk.frequency} Hz</span><button class="text-button" data-chunk="${escapeHTML(result.chunk.id)}">查看数值动作 ↓</button><a class="text-button" href="${escapeHTML(result.chunk.url)}&download=1" download>下载 ↗</a></div><div class="chunk-numbers" hidden></div></div>`;
  return `<article class="memory-card">${common}${content}<div class="memory-foot"><span>排序分数 ${result.score.toFixed(3)} · 非成功概率</span><span>${result.included_in_context?'已加入模型上下文':'预算内未选入'}${result.actions_omitted_for_budget?' · 数值动作改为引用':''}</span></div></article>`;
}

async function showChunk(button){
  const panel=button.closest('.chunk-preview').querySelector('.chunk-numbers');
  if(panel.dataset.loaded){panel.hidden=!panel.hidden;button.textContent=panel.hidden?'查看数值动作 ↓':'收起数值动作 ↑';return;}
  button.disabled=true;
  try{
    const data=await api('/api/chunk?id='+encodeURIComponent(button.dataset.chunk));
    const short=['L dx','L dy','L dz','L rx','L ry','L rz','L grip','R dx','R dy','R dz','R rx','R ry','R rz','R grip'];
    panel.innerHTML=`<p class="field-note">每行是一个控制步：平移 m、旋转向量 rad、夹爪目标。首行相对起始观测，后续行相对上一控制目标。</p><div class="action-table-wrap"><table class="action-table"><thead><tr><th>步</th>${short.map(s=>`<th>${s}</th>`).join('')}</tr></thead><tbody>${data.actions.map((row,i)=>`<tr><th>${i+1}</th>${row.map(n=>`<td>${n.toFixed(5)}</td>`).join('')}</tr>`).join('')}</tbody></table></div><details><summary>执行前后状态与测量摘要</summary><pre>${escapeHTML(JSON.stringify({before:data.before.state,after:data.after.state,measured_summary:data.measured_summary},null,2))}</pre></details><p class="field-note">历史阶段反馈覆盖的时间可能长于此动作窗口；这段示范未提供独立的任务成功标签。</p>`;
    panel.dataset.loaded='true';panel.hidden=false;button.textContent='收起数值动作 ↑';
  }catch(error){showError(error)}finally{button.disabled=false;}
}

$('retrievalForm').onsubmit=event=>{event.preventDefault();runRetrieval()};
$('retrievalMode').onchange=updateRetrievalMode;
document.querySelectorAll('[data-preset]').forEach(button=>button.onclick=()=>{applyPreset(button.dataset.preset);runRetrieval()});
$('retrievalResults').onclick=event=>{const button=event.target.closest('[data-chunk]');if(button)showChunk(button)};
$('queryFromPhase').onclick=async()=>{
  try{retrievalSeed=await api('/api/query-example?id='+encodeURIComponent(state.graph.id)+'&node='+state.phase.id);location.hash='#/retrieve'}catch(error){showError(error)}
};
$('copyContext').onclick=async()=>{
  if(!lastModelContext)return;
  try{await navigator.clipboard.writeText(lastModelContext);$('copyContext').textContent='已复制';setTimeout(()=>{$('copyContext').textContent='复制'},1200)}catch{showError(new Error('请选中上下文内容复制。'))}
};
