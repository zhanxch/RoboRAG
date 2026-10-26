"use strict";
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const names = {cloth_landmarks:"测量衣物",surface_transfer:"单臂搬运",edge_transfer:"双臂搬运",table_xy:"像素定位",table_plane:"测量桌面",obs:"读取观测",home:"机械臂归位",gripper:"控制夹爪",move:"移动机械臂",done:"结束回合",surface_outline:"检测轮廓",crease_transfer:"沿折线搬运",view_reacquire:"重新观察"};
const descriptions = {cloth_landmarks:"从当前 RGB 观测提取轮廓、抓取点和几何折叠候选。此命令不移动机器人。",surface_transfer:"按指定的起点与终点执行单臂抓取、抬升、搬运、释放与归位；具体阶段以实际回执为准。",edge_transfer:"双臂抓取两个端点，沿协调轨迹搬运并释放；具体阶段以实际回执为准。",table_xy:"将指定像素投影到测得的平面，返回世界坐标。",table_plane:"通过标定 RGB 视图测量或刷新桌面高度。",obs:"刷新相机图像与机器人状态，不消耗机器人动作步。",home:"将指定机械臂返回初始关节位置；已经归位时可能不消耗动作步。",gripper:"设置指定夹爪的开合状态；指令完成不等于抓持成立。",move:"按请求中的世界坐标增量移动机械臂。",done:"智能体请求结束本回合；任务得分以原生评分器为准。"};
const stageNames = {approach:"接近抓取点",contact:"接触 / 对齐",carry_arc_1:"抬升并搬运 · 1",carry_arc_2:"搬运 · 2",carry_arc_3:"搬运 · 3",release:"释放",park:"归位",close:"闭合夹爪",lift:"抬升",open:"打开夹爪"};
const cameraNames = {head:"头部相机",wrist_l:"左腕相机",wrist_r:"右腕相机"};
const state = {data:null,episode:null,frame:0,camera:"head",variant:"all",outcome:"all",loadingVideo:false,view:"session",session:null,selectedEvent:null,actionStep:null};
const sessionCache = new Map();
const video = $("video");
const physical = steps => `${(steps / state.data.controlHz).toFixed(2)} s`;
const shortTitle = command => names[command.name] || command.name;
const ended = command => command.result.plan_fail_reason === "episode_over";
const problem = command => command.result.plan_ok === false && !ended(command);

function writeHash() {
  const q = new URLSearchParams({episode:state.episode.id,frame:String(state.frame),camera:state.camera,view:state.view});
  if(state.selectedEvent)q.set("event",state.selectedEvent);
  history.replaceState(null,"","#"+q);
}

function renderEpisodes() {
  const filtered = state.data.episodes.filter(e => (state.variant === "all" || e.variant === state.variant) && (state.outcome === "all" || e.success === (state.outcome === "success")));
  $("caseCount").textContent = `${filtered.length} / ${state.data.episodes.length}`;
  $("episodeList").innerHTML = filtered.length ? filtered.map(e => `<button class="case" data-episode="${esc(e.id)}" aria-current="${e.id === state.episode.id}" aria-label="${e.variant === "standard" ? "标准" : "杂物"}布局 ${e.layout}，${e.success ? "成功" : "未成功"}，${e.score} 分"><img src="${esc(e.frames[0].images.head)}" alt="" loading="lazy"><span class="case-content"><span class="case-title">${e.variant === "standard" ? "标准" : "杂物"} <span> / L${e.layout}</span></span><span class="case-meta"><i class="${e.success ? "success" : e.score > 0 ? "partial" : ""}"></i>${e.success ? "成功" : "未成功"} · ${e.score} 分</span></span></button>`).join("") : '<p class="empty-filter">没有符合此筛选的回合。</p>';
}

function renderTimeline() {
  const commands = state.episode.commands;
  $("timeline").innerHTML = `<button data-frame="0" aria-current="${state.frame === 0}" aria-label="初始观测"><span class="number">00</span><span class="action-name">初始观测</span><small>step 0</small></button>` + commands.map(c => `<button data-frame="${c.index}" aria-current="${state.frame === c.index}" aria-label="第 ${c.index} 条命令：${esc(c.name)}"><span class="number">${String(c.index).padStart(2,"0")}<i class="${problem(c) ? "issue" : ""}"></i></span><span class="action-name">${esc(shortTitle(c))}</span><small>step ${c.after}</small></button>`).join("");
}

function metric(label, value) { return `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`; }

function renderRobotDetail() {
  const c = state.episode.commands[state.frame - 1];
  $("commandIndex").textContent = `${String(state.frame).padStart(2,"0")} / ${state.episode.commands.length}`;
  if (!c) {
    $("commandDetail").innerHTML = `<span class="badge">初始观测</span><h3>操作开始之前</h3><p class="initial-description">这是本回合的初始相机画面。点击播放，或选择时间线中的任一命令，查看执行后的观测、参数与实际反馈。</p><dl class="detail-metrics">${metric("仿真步数", "0 / 500")}${metric("已执行命令", "0")}</dl><p class="detail-label">记录来源</p><p class="detail-description">初始观测：observations/000_initial<br>后续观测：每条命令完成后保存<br>最终结果：RoboDojo 原生评分</p>`;
    return;
  }
  const r = c.result;
  let label = ended(c) ? "回合已结束" : problem(c) ? "执行受阻" : c.exitCode === 0 ? "回执正常" : "查看退出状态";
  const arm = ["left","right","both"].includes(c.argv[2]) ? ({left:"左臂",right:"右臂",both:"双臂"}[c.argv[2]]) : "";
  const commandText = c.argv.slice(0,3).join(" ") + (c.argv.length > 3 ? "\n  " + c.argv.slice(3).join(" ").replace(/ --/g,"\n  --") : "");
  const stages = r.stages || [];
  let note = "";
  if (ended(c)) note = `<p class="receipt-note">工具在任务结束时返回 episode_over。本回合原生评分：${state.episode.success ? "成功" : "未成功"}，${state.episode.score} 分；两者分别展示。</p>`;
  else if (problem(c)) note = `<p class="receipt-note">失败原因：<code>${esc(r.plan_fail_reason || r.failure_class || "详见原始回执")}</code>${r.holding_arm ? `<br>回执显示仍持物的机械臂：${esc(r.holding_arm)}` : ""}</p>`;
  else if (stages.some(s => s.plan_ok === false)) note = `<p class="receipt-note">中间阶段曾报告失败，工具最终回执正常。下方阶段列表保留了实际恢复过程。</p>`;
  const measurements = [];
  if (r.table_z != null) measurements.push(["测得桌面高度", `${Number(r.table_z).toFixed(4)} m`]);
  if (r.plane_z != null) measurements.push(["使用的投影平面", `${Number(r.plane_z).toFixed(4)} m`]);
  if (r.uncertainty_m != null) measurements.push(["报告的不确定度", `${Number(r.uncertainty_m)*1000} mm`]);
  if (r.grasp) measurements.push(["返回抓取候选", `${Object.keys(r.grasp).length} 个`]);
  if (r.xy) measurements.push(["世界坐标 XY", JSON.stringify(r.xy)]);
  $("commandDetail").innerHTML = `<span class="badge ${problem(c) || ended(c) ? "warn" : "good"}">${label}</span><h3>${esc(shortTitle(c))}${arm ? " · " + arm : ""}</h3><p class="detail-description">${esc(descriptions[c.name] || "按请求执行此工具，详细反馈见下方原始回执。")}</p><dl class="detail-metrics">${metric("仿真步数",`${c.before} → ${c.after}`)}${metric("消耗动作步",`${c.after-c.before}`)}${metric("对应机器人时间",physical(c.after-c.before))}${metric("命令墙钟耗时",`${c.wallSeconds.toFixed(2)} s`)}</dl>${note}<p class="detail-label">实际命令 · ROBO</p><pre class="command-code">${esc(commandText)}</pre>${stages.length ? `<p class="detail-label">工具内部阶段 · ${stages.length}</p><ol class="stage-list">${stages.map(s => `<li><span class="stage-dot ${s.plan_ok === false ? "issue" : ""}"></span><div><div class="stage-heading"><span>${esc(stageNames[s.stage] || s.stage || "阶段")}</span><small>${esc(s.arm || "")}</small></div><div class="stage-detail">${esc(s.plan_fail_reason || (s.plan_ok === true ? "plan_ok" : ""))}${s.error_m != null ? ` · 误差 ${(s.error_m*1000).toFixed(2)} mm` : ""}</div></div></li>`).join("")}</ol>` : ""}${measurements.length ? `<p class="detail-label">测量回执</p><div class="measurement">${measurements.map(([k,v])=>`<div><span>${esc(k)}</span><code>${esc(v)}</code></div>`).join("")}</div>` : ""}<details class="raw-receipt"><summary>完整 JSON 回执 · exit ${c.exitCode}</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details>`;
}

function renderCurrent(scroll=false) {
  const e = state.episode, frame = e.frames[state.frame];
  $("scrub").value = state.frame;
  $("scrub").setAttribute("aria-valuetext", `第 ${state.frame} 个观测，仿真步 ${frame.step}`);
  $("frameCounter").textContent = `${String(state.frame).padStart(2,"0")} / ${e.commands.length}`;
  $("frameTag").textContent = state.frame === 0 ? "INITIAL / 初始观测" : `AFTER COMMAND ${String(state.frame).padStart(2,"0")} / 命令后`;
  $("physicalTime").textContent = `STEP ${frame.step} · ${physical(frame.step)}`;
  const command=e.commands[state.frame-1];
  const parked=command && (command.argv.includes("home") || command.name==="edge_transfer");
  $("captureNote").textContent = frame.step===0 ? "这张图仍在 step 0：截至此处只进行了读取或测量，机械臂尚未执行动作。" : command?.before===command?.after ? "本条命令未推进物理仿真。画面沿用这一时刻的机器人状态；查看会话可追溯前面的动作。" : parked ? `截图在整条工具返回后保存；该工具包含归位操作。中间实际下发了 ${command.actionCount} 个控制 action，可在详情中逐步查看。` : "这是整条命令返回后的截图。抓取和搬运过程没有连续录像；下方会话与详情中的控制 actions 保留了执行记录。";
  $("previous").disabled = state.frame === 0;
  $("next").disabled = state.frame === e.commands.length;
  document.querySelectorAll("[data-frame]").forEach(b => b.setAttribute("aria-current",String(Number(b.dataset.frame) === state.frame)));
  if (scroll) {
    const track=$("timeline"), active=track.querySelector('[aria-current="true"]');
    if(active){
      const outer=track.getBoundingClientRect(), inner=active.getBoundingClientRect();
      if(inner.right>outer.right)track.scrollLeft+=inner.right-outer.right;
      if(inner.left<outer.left)track.scrollLeft+=inner.left-outer.left;
    }
  }
  renderDetail();
  markSessionSelection();
  writeHash();
}

function setFrame(index, pause=true) {
  state.frame = Math.max(0,Math.min(state.episode.commands.length,index));
  if (pause) video.pause();
  state.selectedEvent=null;
  state.actionStep=null;
  if (video.readyState >= 1) video.currentTime = Math.min(state.frame + .04, video.duration - .01);
  renderCurrent(true);
}

function loadVideo() {
  state.loadingVideo = true;
  video.pause();
  video.poster = state.episode.frames[state.frame].images[state.camera];
  video.src = state.episode.videos[state.camera];
  video.load();
  $("downloadReplay").href = state.episode.videos[state.camera];
  $("downloadReplay").download = `${state.episode.id}-${state.camera}-observations.mp4`;
  $("downloadOriginal").href = state.episode.originals[state.camera];
  document.querySelectorAll("[data-camera]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.camera === state.camera)));
  video.setAttribute("aria-label",`${state.episode.id} ${cameraNames[state.camera]}真实观测回放`);
}

function selectEpisode(id, frame=0) {
  const episode = state.data.episodes.find(e=>e.id === id);
  if (!episode) return;
  state.episode = episode;
  state.session=sessionCache.get(id)||null;
  state.selectedEvent=null;
  state.actionStep=null;
  state.frame = Math.max(0,Math.min(episode.commands.length,Number.isInteger(frame) ? frame : 0));
  $("episodeSplit").textContent = `${episode.split === "development" ? "DEVELOPMENT / 开发布局复测" : "NEW LAYOUT / 新布局测试"} · SEED 0`;
  $("episodeTitle").textContent = `${episode.variant === "standard" ? "标准场景" : "杂物场景"} / 布局 ${episode.layout}`;
  $("episodeOutcome").textContent = `${episode.success ? "任务成功" : "任务未成功"} · ${episode.score} 分`;
  $("episodeOutcome").className = `badge ${episode.success ? "good" : "warn"}`;
  $("episodeStep").textContent = `机器人 ${physical(episode.steps)} · ${episode.steps}/500 步`;
  $("episodeWall").textContent = `回合墙钟 ${(episode.wallSeconds/60).toFixed(1)} min`;
  $("episodeCommands").textContent = `${episode.commands.length} 条命令`;
  $("scrub").max = episode.commands.length;
  $("downloadLog").href = episode.sources.commands;
  $("downloadResult").href = episode.sources.result;
  $("downloadSession").href=episode.sessionUrl;
  $("downloadActions").href=episode.sources.actions;
  const s=episode.sessionSummary;
  $("executionLayers").innerHTML=`<div><span>Codex 发出的 shell</span><strong>${s.shellCalls}<small> 次</small></strong></div><b>→</b><div><span>执行器收到的 robo</span><strong>${s.dispatchedCommands}<small> 条</small></strong></div><b>→</b><div><span>下发的控制 action</span><strong>${s.actionRecords}<small> 步</small></strong></div>`;
  renderEpisodes();
  renderTimeline();
  renderCurrent();
  loadVideo();
  renderSession();
  loadSession(episode);
}

function applyView(){
  $("robotFlow").hidden=state.view!=="robot";
  $("sessionFlow").hidden=state.view!=="session";
  $("inspectorTitle").textContent=state.view==="session"?"会话事件详情":"操作详情";
  document.querySelectorAll("[data-view]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.view===state.view)));
}

async function loadSession(episode){
  try{
    if(!sessionCache.has(episode.id)){
      const response=await fetch(episode.sessionUrl);
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      sessionCache.set(episode.id,await response.json());
    }
    if(state.episode.id!==episode.id)return;
    state.session=sessionCache.get(episode.id);
    const requested=state.pendingEpisode===episode.id?state.session.events.find(e=>e.id===state.pendingEvent):null;
    if(requested){
      state.selectedEvent=requested.id;state.frame=requested.frame;
      if(video.readyState>=1 && !state.loadingVideo)video.currentTime=Math.min(state.frame+.04,video.duration-.01);
    }
    state.pendingEvent=null;
    renderSession();renderCurrent();
  }catch(error){
    if(state.episode.id===episode.id)$("sessionEvents").textContent=`会话载入失败：${error.message}`;
  }
}

function currentEvent(){
  if(!state.session)return null;
  if(state.selectedEvent)return state.session.events.find(e=>e.id===state.selectedEvent);
  if(state.frame===0)return state.session.events[0];
  return state.session.events.find(e=>e.commandIndices?.includes(state.frame))||state.session.events[0];
}

function renderSession(){
  if(!state.session){$("sessionEvents").textContent="正在载入这一回合的实际会话…";$("sessionCount").textContent="";return;}
  const session=state.session,s=session.summary;
  $("sessionCount").textContent=`${s.messages} 条消息 · ${s.shellCalls} 次 shell`;
  $("sessionCoverage").textContent=`下列内容来自原始 CLI 记录。每次 shell 中的 robo 请求均为一条命令；模型 API 请求分组和独立看图调用未被记录，不能据此推算模型推理次数。${s.clientRejected?`另有 ${s.clientRejected} 次请求被客户端以回合已结束拒绝，已保留在流程中。`:""}`;
  $("sessionEvents").innerHTML=session.events.map((event,index)=>{
    const linked=event.commandIndices?.map(i=>state.episode.commands[i-1])||[];
    const summary=event.kind==="shell"?event.script.split("\n")[0]:event.kind==="completion"?"本次 Codex 会话结束，保留原生评分和最终回复。":event.kind==="task"?event.text.split("\n")[0]:event.text;
    const actionText=linked.length?`robo #${linked[0].index} · ${linked[0].actionCount} 个控制 action`:event.clientRejected?"客户端拒绝 · 未送入执行器":event.kind==="shell"?"未新增机器人命令":"";
    return `<button class="session-event ${event.kind}" data-event="${esc(event.id)}" aria-label="会话事件 ${index+1}：${esc(event.title)}"><span class="event-order">${String(index+1).padStart(2,"0")}</span><span class="event-body"><span class="event-heading"><strong>${esc(event.title)}</strong><small>${event.kind==="shell"?`shell #${event.shellNumber}`:event.line?`日志行 ${event.line}`:"prompt"}</small></span><span class="event-preview ${event.kind==="shell"?"mono":""}">${esc(summary)}</span>${actionText?`<span class="event-relation">${esc(actionText)}</span>`:""}</span></button>`;
  }).join("");
  markSessionSelection();
}

function markSessionSelection(){
  const event=currentEvent();
  document.querySelectorAll("[data-event]").forEach(b=>b.setAttribute("aria-current",String(b.dataset.event===event?.id)));
}

function selectEvent(id){
  const event=state.session?.events.find(e=>e.id===id);
  if(!event)return;
  video.pause();
  state.selectedEvent=id;state.frame=event.frame;state.actionStep=null;
  if(video.readyState>=1)video.currentTime=Math.min(state.frame+.04,video.duration-.01);
  renderCurrent();
}

function renderDetail(){
  $("actionPanel").innerHTML="";
  const event=state.view==="session"?currentEvent():null;
  if(state.view==="robot" || !event){renderRobotDetail();renderActions(state.episode.commands[state.frame-1]);return;}
  const position=state.session.events.findIndex(e=>e.id===event.id)+1;
  $("commandIndex").textContent=`事件 ${position} / ${state.session.events.length}`;
  const source=event.line?`agent-events.jsonl · ${event.startLine?`开始行 ${event.startLine} → `:""}完成行 ${event.line}`:"agent-prompt.txt · 原始任务提示";
  const cliOutput=event.stdout?`<details class="raw-receipt" ${event.stdout.length<3000?"open":""}><summary>shell 原始输出 · ${event.stdout.length.toLocaleString()} 字符</summary><pre>${esc(event.stdout)}</pre></details>`:`<p class="detail-description">CLI 事件未保存此调用的 stdout 文本。${event.commandIndices?.length?"下面的机器人回执来自独立的 commands.jsonl。":""}</p>`;
  let prefix=`<p class="event-source">${esc(source)}</p>`;
  if(event.kind==="shell"){
    const parallel=state.session.overlaps.filter(g=>g.ids.includes(event.id));
    prefix+=`<span class="badge ${event.exitCode===0?"good":"warn"}">shell #${event.shellNumber} · exit ${event.exitCode}</span><h3>${esc(event.title)}</h3><p class="detail-description">${event.clientRejected?"这条请求由 Codex 发出，但客户端发现回合已结束，没有送入机器人执行器。":event.commandIndices.length?`本次 shell 发出了 1 条 robo 命令，对应操作 #${event.commandIndices[0]}。底层控制序列由工具、IK 与轨迹计时生成。`:event.category==="python"?"这是智能体实际编写并执行的 Python 代码，输出参与后续决策；它本身没有新增机器人命令。":"此 shell 调用用于读取文件或状态，没有新增机器人命令。"}</p>${parallel.length?`<p class="receipt-note">CLI 记录显示它与另一条 shell 调用的开始—结束区间重叠。这里保留并发证据，不把它解释成运动 action chunk。</p>`:""}<p class="detail-label">CODEX 实际发出的 SHELL</p><pre class="command-code">${esc(event.command)}</pre>${cliOutput}`;
    if(event.commandIndices.length){
      renderRobotDetail();
      const linked=state.episode.commands[event.commandIndices[0]-1];
      $("commandDetail").innerHTML=prefix+`<button class="action-jump" data-jump-actions>查看实际下发的 ${linked.actionCount} 个控制 actions ↓</button>`+'<div class="inspector-divider">对应的机器人执行</div>'+$("commandDetail").innerHTML;
      $("commandIndex").textContent=`事件 ${position} · robo #${event.commandIndices[0]}`;
      renderActions(state.episode.commands[event.commandIndices[0]-1]);
    }else $("commandDetail").innerHTML=prefix;
  }else if(event.kind==="task"){
    $("commandDetail").innerHTML=prefix+`<span class="badge">用户任务 + recipe + notes</span><h3>任务如何交给 Codex</h3><p class="detail-description">保留当时实际输入的任务提示。工具手册由会话中的 cat INTERFACE.md 调用读取。</p><pre class="session-text">${esc(event.text)}</pre><a class="source-link" href="${esc(state.episode.sources.manual)}" download>下载当时的完整工具手册</a>`;
  }else if(event.kind==="completion"){
    $("commandDetail").innerHTML=prefix+`<span class="badge">turn.completed</span><h3>本次 CLI 会话结束</h3><p class="detail-description">一个 CLI turn 可以包含多次模型与工具往返，不等于一次模型 API 调用。原生评分：${state.episode.success?"成功":"未成功"}，${state.episode.score} 分。</p><pre class="session-text">${esc(JSON.stringify(event.usage,null,2))}</pre>`;
  }else{
    $("commandDetail").innerHTML=prefix+`<span class="badge">${event.kind==="message"?"Codex 消息原文":"运行时提示原文"}</span><h3>${esc(event.title)}</h3><pre class="session-text">${esc(event.text)}</pre><p class="detail-description">这条消息本身不驱动机械臂。画面对应此时最近完成的机器人命令 #${event.frame}。</p>`;
  }
}

function renderActions(command){
  if(!command || !state.session)return;
  const actions=state.session.actions.filter(a=>a.step>command.before && a.step<=command.after);
  if(!actions.length){$("actionPanel").innerHTML='<div class="inspector-divider">底层控制 actions</div><p class="detail-description">0 个控制 action：这条命令没有推进物理仿真。</p>';return;}
  if(!actions.some(a=>a.step===state.actionStep))state.actionStep=actions[0].step;
  const baseline=state.episode.frames[0].joints;
  const delta=(action,arm)=>Math.max(...action.action[`${arm}_arm_joint_state`].map((v,i)=>Math.abs(v-baseline[arm][i])));
  const top=Math.max(.1,Math.ceil(Math.max(...actions.flatMap(a=>[delta(a,"left"),delta(a,"right")]))*10)/10);
  const x=step=>32+(step-actions[0].step)/Math.max(1,actions.length-1)*212;
  const y=value=>112-value/top*78;
  const points=arm=>actions.map(a=>`${x(a.step).toFixed(2)},${y(delta(a,arm)).toFixed(2)}`).join(" ");
  $("actionPanel").innerHTML=`<div class="inspector-divider">实际下发的控制 actions</div><p class="detail-description">本条 robo 命令对应 <strong>${actions.length}</strong> 个控制步（25 Hz）。每步包含左/右臂各 6 个关节目标和两路夹爪指令。它们由执行器生成，不是模型直接预测的关节向量。</p><div class="action-chart"><h4>关节目标相对初始位置的最大变化</h4><div class="chart-legend"><span><i></i>左臂</span><span><i class="right"></i>右臂</span></div><svg viewBox="0 0 270 145" role="img" aria-label="本条命令中左右臂关节目标变化，横轴控制步，纵轴弧度"><text x="7" y="15">rad</text><path d="M32 29V112H249" class="chart-axis"/><path d="M32 73H249 M32 34H249" class="chart-grid"/><text x="5" y="115">0</text><text x="3" y="37">${top.toFixed(1)}</text><text x="28" y="130">${actions[0].step}</text><text x="225" y="130">${actions.at(-1).step}</text><text x="116" y="142">控制步</text><polyline points="${points('left')}" class="action-line"/><polyline points="${points('right')}" class="action-line right"/><line id="actionMarker" x1="${x(state.actionStep)}" x2="${x(state.actionStep)}" y1="29" y2="112" class="action-marker"/></svg><p>来源：本回合 actions.jsonl；显示目标值，不代表实测轨迹。滑块查看控制数值，上方视频保持命令结束时的观测。</p></div><label class="action-slider">查看控制步 <strong id="actionStepLabel"></strong><input id="actionScrub" aria-label="选择底层控制 action" type="range" min="${actions[0].step}" max="${actions.at(-1).step}" step="1" value="${state.actionStep}"></label><div id="actionValues"></div><details class="raw-receipt"><summary>这个控制步的原始 JSON</summary><pre id="actionJson"></pre></details>`;
  const update=()=>{
    const action=actions.find(a=>a.step===state.actionStep);
    $("actionStepLabel").textContent=`${action.step} / ${state.episode.steps} · ${physical(action.step)}`;
    $("actionMarker").setAttribute("x1",x(action.step));$("actionMarker").setAttribute("x2",x(action.step));
    $("actionValues").innerHTML=['left','right'].map(arm=>`<div class="action-arm"><strong>${arm==='left'?'左臂':'右臂'}目标（rad）</strong><code>${action.action[`${arm}_arm_joint_state`].map(v=>v.toFixed(4)).join(', ')}</code><span>夹爪指令（0 闭 / 1 开）：${action.action[`${arm}_ee_joint_state`][0]}</span></div>`).join('');
    $("actionJson").textContent=JSON.stringify(action,null,2);
  };
  $("actionScrub").addEventListener('input',e=>{state.actionStep=Number(e.target.value);update();});update();
}

document.querySelectorAll("[data-view]").forEach(b=>b.addEventListener("click",()=>{state.view=b.dataset.view;applyView();renderDetail();writeHash();}));
$("sessionEvents").addEventListener("click",event=>{const button=event.target.closest("[data-event]");if(button)selectEvent(button.dataset.event);});
$("commandDetail").addEventListener("click",event=>{if(event.target.closest("[data-jump-actions]"))$("actionPanel").scrollIntoView({block:"start"});});

$("episodeList").addEventListener("click",event=>{const b=event.target.closest("[data-episode]");if(b)selectEpisode(b.dataset.episode);});
$("timeline").addEventListener("click",event=>{const b=event.target.closest("[data-frame]");if(b)setFrame(Number(b.dataset.frame));});
document.querySelectorAll("[data-camera]").forEach(b=>b.addEventListener("click",()=>{state.camera=b.dataset.camera;loadVideo();writeHash();}));
$("variantFilter").addEventListener("change",e=>{state.variant=e.target.value;renderEpisodes();});
$("outcomeFilter").addEventListener("change",e=>{state.outcome=e.target.value;renderEpisodes();});
$("previous").addEventListener("click",()=>setFrame(state.frame-1));
$("next").addEventListener("click",()=>setFrame(state.frame+1));
$("scrub").addEventListener("input",e=>setFrame(Number(e.target.value)));
$("speed").addEventListener("change",e=>{video.playbackRate=Number(e.target.value);});
$("play").addEventListener("click",async()=>{
  if (state.loadingVideo) return;
  if (!video.paused) {video.pause();return;}
  if (video.ended || video.currentTime >= video.duration-.1) setFrame(0);
  try {await video.play();} catch(error) {$("loading").textContent=`视频播放失败：${error.message}`;$("loading").hidden=false;}
});
$("fullscreen").addEventListener("click",()=>video.requestFullscreen?.());
document.addEventListener("fullscreenchange",()=>{video.controls=document.fullscreenElement===video;});
video.addEventListener("loadedmetadata",()=>{video.currentTime=Math.min(state.frame+.04,video.duration-.01);video.playbackRate=Number($("speed").value);state.loadingVideo=false;});
video.addEventListener("timeupdate",()=>{
  if (state.loadingVideo || !state.episode) return;
  const frame=Math.min(state.episode.commands.length,Math.floor(video.currentTime));
  if(frame!==state.frame){state.frame=frame;state.selectedEvent=null;state.actionStep=null;renderCurrent(!video.paused);}
});
video.addEventListener("play",()=>{$("play").textContent="暂停";$("play").setAttribute("aria-label","暂停观测回放");});
video.addEventListener("pause",()=>{$("play").textContent="播放";$("play").setAttribute("aria-label","播放观测回放");});
video.addEventListener("error",()=>{$("loading").textContent="视频加载失败，请检查本地服务是否仍在运行。";$("loading").hidden=false;});
document.addEventListener("keydown",event=>{
  if(!state.episode || ["INPUT","SELECT","TEXTAREA","BUTTON","SUMMARY"].includes(event.target.tagName))return;
  if(event.key==="ArrowRight"){event.preventDefault();setFrame(state.frame+1);}
  if(event.key==="ArrowLeft"){event.preventDefault();setFrame(state.frame-1);}
});
window.addEventListener("hashchange",()=>{
  if(!state.data)return;
  const params=new URLSearchParams(location.hash.slice(1));
  const id=params.get("episode");
  if(!state.data.episodes.some(e=>e.id===id))return;
  if(cameraNames[params.get("camera")])state.camera=params.get("camera");
  state.view=params.get("view")==="robot"?"robot":"session";
  state.pendingEvent=params.get("event");
  state.pendingEpisode=id;
  applyView();selectEpisode(id,Number(params.get("frame")||0));
});

async function start() {
  const response=await fetch("/api/manifest");
  if(!response.ok)throw new Error(`实验记录 HTTP ${response.status}`);
  state.data=await response.json();
  const s=state.data.summary;
  $("successStat").innerHTML=`${s.successes}<small> / ${s.episodes}</small>`;
  $("scoreStat").innerHTML=`${s.meanScore}<small> / 100</small>`;
  $("commandStat").textContent=s.commands;
  $("toolCounts").innerHTML=["cloth_landmarks","surface_transfer","edge_transfer","table_xy","table_plane"].map(name=>`<div><code>${name}</code><strong>${s.commandCounts[name] || 0}<small> 次</small></strong><small>${s.commandCoverage[name] || 0} 个回合</small></div>`).join("");
  $("sourceDate").textContent=`实验记录：${state.data.sourceDate} · 仅回放，不运行机器人`;
  $("batchSource").textContent=`数据批次：${state.data.batch}。原始命令与评分可以从上方直接下载。`;
  const params=new URLSearchParams(location.hash.slice(1));
  state.pendingEvent=params.get("event");
  state.pendingEpisode=params.get("episode");
  if(cameraNames[params.get("camera")])state.camera=params.get("camera");
  state.view=params.get("view")==="robot"?"robot":"session";
  applyView();
  const requested=params.get("episode");
  selectEpisode(state.data.episodes.some(e=>e.id===requested)?requested:"standard-l6",Number(params.get("frame") || 0));
  $("app").hidden=false;
  $("loading").hidden=true;
}
start().catch(error=>{$("loading").textContent=`无法读取回放：${error.message}`;});
