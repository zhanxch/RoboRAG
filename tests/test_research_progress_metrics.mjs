import {test} from 'node:test';
import assert from 'node:assert/strict';
import {groupKey, summarize, totalTokens} from '../web/research-progress/metrics.js';

test('token subsets never add twice and unknown differs from zero',()=>{
  assert.equal(totalTokens({input_tokens:1000,output_tokens:100,cached_tokens:800,reasoning_tokens:80,total_tokens:9999}),1100);
  assert.equal(totalTokens({input_tokens:null,output_tokens:0,total_tokens:null}),null);
  assert.equal(totalTokens({input_tokens:0,output_tokens:0}),0);
});
test('failures count; interrupted and unknown outcomes do not enter denominator',()=>{
  const s=summarize([{status:'completed',success:true,completion:100,wall_seconds:10,total_tokens:100,usage:'complete'},
    {status:'completed',success:false,completion:20,wall_seconds:20,total_tokens:200,usage:'complete'},
    {status:'interrupted',success:null,completion:null,wall_seconds:1000,total_tokens:50,usage:'partial'},
    {status:'planned',success:null,completion:null,wall_seconds:null,total_tokens:null,usage:'unknown'}]);
  assert.equal(s.successRate,50);assert.equal(s.scored,2);assert.equal(s.completion,60);assert.equal(s.wall,15);
  assert.equal(s.tokens,350);assert.equal(s.lowerBound,true);assert.equal(s.tokenN,3);
});
test('empty summary has no fabricated zeros for unknown metrics',()=>{
  const s=summarize([]);assert.equal(s.successRate,null);assert.equal(s.wall,null);assert.equal(s.tokens,null);assert.equal(s.completion,null);
});
test('group isolates task split protocol model conditions and MCP versions',()=>{
  const base={task:'stack',condition:'standard',protocol:'v1',model:'model',effort:'medium',split:'test',transport:'mcp',bank_version:'a',code_version:'b',mcp_server:'server',mcp_version:'v1'};
  for(const field of Object.keys(base))assert.notEqual(groupKey(base),groupKey({...base,[field]:'different'}),field);
  assert.equal(groupKey(base),groupKey({...base,method:'baseline',layout:'1',seed:'2'}));
});
