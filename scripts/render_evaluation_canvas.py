#!/usr/bin/env python3
"""Render a self-contained analytical canvas from archived evaluation evidence."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = r'''
import { Stack, Row, H1, H2, Text, Table, BarChart, Button, useState, useHostTheme } from "cursor/canvas";

const data = __DATA__;
const labels: Record<string, string> = {
  cover_blocks: "遮盖积木", deposit_coin: "投币", put_bottles_into_dustbin: "瓶子入桶",
  stack_blocks: "叠积木"
};
const n = (v: number) => Number(v.toFixed(2)).toString();

export default function Evaluation() {
  const theme = useHostTheme();
  const [expanded, setExpanded] = useState(false);
  const valid = data.report.episodes.filter((r: any) => r.status === "native_completed");
  const reference = data.report.public_task_references;
  const completedTasks = Object.entries(data.report.task_summaries)
    .filter(([, s]: any) => s !== null) as [string, any][];
  return (
    <Stack gap={22} style={{ padding: 24, maxWidth: 1100, margin: "0 auto", color: theme.text.primary }}>
      <Stack gap={6}>
        <Text size="small" tone="secondary">RoboRAG 实证验证 · 2026-10-20 · 静态证据快照</Text>
        <H1>与 RoboICL 公开结果对照</H1>
        <Text weight="semibold">
          {data.executionStop ? `已按用户要求停止：${valid.length} 个完整回合；1 个中断、3 个未启动。` :
           data.report.complete ? "九回合已完成：结论仅适用于这三个任务和三个布局。" :
           `评测尚未完成：已有 ${valid.length} 个完整回合；未完成回合不计分。`}
        </Text>
        <Text tone="secondary">
          当前检索算法冻结；每任务同一条公开训练示范、12 个长期片段、12 个短期片段，
          seed 0，standard 布局 2 / 3 / 4，xhigh。CLI 传输与论文 API 有差异，属于外部基线比较。
        </Text>
        <Text size="small" tone="tertiary">结果数据更新时间：{data.report.updated_at_utc}</Text>
      </Stack>

      {valid.length > 0 && <Stack gap={10}>
        <H2>已完成回合：任务进度（0–100 分）</H2>
        <Text size="small" tone="secondary">横轴：任务与布局；纵轴：原生任务进度（分）。进度分不等于成功率。</Text>
        <BarChart categories={valid.map((r: any) => `${labels[r.task]} L${r.layout}`)}
          series={[
            { name: "RoboRAG 本轮实测", data: valid.map((r: any) => r.roborag_progress), tone: "info" },
            { name: "RoboICL 公开同布局", data: valid.map((r: any) => r.public_progress), tone: "neutral" }
          ]} yMin={0} yMax={100} valueSuffix=" 分" showValues height={280} />
        <Text size="small" tone="tertiary">
          来源：本地原生 _result.json 与 RoboICL 2026-09-28 公开快照。仅纳入完整回合；
          本地测试日期 2026-10-20。小样本不支持显著性或全任务优势结论。
        </Text>
        <Table headers={["任务 / 布局", "RoboRAG 进度", "公开进度", "差值（分）", "RoboRAG 完整成功", "完成调用数"]}
          rows={valid.map((r: any) => [ `${labels[r.task]} / ${r.layout}`, n(r.roborag_progress),
            n(r.public_progress), n(r.difference), r.roborag_success ? "是" : "否",
            r.evidence.completed_model_calls ])} />
      </Stack>}

      {completedTasks.length > 0 && <Stack gap={10}>
        <H2>任务均值与完整成功率</H2>
        <Table headers={["任务", "完成回合 / 计划", "RoboRAG 均分", "同布局公开均分", "RoboRAG 成功率", "公开成功率"]}
          rows={completedTasks.map(([task, s]) => [
            labels[task], `${s.completed_episodes} / 3`, n(s.roborag_progress_mean),
            n(s.public_same_layout_progress_mean), `${n(s.roborag_success_rate * 100)}%`,
            `${n(s.public_same_layout_success_rate * 100)}%`
          ])} />
        <Text size="small" tone="tertiary">
          均值只覆盖对应任务已完成的相同布局；不足 3 回合时属于中间结果。
          各任务完成数相同且九回合全部结束后才报告本轮总均分。
        </Text>
      </Stack>}

      <Stack gap={10}>
        <H2>Baseline 的两个统计口径</H2>
        <Table headers={["任务", "类别", "公开完整样本数", "公开全布局均分", "本轮布局 2 / 3 / 4 均分"]}
          rows={reference.map((r: any) => [
            labels[r.task], data.publicTasks.find((t: any) => t.task === r.task)?.category,
            r.published_full_episode_count, n(r.published_all_layouts_progress_mean),
            n(r.selected_layout_progress_mean)
          ])} />
        <Text size="small" tone="tertiary">
          单位：原生进度分（0–100）。来源：
          <a href={data.source} style={{ color: theme.text.link }}>RoboICL 公开逐布局数据</a>，
          快照 2026-09-28。全布局分数不能直接代替所选三个布局的 Baseline。
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>已有叠积木试点：没有稳定的检索收益</H2>
        <Table headers={["角色 / 布局", "RoboRAG 进度 / 成功", "固定上下文进度 / 成功", "RoboRAG 步数 / 调用", "固定上下文步数 / 调用"]}
          rows={[
            ["开发 / 0", "100 / 是", "100 / 是", "195 / 20", "240 / 24"],
            ["验证 / 1", "100 / 是", "100 / 是", "302 / 31", "237 / 24"]
          ]} />
        <Text size="small" tone="tertiary">
          来源：results/pilot-current.json，2026-10-16。两种方法均成功；检索在开发场景更省调用，
          验证场景反而更多。stack_blocks 不在 RoboICL 30 任务主表中，且旧试点是 8+4、medium，
          不与本轮合并。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>证据与限制</H2>
        <Text>9 / 9 布局哈希与官方锁定版本一致；3 / 3 示范源哈希一致；检索算法没有在本轮调参。</Text>
        <Text tone="secondary">
          公开资料包含 880 个完整回合和 49 个部分回合；部分回合不进入本轮比较。
          完整资产树和服务端模型快照仍未获得一致性认证。并行仿真的墙钟时间不用于论文延迟比较。
        </Text>
        {data.report.excluded_infrastructure_attempts.length > 0 && <Table
          headers={["排除的基础设施记录", "原因（不作为机器人失败分数）"]}
          rows={data.report.excluded_infrastructure_attempts.map((r: any) => [r.path, r.reason])} />}
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between">
          <H2>RoboICL 30 任务公开主表</H2>
          <Button onClick={() => setExpanded(!expanded)}>{expanded ? "收起" : "展开"}</Button>
        </Row>
        {expanded && <Table headers={["任务", "类别", "公开进度均分", "完整回合数", "评测状态"]}
          rows={data.publicTasks.map((t: any) => [t.task, t.category, n(t.published_score_0_100),
            t.official_episode_count, t.official_episode_count ? "完整评测" : "论文零值占位"])} />}
        <Text size="small" tone="tertiary">
          论文 Overall 为 50.64，是 30 个任务均分的非加权平均，包含 7 个没有完整评测的零值占位；
          不可将其当作 30 个任务均已完成，也不是本轮 RoboRAG 的对照总分。
        </Text>
      </Stack>
    </Stack>
  );
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results/roboicl-public-comparison-20261020.json")
    parser.add_argument("--stop-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    public = json.loads((ROOT / "results/roboicl-public-baseline.json").read_text())
    data = {"report": report, "source": public["source_url"],
            "executionStop": json.loads(args.stop_record.read_text()) if args.stop_record else None,
            "publicTasks": [{k: v for k, v in t.items() if k != "layouts"} for t in public["tasks"]]}
    args.output.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    print(args.output)


if __name__ == "__main__":
    main()
