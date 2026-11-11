"""Read-only round display and numbered evidence from controller observations."""

import json
import sys
import time
from collections import Counter
from html import escape
from urllib.parse import quote

import numpy as np

from piper_sim.l2_progress import L2Progress

LABELS = {
    "red": "红色",
    "yellow": "黄色",
    "green": "绿色",
    "orange_brown": "橙棕色",
    "yellow_green": "黄绿色",
    "banana": "香蕉",
    "potato": "土豆",
    "cucumber": "黄瓜",
    "pear": "梨",
    "bell_pepper": "彩椒",
    "persimmon": "柿子",
}
STAGES = {
    "connect": "等待工位反馈",
    "scan": "扫描目标区",
    "observe": "寻找和识别目标",
    "planning": "规划抓放",
    "approach": "移向目标",
    "descend": "下降至抓取位",
    "close": "闭合夹爪",
    "check_grip": "检查夹爪反馈",
    "test_lift": "试提",
    "lift": "抬升",
    "reorient_transport": "调整搬运朝向",
    "transport": "搬运",
    "place": "下降至放置位",
    "release": "释放物体",
    "retreat": "退避",
    "reobserve_place": "重新观察目标区",
    "verify": "检查入区和稳定时间",
    "recover_lift": "退避后重试",
    "recheck_grip": "复查夹爪反馈",
    "done": "整理本轮记录",
    "failed": "结束本轮",
}


class FieldProcess:
    def __init__(
        self,
        output,
        task,
        mapping,
        limit_s,
        *,
        probe=False,
        title=None,
        dashboard=None,
        mask_observer=None,
    ):
        self.output, self.task, self.mapping = output, task, mapping
        self.limit_s, self.probe = limit_s, probe
        self.title = title or ("L1 · 精准抓取" if task == "l1" else "L2 · 视觉分拣")
        self.dashboard = dashboard
        self.mask_observer = mask_observer
        self.current_plan = None
        self.total = 3 if task == "l1" else 6
        self.path = output / "process.html"
        self.log = output / "process_events.jsonl"
        self.rows = {}
        self.identity = L2Progress(self.total, color_mapping=mapping) if task == "l2" else None
        self.bindings = {}
        self.candidates = {}
        self.candidate_frames = {}
        self.active = None
        self.frame = self.frame_time = self.labelled_frame = None
        self.boxes = []
        self.stage_name = "connect"
        self.clock = None
        self.elapsed = None
        self.outcome = None
        self.failure = None
        self.last_render = -float("inf")
        self.available = True
        if task == "l1":
            for number, label in enumerate(mapping, start=1):
                self._add_row(label, number, label)
        self._emit("display_opened", mode="probe" if probe else "round")
        self._publish(force=True)

    def _add_row(self, key, number, label):
        return self.rows.setdefault(
            key,
            {
                "target_id": f"{self.task.upper()}-{number:02d}",
                "category": label,
                "destination": self.mapping[label],
                "identified": False,
                "attempt": 0,
                "grasp": "待抓取",
                "placement": "待放置",
            },
        )

    def _disable(self, exc):
        self.available = False
        print(f"过程展示写入失败 {exc}", file=sys.stderr, flush=True)

    def _emit(self, name, row=None, **details):
        if not self.available:
            return
        entry = {
            "event": name,
            "round_elapsed_s": None if self.clock is None else round(self.clock(), 3),
            "source_frame": self.frame,
            "observation_time_s": self.frame_time,
            **details,
        }
        if row is not None:
            entry.update(row)
        try:
            with self.log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._disable(exc)

    def begin(self, clock):
        self.clock = clock
        self._emit("round_started")
        self._publish(force=True)

    def observe(self, frame, timestamp, blocks, *, observation=None):
        self.frame, self.frame_time = frame, float(timestamp)
        if self.task == "l1" and (self.clock is not None or self.probe):
            counts = Counter(color for color, _ in blocks)
            boxes_before = self.boxes
            self.boxes = []
            for color, target in blocks:
                if color in self.rows and counts[color] == 1:
                    self._identified(self.rows[color], color, target)
            if self.boxes:
                self.labelled_frame = self.frame
            else:
                self.boxes = boxes_before
            self.show_masks(observation, [(color, target.mask) for color, target in blocks])
        self._publish()

    def _identified(self, row, category, target):
        y, x = np.where(target.mask)
        if not len(x):
            return
        height, width = target.mask.shape
        box = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
        self.boxes.append((row["target_id"], row["destination"], box, width, height))
        changed = row["identified"] and row["category"] != category
        if not row["identified"] or changed:
            row.update(identified=True, category=category, destination=self.mapping[category])
            self._emit(
                "classification_updated" if changed else "identified",
                row,
                bbox_xyxy_px=box,
                center_base_m=target.center.tolist(),
            )
            if not changed:
                print(
                    f"{row['target_id']}  {LABELS[category]} → {LABELS[row['destination']]}区",
                    flush=True,
                )

    def fruits(self, proposals, *, allow_new, observation=None):
        self.show_masks(observation, [(item.color, item.target.mask) for item in proposals])
        self.candidates = {}
        self.candidate_frames = {
            item.key: (self.frame, self.frame_time)
            for item in proposals
            if item.color in self.mapping
        }
        matches = []
        for proposal in proposals:
            label = proposal.color
            if label not in self.mapping:
                continue
            try:
                item = self.identity.associate(label, proposal.target)
                if item is None and allow_new:
                    item = self.identity.identify(label, self.mapping[label], proposal.target)
            except ValueError:
                self._emit("identity_unresolved", candidate_id=proposal.key, category=label)
                continue
            if item is not None:
                matches.append((proposal, item))
        counts = Counter(item["id"] for _, item in matches)
        boxes_before = self.boxes
        self.boxes = []
        for proposal, item in matches:
            key = item["id"]
            if counts[key] != 1:
                self._emit(
                    "identity_unresolved", candidate_id=proposal.key, category=proposal.color
                )
                continue
            row = self._add_row(key, int(key.rsplit("_", 1)[1]), proposal.color)
            self.candidates[proposal.key] = (key, self.frame, self.frame_time)
            self._identified(row, proposal.color, proposal.target)
        if self.boxes:
            self.labelled_frame = self.frame
        else:
            self.boxes = boxes_before
        self._publish()

    def show_masks(self, observation, masks):
        if observation is None:
            return
        candidates = [(label, self.mapping.get(label), mask) for label, mask in masks]
        if self.dashboard is not None:
            self.dashboard.publish_segmentation(observation, candidates)
        if self.mask_observer is not None:
            self.mask_observer(observation, [item for item in candidates if item[1] is not None])

    def plan(self, category, pick, place, timestamp):
        self.current_plan = {
            "target_id": self.rows.get(self.active, {}).get("target_id"),
            "category": category,
            "destination": self.mapping.get(category),
            "pick_m": np.asarray(pick).tolist(),
            "place_m": np.asarray(place).tolist(),
            "source_s": None if timestamp is None else float(timestamp),
        }
        self._emit("grasp_place_plan", **self.current_plan)
        self._publish(force=True)

    def bind_fruit(self, item, proposal):
        candidate = self.candidates.get(proposal.key)
        key = self.bindings.get(item["id"])
        if key is None and candidate is not None:
            key = candidate[0]
        if key is None and proposal.key in self.candidate_frames:
            # A scan may fill the display with provisional candidates. The
            # controller's selected instance can take a stale, unbound row.
            current = {value[0] for value in self.candidates.values()}
            bound = set(self.bindings.values())
            spare = next(
                (old for old in self.identity.items if old["id"] not in bound | current),
                None,
            )
            if spare is not None:
                key = spare["id"]
                frame, timestamp = self.candidate_frames[proposal.key]
                candidate = key, frame, timestamp
                spare.update(item, id=key, anchors=[point.copy() for point in item["anchors"]])
                row = self._add_row(key, int(key.rsplit("_", 1)[1]), proposal.color)
                previous_category = row["category"]
                row.update(
                    category=proposal.color, destination=item["destination"], identified=True
                )
                self._emit(
                    "candidate_replaced",
                    row,
                    candidate_id=proposal.key,
                    previous_category=previous_category,
                    source_frame=frame,
                    observation_time_s=timestamp,
                )
        if key is None or any(
            value == key and name != item["id"] for name, value in self.bindings.items()
        ):
            self.active = None
            self._emit(
                "attempt_identity_unresolved", controller_id=item["id"], candidate_id=proposal.key
            )
            return
        self.bindings[item["id"]] = key
        evidence = {"candidate_id": proposal.key}
        if candidate is not None and candidate[0] == key:
            evidence.update(source_frame=candidate[1], observation_time_s=candidate[2])
        self._attempt(key, item["attempts"], **evidence)

    def _attempt(self, key, number, **details):
        self.active = key
        row = self.rows[key]
        row.update(attempt=number, grasp="抓取中", placement="待放置")
        self._emit("attempt_started", row, **details)

    def sync_fruits(self, items):
        identities = {item["id"]: item for item in self.identity.items}
        for item in items:
            key = self.bindings.get(item["id"])
            if key is None:
                continue
            identities[key].update(
                status=item["status"],
                center=item["center"].copy(),
                anchors=[p.copy() for p in item["anchors"]],
            )
            if item["status"] == "verified":
                self._placed(
                    key, controller_id=item["id"], stable_at_s=item["placement_verified_at_s"]
                )
            elif item["status"] in {"retry", "exhausted"} and self.rows[key]["placement"] not in {
                "待重试",
                "复查待确认",
            }:
                self._retry(key, "复查待确认", item["last_error"])
        self._publish()

    def _placed(self, key, **details):
        row = self.rows[key]
        if row["placement"] != "复查通过":
            row["placement"] = "复查通过"
            self._emit("placement_verified", row, **details)
        if self.active == key:
            self.active = None

    def _retry(self, key, status, reason):
        row = self.rows[key]
        if row["placement"] != status:
            row["placement"] = status
            self._emit("retry_or_review", row, reason=reason)
        if self.active == key:
            self.active = None

    def event(self, record):
        name = record["event"]
        key = record.get("color") if self.task == "l1" else self.bindings.get(record.get("id"))
        if key not in self.rows:
            return
        row = self.rows[key]
        if name == "pregrasp":
            self._attempt(key, record["attempt"])
        elif name == "holding_estimate":
            row["grasp"] = "试提保持"
            self._emit(
                "lift_estimate",
                row,
                source="gripper_feedback",
                source_frame=None,
                observation_time_s=None,
                official_lift_verified=False,
            )
        elif name == "grasp_failure":
            row["grasp"] = "抓取待确认"
            self._retry(key, "待重试", record.get("reason"))
        elif name in {
            "placement_failure",
            "placement_retry",
            "completion_revoked",
            "neighbor_displaced",
        }:
            self._retry(key, "复查待确认", record.get("reason", name))
        elif name == "placement_observation" and record["accepted"]:
            self._placed(key, stable_duration_s=record["stable_duration_s"])
        self._publish()

    def stage(self, name):
        self.stage_name = name
        row = self.rows.get(self.active)
        if row is not None:
            if name == "test_lift":
                row["grasp"] = "试提中"
            elif name in {"release", "reobserve_place", "verify"}:
                row["placement"] = "复查中"
        self._publish()

    def finish(self, success, failure):
        active = self.rows.get(self.active)
        if active is not None and not success:
            if active["grasp"] in {"抓取中", "试提中"}:
                active["grasp"] = "抓取待确认"
            if active["placement"] != "复查通过":
                active["placement"] = "复查待确认"
        self.outcome = ("只读检查完成" if self.probe else "本轮完成") if success else "本轮结束"
        self.failure = failure
        self.elapsed = None if self.clock is None else self.clock()
        self._emit("display_finished", success=success, failure=failure)
        self._publish(force=True)

    def _publish(self, *, force=False):
        now = time.monotonic()
        if not self.available or (not force and now - self.last_render < 0.6):
            return
        if self.dashboard is not None:
            self.dashboard.publish_task(
                {
                    "title": self.title,
                    "stage": self.outcome or STAGES.get(self.stage_name, self.stage_name),
                    "elapsed_s": self.elapsed
                    if self.outcome
                    else (self.clock() if self.clock else None),
                    "limit_s": self.limit_s,
                    "total": self.total,
                    "identified": sum(row["identified"] for row in self.rows.values()),
                    "verified": sum(row["placement"] == "复查通过" for row in self.rows.values()),
                    "rows": [row.copy() for row in self.rows.values()],
                    "active": self.rows.get(self.active, {}).get("target_id"),
                    "plan": self.current_plan,
                    "failure": self.failure,
                }
            )
        try:
            temp = self.path.with_suffix(".tmp")
            temp.write_text(self._html(), encoding="utf-8")
            temp.replace(self.path)
            self.last_render = now
        except OSError as exc:
            self._disable(exc)

    def _html(self):
        observed = sum(row["identified"] for row in self.rows.values())
        verified = sum(row["placement"] == "复查通过" for row in self.rows.values())
        elapsed = self.elapsed if self.outcome else (self.clock() if self.clock else None)
        clock = "待开始" if elapsed is None else f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
        clock += f" / {self.limit_s // 60:02d}:{self.limit_s % 60:02d}"
        stage = self.outcome or STAGES.get(self.stage_name, self.stage_name)
        frame = self.labelled_frame or self.frame
        if frame:
            camera = f'<img src="{quote(frame)}" alt="工位相机观测帧">'
            for key, zone, (x1, y1, x2, y2), width, height in self.boxes:
                style = f"left:{100 * x1 / width:.3f}%;top:{100 * y1 / height:.3f}%;width:{100 * (x2 - x1) / width:.3f}%;height:{100 * (y2 - y1) / height:.3f}%"
                camera += f'<div class="box {zone}" style="{style}"><b>{escape(key)}</b></div>'
        else:
            camera = '<p class="empty">等待相机观测</p>'
        lines = []
        for key, row in self.rows.items():
            category = LABELS[row["category"]] if row["identified"] else "待识别"
            zone = row["destination"]
            lines.append(
                f'<tr class="{"active" if self.active == key else ""}"><th>{row["target_id"]}</th>'
                f'<td>{category}</td><td><span class="zone {zone}">{LABELS[zone]}区</span></td>'
                f"<td>{row['grasp']}</td><td>{row['placement']}</td></tr>"
            )
        for _ in range(self.total - len(lines)):
            lines.append('<tr class="waiting"><td colspan="5">等待识别目标</td></tr>')
        refresh = '<meta http-equiv="refresh" content="1">' if self.outcome is None else ""
        failure = f'<p class="failure">{escape(self.failure)}</p>' if self.failure else ""
        frame_text = escape(frame or "等待采集")
        latest = f'<a href="{quote(self.frame)}">打开最近观测</a>' if self.frame else ""
        return PAGE.format(
            refresh=refresh,
            title=escape(self.title),
            stage=escape(stage),
            clock=clock,
            observed=observed,
            verified=verified,
            total=self.total,
            frame=frame_text,
            latest=latest,
            camera=camera,
            rows="".join(lines),
            failure=failure,
        )


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}<title>{title} · 过程展示</title><style>
:root{{--surface:#fff;--ink:#172a2c;--quiet:#566a6b;--rule:#d9e4e2;--accent:#176554}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf3f2;color:var(--ink);font:16px/1.45 "Inter","Noto Sans CJK SC",sans-serif}}
main{{max-width:1550px;margin:auto;padding:26px 30px}}header{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:22px}}
h1{{font-size:clamp(23px,2.3vw,34px);margin:0;font-weight:760;letter-spacing:-.03em}}.stage{{margin:6px 0 0;color:var(--accent);font-weight:650}}
.metrics{{display:flex;gap:24px}}.metrics div{{min-width:max-content}}.metrics b{{font-size:23px;font-variant-numeric:tabular-nums;display:block}}.metrics span,small{{color:var(--quiet)}}
.layout{{display:grid;grid-template-columns:minmax(0,1.08fr) minmax(500px,1fr);gap:20px;align-items:start}}
section{{border:1px solid var(--rule);border-radius:14px;background:var(--surface);padding:16px;min-width:0}}h2{{font-size:17px;margin:0}}.sub{{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px}}
.frame{{position:relative;background:#1b3033;border-radius:9px;overflow:hidden;min-height:130px}}.frame img{{width:100%;display:block}}.empty{{color:#e1eceb;text-align:center;padding:38px 0}}
.box{{position:absolute;border:2px solid var(--zone);pointer-events:none}}.box b{{position:absolute;top:0;left:0;padding:1px 5px;background:#1b292b;color:white;font:650 12px/1.4 monospace;white-space:nowrap}}
.red{{--zone:#e27c75;--chip:#fbe6e4;--tone:#84261f}}.yellow{{--zone:#f2cd53;--chip:#faefc6;--tone:#604e12}}.green{{--zone:#6ed4a3;--chip:#def3e9;--tone:#145e43}}
table{{width:100%;border-collapse:collapse;white-space:nowrap;font-size:15px}}th,td{{padding:12px 6px;text-align:left;border-bottom:1px solid var(--rule)}}thead th{{color:var(--quiet);font-weight:550;font-size:13px}}tbody th{{font-variant-numeric:tabular-nums}}
.active{{background:#e8f4f0}}.waiting{{color:var(--quiet)}}.zone{{color:var(--tone);background:var(--chip);padding:3px 7px;border-radius:5px}}
footer{{margin-top:15px;color:var(--quiet);font-size:13px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}}a{{color:var(--accent)}}.failure{{padding:10px;background:#fbe6e4;color:#84261f;overflow-wrap:anywhere}}
@media(max-width:1050px){{main{{padding:18px}}.layout{{grid-template-columns:minmax(0,1fr) minmax(410px,1fr);gap:12px}}th,td{{padding:10px 4px;font-size:13px}}.metrics{{gap:13px}}}}
@media(max-width:760px){{header{{align-items:flex-start;flex-direction:column}}.layout{{grid-template-columns:minmax(0,1fr)}}section{{padding:12px}}}}
</style></head><body><main>
<header><div><h1>{title}</h1><p class="stage">{stage}</p></div>
<div class="metrics"><div><span>本轮用时</span><b>{clock}</b></div><div><span>已识别</span><b>{observed}/{total}</b></div><div><span>放置复查</span><b>{verified}/{total}</b></div></div></header>
<div class="layout"><section><div class="sub"><h2>带编号的相机观测</h2><small>{frame}</small></div><div class="frame">{camera}</div></section>
<section><div class="sub"><h2>逐目标进度</h2></div><table aria-label="逐目标进度"><thead><tr><th>编号</th><th>识别类别</th><th>目标区</th><th>抓取反馈</th><th>放置复查</th></tr></thead><tbody>{rows}</tbody></table>{failure}</section></div>
<footer><span>试提保持依据夹爪反馈。有效抬升由裁判核验。</span><span>{latest} · <a href="process_events.jsonl">逐目标日志</a></span></footer>
</main></body></html>"""
