"""Interactive L1/L2 rounds. Prompts use public rules, never object poses."""

LABELS = {
    "cube_red": "红色方块",
    "cube_green": "绿色方块",
    "cube_blue": "蓝色方块",
    "cube_yellow": "黄色方块",
    "persimmon": "柿子",
    "bell_pepper": "彩椒",
    "cucumber": "黄瓜",
    "potato": "土豆",
    "apple": "苹果",
    "pear": "梨",
    "banana": "香蕉",
    "lemon": "柠檬",
    "orange": "橙子",
    "strawberry": "草莓",
    "peach": "桃子",
    "plum": "李子",
}
COLORS = {"red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色"}


def round_requests(mode, scene_config, demo_config):
    task = scene_config["tasks"][mode]
    mapping = task["mapping"]
    region_name = (
        "目标区" if scene_config.get("tray_geometry", {}).get("kind") == "flat_zone" else "托盘"
    )
    if mode == "l1":
        return [
            {
                "instruction": f"把{LABELS[category]}放到{COLORS[mapping[category]]}{region_name}",
                "backend": "rule",
            }
            for category in task["objects"]
        ]
    if mode != "l2":
        raise ValueError("Only L1/L2 are automatic rounds")
    rules = "；".join(f"{LABELS[k]}→{COLORS[v]}{region_name}" for k, v in mapping.items())
    instruction = (
        "识别桌面待分拣区域中一个果蔬，按以下演示分类规则选择对应托盘："
        + rules
        + "。每次只选一个目标，优先完整可见、独立且容易夹取的物体。"
        "不要选择已在目标区或托盘内的物体；画面中没有待分拣果蔬则返回target_not_found。"
    )
    return [
        {"instruction": instruction, "backend": "qwen", "classification": True}
        for _ in range(demo_config.get("l2_max_attempts", 8))
    ]


def run_round(port):
    mode = port.task_mode
    task = port.scene_cfg["tasks"][mode]
    target_count = len(task["objects"]) if "objects" in task else task["object_count"]
    port.world.begin_attempt()
    import time

    port.started = port.last_tick = time.monotonic()
    results = []
    completed = 0
    round_id = str(time.time_ns())
    progress = {
        "mode": mode,
        "target_count": target_count,
        "completed": 0,
        "time_limit_s": task["time_limit_s"],
        "elapsed_s": 0,
        "attempt": 0,
        "results": [],
        "rules_status": task.get("mapping_status", "development_mapping_not_official"),
    }
    port.update(round_progress=progress)
    try:
        for index, request in enumerate(
            round_requests(mode, port.scene_cfg, port.cfg.get("demos", {}))
        ):
            port.checkpoint()
            if port.world.elapsed_s >= task["time_limit_s"] - 1e-8:
                break
            progress = {**progress, "attempt": index + 1}
            port.update(round_progress=progress)
            try:
                result = port.execute({**request, "round_id": round_id}, new_attempt=False)
            except TimeoutError as exc:
                result = {
                    "success": False,
                    "error": str(exc),
                    "failure_code": "round_timeout"
                    if port.world.elapsed_s >= task["time_limit_s"] - 1e-8
                    else "motion_timeout",
                }
            results.append(
                {
                    "attempt": index + 1,
                    "instruction": request["instruction"],
                    "success": result["success"],
                    "message": result.get("message") or result.get("error"),
                    "failure_code": result.get("failure_code"),
                }
            )
            # Evaluator counts distinct objects; no hidden labels enter perception/motion.
            completed = len(port.world.evaluation()["correctly_placed"])
            progress = {
                **progress,
                "completed": completed,
                "elapsed_s": round(port.world.elapsed_s, 2),
                "results": list(results),
            }
            port.update(round_progress=progress)
            if completed >= target_count or result.get("failure_code") in (
                "model_unavailable",
                "round_timeout",
                "grip_unstable",
                "grip_feedback_invalid",
                "recovery_failed",
            ):
                break
        result = {
            "success": completed >= target_count,
            "completed": completed,
            "target_count": target_count,
            "round_progress": progress,
            "message": f"{mode.upper()} 演示结束：{completed}/{target_count} 个物体通过分类与稳定放置检查。"
            "现场已保留，可继续尝试或随机重置。",
            "official_score": None,
        }
        port.update(result=result, active_skill=None, needs_reset=False)
        port.stage("done" if result["success"] else "failed", result["message"])
        return result
    finally:
        import json

        path = port.output / "rounds" / (round_id + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "seed": port.seed,
                    "progress": progress,
                    "evaluation": port.world.evaluation(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
