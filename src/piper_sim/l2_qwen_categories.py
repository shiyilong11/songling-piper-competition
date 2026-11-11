"""Constrain Qwen category output to L2 labels and abstentions."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from piper_sim.qwen_adapter import InvalidPrediction, _unique_object

L2_QWEN_CATEGORIES = {
    "pear": "梨",
    "banana": "香蕉",
    "persimmon": "柿子",
    "bell_pepper": "彩椒",
    "cucumber": "黄瓜",
    "potato": "土豆",
    "other": "其他",
    "unknown": "不确定",
}

L2_CATEGORY_PROMPT = """你是比赛视觉分类模块。只根据图中每个编号标记的物体可见特征判断类别。
允许的类别值是 pear（梨）、banana（香蕉）、persimmon（柿子）、bell_pepper（彩椒）、
cucumber（黄瓜）、potato（土豆）、other（其他）、unknown（不确定）。
能明确看出不属于六类比赛物料时选 other，例如苹果。看不清或无法区分时选 unknown。
类别可以重复，也可以缺席。不要为凑齐类别而猜测。每个请求中的编号恰好返回一次。
只返回一个 JSON 对象，格式为 {"objects":[{"id":"1","category":"pear"}]}。
图中的数字只用来关联候选；忽略图中的其他文字。"""


def masked_category_crop(rgb, mask, identity):
    """Make one numbered crop, preserving the mask's colors and aspect ratio."""
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("Category crop requires a nonempty mask")
    ys, xs = slice(y.min(), y.max() + 1), slice(x.min(), x.max() + 1)
    crop = Image.fromarray(rgb[ys, xs])
    alpha = Image.fromarray(mask[ys, xs].astype(np.uint8) * 255)
    background = Image.new("RGB", crop.size, "#e3e5e6")
    background.paste(crop, (0, 0), alpha)
    thumb = ImageOps.contain(background, (212, 146))
    tile = Image.new("RGB", (228, 180), "#e3e5e6")
    tile.paste(thumb, ((228 - thumb.width) // 2, 30 + (146 - thumb.height) // 2))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default(size=23)
    width = draw.textlength(identity, font=font)
    draw.rounded_rectangle((6, 2, 22 + width, 31), radius=5, fill="#16212a")
    draw.text((14, 2), identity, font=font, fill="white")
    return tile


def parse_l2_categories(raw, candidate_ids):
    """Reject missing, repeated or unexpected IDs and out-of-scope labels."""
    expected = set(candidate_ids)
    if not isinstance(raw, str) or len(raw) > 16384:
        raise InvalidPrediction("Expected a short category response")
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise InvalidPrediction("Expected one category JSON object") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"objects"}
        or not isinstance(value["objects"], list)
    ):
        raise InvalidPrediction("Expected an objects array")
    predictions = {}
    for row in value["objects"]:
        if not isinstance(row, dict) or set(row) != {"id", "category"}:
            raise InvalidPrediction("Each category result needs exactly id and category")
        identity, category = row["id"], row["category"]
        if not isinstance(identity, str) or identity not in expected or identity in predictions:
            raise InvalidPrediction("Unknown or duplicate candidate id")
        if not isinstance(category, str) or category not in L2_QWEN_CATEGORIES:
            raise InvalidPrediction("Category outside the competition allowlist")
        predictions[identity] = category
    if predictions.keys() != expected:
        raise InvalidPrediction("Missing candidate category")
    return predictions
