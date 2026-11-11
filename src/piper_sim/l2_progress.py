"""Conservative, sensor-only per-object progress for a finite sorting round.

Spatial/appearance association is bounded evidence, not persistent visual identity.
Ambiguous associations must be reobserved; no simulator IDs enter this ledger.
"""

import numpy as np


class L2Progress:
    def __init__(self, total=6, max_attempts=3, radius=0.10, color_mapping=None):
        self.total, self.max_attempts, self.radius = total, max_attempts, radius
        self.items = []
        self.color_mapping = color_mapping or {}

    def compatible(self, item, color, target):
        if item["color"] != color and self.color_mapping.get(color) != item["destination"]:
            return False
        appearance = target.appearance_rgb
        if appearance is None or item["appearance"] is None:
            return False
        a, b = np.asarray(appearance), np.asarray(item["appearance"])
        return np.linalg.norm(a / max(a.sum(), 1) - b / max(b.sum(), 1)) <= 0.18

    def associate(self, color, target):
        matches = []
        for item in self.items:
            if not self.compatible(item, color, target):
                continue
            # A verified object is bound to its final observed position, not its old source.
            anchors = [item["center"]] if item["status"] == "verified" else item["anchors"]
            distance = min(np.linalg.norm(target.center[:2] - np.asarray(p)[:2]) for p in anchors)
            if distance <= self.radius:
                matches.append((distance, item))
        matches.sort(key=lambda x: x[0])
        if len(matches) > 1 and matches[1][0] - matches[0][0] < 0.025:
            raise ValueError("Ambiguous object identity in progress ledger")
        return matches[0][1] if matches else None

    def identify(self, color, destination, target):
        item = self.associate(color, target)
        if item is None:
            if len(self.items) >= self.total:
                raise ValueError("Unbound observation cannot create a seventh task object")
            item = {
                "id": f"fruit_{len(self.items) + 1:02d}",
                "color": color,
                "destination": destination,
                "status": "pending",
                "attempts": 0,
                "appearance": None
                if target.appearance_rgb is None
                else target.appearance_rgb.tolist(),
                "center": target.center.tolist(),
                "anchors": [target.center.tolist()],
                "placement_verified_at_s": None,
                "last_error": None,
            }
            self.items.append(item)
        return item

    def start(self, color, destination, target):
        item = self.identify(color, destination, target)
        if item["status"] == "verified" or item["attempts"] >= self.max_attempts:
            raise ValueError("Already verified or exhausted target cannot be picked again")
        item["attempts"] += 1
        item["status"] = "executing"
        item["center"] = target.center.tolist()
        item["anchors"] = [target.center.tolist()]
        return item

    def expect_release(self, item, expected):
        item["anchors"].append(np.asarray(expected).tolist())
        item["status"] = "awaiting_verification"

    def fail(self, item, reason, target=None):
        item["status"] = "retry" if item["attempts"] < self.max_attempts else "exhausted"
        item["last_error"] = str(reason)
        item["placement_verified_at_s"] = None
        if target is not None:
            item["center"] = target.center.tolist()
            item["anchors"] = [target.center.tolist()]

    def verify(self, item, target, timestamp):
        item["status"] = "verified"
        item["center"] = target.center.tolist()
        item["anchors"] = [target.center.tolist()]
        item["placement_verified_at_s"] = float(timestamp)
        item["last_error"] = None

    @property
    def completed(self):
        return [i["id"] for i in self.items if i["status"] == "verified"]
