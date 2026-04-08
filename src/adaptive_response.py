"""
Adaptive Response Module
========================
Rule-based intervention system that triggers actions based on
detected cognitive state and focus score.

Actions: show hints, adjust difficulty, send notifications, alert instructor.
"""

import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class AdaptiveResponseEngine:
    """
    Processes student state and triggers appropriate interventions.
    Uses configurable thresholds and intervention definitions.
    """

    def __init__(self, config_path="config/config.yaml"):
        config = load_config(config_path)
        self.thresholds = config["adaptive_response"]["thresholds"]
        self.interventions = config["adaptive_response"]["interventions"]
        self.student_history = {}  # track recent states per student

    def update_history(self, student_id, state, focus_score):
        """Track recent states for pattern detection."""
        if student_id not in self.student_history:
            self.student_history[student_id] = []

        self.student_history[student_id].append({
            "state": state,
            "focus_score": focus_score,
        })

        # Keep last 10 entries
        self.student_history[student_id] = self.student_history[student_id][-10:]

    def detect_pattern(self, student_id):
        """Detect concerning patterns from recent state history."""
        history = self.student_history.get(student_id, [])
        if len(history) < 3:
            return None

        recent = [h["state"] for h in history[-5:]]
        recent_scores = [h["focus_score"] for h in history[-5:]]

        # Pattern: sustained confusion (3+ confused states in a row)
        if recent[-3:].count("confused") >= 3:
            return "sustained_confusion"

        # Pattern: declining focus (scores dropping)
        if len(recent_scores) >= 3:
            if all(recent_scores[i] > recent_scores[i+1] for i in range(len(recent_scores)-1)):
                return "declining_focus"

        # Pattern: boredom spiral (alternating bored/distracted)
        non_focused = [s for s in recent if s != "focused"]
        if len(non_focused) >= 4:
            return "disengagement"

        return None

    def get_response(self, student_id, state, focus_score, snapshot=None):
        """
        Determine the appropriate adaptive response.

        Returns dict with:
        - action: action name (e.g., "show_hint")
        - message: message to display
        - priority: 0 (critical) to 3 (low)
        - reason: why this response was triggered
        """
        self.update_history(student_id, state, focus_score)

        responses = []

        # Check state-based thresholds
        if state == "confused":
            cfg = self.thresholds["confused"]
            if focus_score < cfg["focus_score_below"]:
                responses.append({
                    "action": cfg["action"],
                    "reason": f"Confused state with low focus ({focus_score})",
                    **self.interventions[cfg["action"]],
                })
            elif snapshot and snapshot.get("replay_count", 0) > cfg["replay_count_above"]:
                responses.append({
                    "action": "show_hint",
                    "reason": f"High replay count ({snapshot['replay_count']})",
                    **self.interventions["show_hint"],
                })

        elif state == "bored":
            cfg = self.thresholds["bored"]
            if focus_score < cfg["focus_score_below"]:
                responses.append({
                    "action": cfg["action"],
                    "reason": f"Bored state with low focus ({focus_score})",
                    **self.interventions[cfg["action"]],
                })

        elif state == "distracted":
            cfg = self.thresholds["distracted"]
            if focus_score < cfg["focus_score_below"]:
                responses.append({
                    "action": cfg["action"],
                    "reason": f"Distracted state with low focus ({focus_score})",
                    **self.interventions[cfg["action"]],
                })

        # Check critical threshold
        if focus_score < self.thresholds["critical"]["focus_score_below"]:
            responses.append({
                "action": "alert_instructor",
                "reason": f"Critical focus level ({focus_score})",
                **self.interventions["alert_instructor"],
            })

        # Check patterns
        pattern = self.detect_pattern(student_id)
        if pattern == "sustained_confusion":
            responses.append({
                "action": "show_hint",
                "message": "You've been struggling for a while. Would you like a simplified explanation?",
                "priority": 1,
                "reason": "Sustained confusion detected",
            })
        elif pattern == "declining_focus":
            responses.append({
                "action": "send_notification",
                "message": "Your focus has been declining. A short break might help!",
                "priority": 2,
                "reason": "Declining focus trend",
            })
        elif pattern == "disengagement":
            responses.append({
                "action": "increase_difficulty",
                "message": "Let's try something more engaging — how about a challenge question?",
                "priority": 2,
                "reason": "Disengagement pattern detected",
            })

        if not responses:
            return {
                "action": "none",
                "message": "Keep up the good work!",
                "priority": 4,
                "reason": "No intervention needed",
            }

        # Return highest priority (lowest number) response
        responses.sort(key=lambda r: r.get("priority", 4))
        return responses[0]


def demo():
    """Demo the adaptive response engine with sample scenarios."""
    engine = AdaptiveResponseEngine()

    scenarios = [
        ("STU001", "focused", 85, {"tab_switch": 1, "replay_count": 0, "skip_count": 0}),
        ("STU001", "confused", 45, {"tab_switch": 2, "replay_count": 5, "skip_count": 0}),
        ("STU001", "confused", 38, {"tab_switch": 1, "replay_count": 7, "skip_count": 0}),
        ("STU001", "confused", 32, {"tab_switch": 1, "replay_count": 8, "skip_count": 0}),
        ("STU002", "bored", 30, {"tab_switch": 3, "replay_count": 0, "skip_count": 5}),
        ("STU002", "distracted", 25, {"tab_switch": 8, "replay_count": 0, "skip_count": 2}),
        ("STU003", "focused", 90, {"tab_switch": 0, "replay_count": 0, "skip_count": 0}),
        ("STU003", "focused", 15, {"tab_switch": 0, "replay_count": 0, "skip_count": 0}),
    ]

    print("Adaptive Response Engine — Demo")
    print("=" * 70)

    for student_id, state, score, snapshot in scenarios:
        response = engine.get_response(student_id, state, score, snapshot)
        print(f"\n  Student: {student_id} | State: {state:12s} | Score: {score}")
        print(f"  Action:  {response['action']}")
        print(f"  Message: {response['message']}")
        print(f"  Reason:  {response['reason']}")
        print(f"  Priority: {response.get('priority', '-')}")


if __name__ == "__main__":
    demo()
