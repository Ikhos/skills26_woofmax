from typing import Dict, Any

def baymax_summary(auth: Dict[str, Any], scan: Dict[str, Any], trend: Dict[str, Any] | None = None) -> str:

    if auth.get("status") in ("unknown_or_low_conf", "denied_pin", "locked"):
        return (
            "I cannot access any saved health history right now. "
            "If you would like, we can try again with better lighting, or you can verify with your PIN."
        )

    if not scan or not scan.get("ok"):
        return "I was unable to complete the scan. Please center your face and try again."

    score = scan.get("symmetry_score")
    flags = scan.get("flags", [])

    msg = []
    msg.append("Scan complete.")

    if score is not None:
        msg.append(f"Your symmetry score is {score:.2f}.")

    if flags:
        friendly = []
        for f in flags:
            if f == "mouth_left_lower":
                friendly.append("your left mouth corner appears slightly lower")
            elif f == "mouth_right_lower":
                friendly.append("your right mouth corner appears slightly lower")
            elif f == "eye_left_more_closed":
                friendly.append("your left eye appears slightly more closed")
            elif f == "eye_right_more_closed":
                friendly.append("your right eye appears slightly more closed")
            else:
                friendly.append(f.replace("_", " "))
        msg.append("I noticed that " + ", and ".join(friendly) + ".")
    else:
        msg.append("I did not detect any obvious asymmetry signals.")

    if trend and trend.get("status") == "ok":
        classification = trend.get("classification")

        if classification == "stable":
            msg.append("Compared to your recent scans, this appears stable.")
        elif classification == "improving":
            msg.append("Compared to your recent scans, this appears slightly improved.")
        elif classification == "deteriorating":
            msg.append("Compared to your recent scans, this appears slightly worse.")

        msg.append("Would you like a progress check or another scan?")
    else:
        msg.append("Would you like me to perform another scan or monitor changes over time?")

    return " ".join(msg)
