"""离线解析 TensorBoard 事件，打印每个奖励项/关键指标的起始、中段、末段均值，定位无效奖励项。"""

import sys
import glob
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def main(run_dir):
    ev_files = glob.glob(f"{run_dir}/events.out.tfevents.*")
    if not ev_files:
        print(f"no event files in {run_dir}")
        return
    ev = sorted(ev_files, key=lambda p: p)[-1]
    print(f"reading: {ev}\n")
    acc = event_accumulator.EventAccumulator(
        ev, size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()
    tags = acc.Tags().get("scalars", [])

    def summarize(tag):
        events = acc.Scalars(tag)
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        n = len(vals)
        if n == 0:
            return None
        seg = max(1, n // 10)
        start = vals[:seg].mean()
        mid = vals[n // 2 - seg // 2 : n // 2 + seg // 2 + 1].mean()
        end = vals[-seg:].mean()
        return steps[-1], start, mid, end

    # 分组打印
    groups = {}
    for t in tags:
        prefix = t.split("/")[0]
        groups.setdefault(prefix, []).append(t)

    for prefix in sorted(groups):
        print(f"==== {prefix} ====")
        print(f"{'tag':<42}{'last_step':>10}{'start':>12}{'mid':>12}{'end':>12}")
        for t in sorted(groups[prefix]):
            r = summarize(t)
            if r is None:
                continue
            last_step, start, mid, end = r
            print(f"{t:<42}{last_step:>10}{start:>12.4f}{mid:>12.4f}{end:>12.4f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
