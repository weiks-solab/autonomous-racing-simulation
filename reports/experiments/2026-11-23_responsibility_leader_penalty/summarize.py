"""列出本資料夾各組 json 的主要指標：python summarize.py name1 name2 ..."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name in sys.argv[1:]:
    r = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    att, dfd = r["strategies"]["relative_overtake"], r["strategies"]["defend"]
    first_def = next(iter(r["rejection_timeline"]["defend"]), None)
    print(f"== {name}")
    print(f"  final {{{', '.join(f'{k}: {v:.1f}' for k, v in r['final_s_from_start'].items())}}}"
          f"  lead {r['first_attacker_lead_time']}  coll {r['collision_steps']} ({r['first_collision_time']})"
          f"  block|dd| {r['blocking_mean_abs_delta_d_before_lead']:.3f} (n={r['blocking_samples']})")
    print(f"  att: {att['mode_counts']} rej {att['rejected_candidates']['responsibility']} "
          f"switch {att['target_d_switches']} ms {att['mean_ms']:.0f} leaderSel {att['evaluations_selected_with_leader_cost']}")
    print(f"  def: {dfd['mode_counts']} rej {dfd['rejected_candidates']['responsibility']} "
          f"switch {dfd['target_d_switches']} ms {dfd['mean_ms']:.0f} leaderSel {dfd['evaluations_selected_with_leader_cost']} "
          f"maxLeader {dfd['selected_leader_cost_max']:.2f}"
          f"  firstDefRej {None if first_def is None else (first_def['t'], first_def['s_attacker_minus_defender'])}")
