"""Read-only A17 derived-v2/v3 figures; never launch or import physics code."""
from pathlib import Path
import argparse, collections, hashlib, json, math, statistics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

EFF = Path(__file__).resolve().parents[1]
OBJECTS = {2002: 'Smooth voxel', 2009: 'Shell Gaussian'}
POLICIES = {
    'receiver_only': ('Receiver only', '#222222'),
    'original_top2_cap3': ('Original: top 2, max 3 rounds', '#777777'),
    'verified_exchange': ('Original verified exchange', '#777777'),
    'top1_cap3': ('Top 1, max 3 rounds', '#0072B2'),
    'top2_cap1': ('Top 2, max 1 round', '#E69F00'),
    'top1_cap1': ('Top 1, max 1 round', '#CC79A7'),
    'top1_cap2': ('Top 1, max 2 rounds', '#009E73'),
    'reuse_previous': ('Previous-space reuse', '#D55E00'),
    'reuse_reset3': ('Previous-space reuse; reset every 3', '#56B4E9'),
}
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.facecolor': 'white', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def digest(b):
    return hashlib.sha256(b).hexdigest()


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_schema(schema):
    allowed = ('a17.efficiency.local_analysis.v2', 'a17.efficiency.local_analysis.v3')
    if schema not in allowed:
        raise ValueError('Expected derived-v2/v3 analysis schema; unknown schemas are rejected.')


def appearance(r):
    name, col = POLICIES.get(r['policy'], (r['policy'].replace('_', ' '), '#777777'))
    if r.get('geometry_cache') is not None:
        name += ' / geometry cache' if r['geometry_cache'] else ' / uncached'
    elif ':' in r.get('method', ''):
        name += ' / ' + r['method'].split(':', 1)[1].replace('_', ' ')
    if r['policy'] in ('reuse_previous', 'reuse_reset3'):
        name += ' [secondary pilot]'
    return name, col


def empty(ax, reason):
    ax.text(.5, .5, reason, ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def run(args):
    data = args.data.resolve()
    out = args.out.resolve() if args.out else data / 'figures_final'
    out.mkdir(parents=True, exist_ok=True)
    inputs = []

    def read(path):
        b = path.read_bytes()
        inputs.append({'path': str(path), 'bytes': len(b), 'sha256': digest(b)})
        return json.loads(b)

    receipt = read(data / 'analysis_receipt.json')
    validate_schema(receipt.get('schema'))
    frozen = read(data / 'frozen_object_policy.json')
    cells = read(data / 'frozen_cells.json')
    nl = read(data / 'nonlinear_runs.json')
    costs = read(data / 'cost_components.json')
    matched = read(data / 'nonlinear_matched_summary.json')
    prelock = read(args.policy.resolve())
    status = receipt.get('status', 'PARTIAL')
    summary = {'analysis_schema': receipt['schema'], 'status': status,
               'frozen': [], 'nonlinear': [], 'original_net_cost': [], 'wall_ratios': [],
               'strata': [], 'missing': [], 'limitations': [
                   'Saved frozen marginal verification and cold nonlinear are separate scopes.',
                   'Three timing repeats are the same two objects, not independent scenes.',
                   'No missing policy or object measurements are interpolated.',
                   'Jd wall is nested inside original total child wall; bars must not be added.',
                   'Receiver frozen origin is an algebraic no-exchange reference, not a new run.',
                   'No gate decision or scientific acceptance is made by this plotter.']}
    figures = []

    def save(fig, name, scope, caption):
        fig.suptitle(caption, fontsize=11)
        fig.tight_layout(rect=(0, .04, 1, .93))
        artifacts = []
        for ext in ('pdf', 'svg', 'png'):
            path = out / (name + '.' + ext)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            artifacts.append({'path': str(path), 'bytes': path.stat().st_size,
                              'sha256': digest(path.read_bytes())})
        figures.append({'name': name, 'scope': scope, 'artifacts': artifacts})
        plt.close(fig)

    # Frozen: measured prelocked policies only, plus labeled receiver reference.
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for ax, obj in zip(axs, OBJECTS):
        ax.scatter(0, 0, marker='x', s=60, color='#222222', label='Receiver reference (0, 0)')
        for policy in prelock['policies']:
            rows = [r for r in frozen if r['scope'] == 'eff_frozen' and r['object'] == obj
                    and r['policy'] == policy and finite(r.get('avg_Jd'))
                    and finite(r.get('object_gain_retention'))]
            if len(rows) != 1:
                summary['missing'].append({'figure': 'frozen', 'object': obj,
                                           'policy': policy, 'reason': 'missing or ambiguous aggregate'})
                continue
            r = rows[0]
            name, col = appearance(r)
            partial = r.get('coverage_status') != 'COMPLETE_MEASUREMENT_GRID'
            ax.scatter(r['avg_Jd'], r['object_gain_retention'], s=65, color=col,
                       marker='o', facecolors='none' if partial else col,
                       label=name + (' [PARTIAL]' if partial else ''))
            summary['frozen'].append(r)
        ax.set(xlabel='Mean full Jd directions / frozen cell',
               ylabel='Retained positive gain / original positive gain',
               title=OBJECTS[obj] + ' / frozen cells')
        ax.set_ylim(-.08, 1.12)
        ax.set_xlim(left=-.25)
        ax.grid(alpha=.18)
    handles, labels = axs[0].get_legend_handles_labels()
    # Include policies that may only be measured on the second object.
    for ha, la in zip(*axs[1].get_legend_handles_labels()):
        if la not in labels:
            handles.append(ha); labels.append(la)
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(.5, -.10), frameon=False)
    save(fig, '01_frozen_gain_vs_Jd', 'eff_frozen only; no nonlinear data',
         'Frozen verification: measured gain retention and direction count / ' + status)

    # Preserve the full matched stratum, driver and config identities, not only version.
    groups = collections.defaultdict(list)
    for r in nl:
        if r['scope'] == 'eff_nonlinear':
            key = (r.get('comparison_stratum'), r.get('version'), r.get('config_sha256'),
                   r.get('nonlinear_driver_sha256'), r.get('match_identity_driver_sha256'))
            groups[key].append(r)
    if not groups:
        groups[(None, None, None, None, None)] = []
    for number, (key, rows) in enumerate(sorted(groups.items(), key=lambda t: str(t[0])), 1):
        stratum, version, config, driver, identity_driver = key
        short = f'Stratum {number}'
        meta = {'figure_stratum': short, 'comparison_stratum': stratum, 'version': version,
                'config_sha256': config, 'driver_sha256': driver,
                'identity_driver_sha256': identity_driver, 'runs': len(rows)}
        summary['strata'].append(meta)
        technical = f'{short}: {version or "NOT_RUN"}\nconfig {(config or "missing")[:12]} / driver {(driver or identity_driver or "missing")[:12]}'
        fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
        for ii, obj in enumerate(OBJECTS):
            for jj, (field, ylabel) in enumerate([
                    ('material_complex_relative', 'Final complex-material relative error'),
                    ('heldout_data_relative', 'Final held-out data relative error')]):
                ax = axs[ii, jj]
                available = [r for r in rows if r['object'] == obj
                             and finite(r.get('child_wall_total_s')) and finite(r.get(field))]
                if not available:
                    empty(ax, 'PARTIAL / no saved new nonlinear measurement')
                seen = set()
                for r in available:
                    name, col = appearance(r)
                    label = name if name not in seen else None
                    seen.add(name)
                    incomplete = r.get('measurement_status') != 'MEASURED'
                    eligible = r.get('match_identity_eligible') is True
                    ax.scatter(r['child_wall_total_s'], r[field], s=60, color=col,
                               marker=('s' if r.get('geometry_cache') else 'o') if eligible else 'x',
                               facecolors='none' if incomplete else col, label=label)
                    ax.annotate('rep ' + str(r.get('repetition')) + (' PARTIAL' if incomplete else ''),
                                (r['child_wall_total_s'], r[field]), xytext=(4, 6),
                                textcoords='offset points', fontsize=7)
                if available:
                    ax.legend(fontsize=7, loc='best', frameon=False)
                ax.set(xlabel='Total child wall (s)', ylabel=ylabel, title=OBJECTS[obj])
                ax.grid(alpha=.18)
        save(fig, f'02_nonlinear_wall_error_stratum_{number:02d}',
             {**meta, 'all_repetitions': True}, 'New nonlinear: all saved repetitions / ' + status + '\n' + technical)
        summary['nonlinear'].extend(rows)

        # Ratios: exact 3-repeat median/range only; incomplete groups remain individual points.
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
        for ax, obj in zip(axs, OBJECTS):
            methods = collections.defaultdict(list)
            for r in rows:
                if r['object'] == obj:
                    methods[r['method']].append(r)
            if not methods:
                empty(ax, 'PARTIAL / no saved same-stratum ratio')
            yticks, ylabels = [], []
            for pos, (method, methodrows) in enumerate(sorted(methods.items())):
                name, col = appearance(methodrows[0])
                valid = [r for r in methodrows if r.get('match_identity_eligible') is True
                         and r.get('measurement_status') == 'MEASURED'
                         and finite(r.get('matched_receiver_wall_ratio'))]
                reps = [r.get('repetition') for r in valid]
                unambiguous = len(reps) == len(set(reps)) and None not in reps
                secondary = methodrows[0]['policy'] in ('reuse_previous', 'reuse_reset3')
                complete = unambiguous and len(valid) == prelock['nonlinear']['matched_repeat_count'] and not secondary
                vals = [r['matched_receiver_wall_ratio'] for r in valid]
                yticks.append(pos)
                if complete:
                    median, low, high = statistics.median(vals), min(vals), max(vals)
                    ax.errorbar(median, pos, xerr=[[median-low], [high-median]],
                                fmt='o', color=col, capsize=4)
                    ylabels.append(name + ' / 3 repeats')
                    entry = {'mode': 'three_repeat_median_and_range', 'median': median,
                             'min': low, 'max': high}
                else:
                    for n, r in enumerate(valid):
                        ax.scatter(r['matched_receiver_wall_ratio'], pos + .08 * (n-(len(valid)-1)/2),
                                   marker='o', s=45, facecolors='none', edgecolors=col)
                    ylabels.append(name + (f' / PARTIAL {len(valid)}/3' if not secondary else f' / n={len(valid)}'))
                    entry = {'mode': 'secondary_pilot' if secondary else 'partial_individual_repeats',
                             'median': None, 'min': None, 'max': None}
                summary['wall_ratios'].append({**meta, 'object': obj, 'method': method,
                    'repetitions': reps, 'ratios': vals, 'identity_unambiguous': unambiguous,
                    'source_result_sha256': [r['result_sha256'] for r in valid], **entry})
            if yticks:
                ax.set_yticks(yticks, ylabels, fontsize=8)
            ax.axvline(1, linestyle=':', color='#999999', linewidth=1)
            ax.set(xlabel='Total child wall / matched receiver total child wall', title=OBJECTS[obj])
            ax.grid(axis='x', alpha=.18)
        save(fig, f'04_wall_ratio_stratum_{number:02d}', meta,
             'Within-stratum wall ratios: median + min/max only for 3 matched repeats / ' + status + '\n' + technical)

    # Original 18-update all-in net difference versus actually measured nested Jd wall.
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, obj in zip(axs, OBJECTS):
        original = [r for r in nl if r['scope'] == 'closure_saved_nonlinear'
                    and r['object'] == obj and r['policy'] == 'verified_exchange']
        receiver = [r for r in nl if r['scope'] == 'closure_saved_nonlinear'
                    and r['object'] == obj and r['policy'] == 'receiver_only']
        if len(original) != 1 or len(receiver) != 1 or any(r.get('trajectory_records') != 18 for r in original+receiver):
            empty(ax, 'Missing / ambiguous original 18-update pair')
            continue
        a, b = original[0], receiver[0]
        jd = [c for c in costs if c['scope'] == 'closure_saved_nonlinear'
              and c['run_id'] == a['result_sha256'] and c['label'] == 'Jd_wall_s'
              and c['time_kind'] == 'within_online_disjoint_stage' and finite(c.get('wall_s'))]
        if len(jd) != 1 or not finite(a.get('child_wall_total_s')) or not finite(b.get('child_wall_total_s')):
            empty(ax, 'Missing / ambiguous original Jd wall')
            continue
        net = a['child_wall_total_s'] - b['child_wall_total_s']
        ax.bar([0, 1], [net, jd[0]['wall_s']], color=['#777777', '#0072B2'], width=.6)
        ax.set_xticks([0, 1], ['Net extra child wall\nexchange − receiver', 'Actual Jd wall\nnested substage'])
        for x, val in enumerate([net, jd[0]['wall_s']]):
            ax.annotate(f'{val:.2f} s', (x, val), xytext=(0, 5), textcoords='offset points', ha='center')
        ax.set(title=OBJECTS[obj] + ' / original 18 updates', ylabel='Measured wall (s)')
        ax.grid(axis='y', alpha=.18)
        summary['original_net_cost'].append({'object': obj, 'exchange_total_child_wall_s': a['child_wall_total_s'],
            'receiver_total_child_wall_s': b['child_wall_total_s'], 'net_increment_s': net,
            'actual_Jd_wall_s': jd[0]['wall_s'], 'exchange_result_sha256': a['result_sha256'],
            'receiver_result_sha256': b['result_sha256'], 'nested_not_additive': True})
    save(fig, '03_original_18_update_net_vs_Jd', 'closure_saved_nonlinear only',
         'Original 18-update net cost and paid Jd wall / separate saved scope\nParallel bars; Jd is nested, not an additive cost decomposition')

    # Bind every consumed byte, including the old style reference and this implementation.
    for path in [EFF / 'code/plot_efficiency.py', Path(__file__).resolve()]:
        b = path.read_bytes(); inputs.append({'path': str(path), 'bytes': len(b), 'sha256': digest(b)})
    summary['matched_summary_rows_available'] = len(matched)
    summary['counts'] = {'frozen_measured_policy_points': len(summary['frozen']),
                         'new_nonlinear_runs': len(summary['nonlinear']),
                         'strata': len(summary['strata']), 'original_pairs': len(summary['original_net_cost']),
                         'ratio_three_repeat_groups': sum(r['mode'] == 'three_repeat_median_and_range' for r in summary['wall_ratios'])}
    for name, content in [('data_summary.json', summary), ('input_manifest.json', inputs),
                           ('figure_manifest.json', {'status': status, 'figures': figures,
                             'style': 'White; limited fixed palette; editable text SVG and vector PDF; PNG',
                             'input_sha256': inputs, 'gate_decision': 'OWNER_NOT_ASSIGNED'})]:
        (out / name).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    handoff = f'''# A17 final-figure implementation handoff

当前数据状态：{status}。只读取 derived-v2 表，未读取或重跑 physics，也未连接 SSH/GPU。旧 plot_efficiency.py、配置和结果均未修改。当前摘要：{json.dumps(summary['counts'], ensure_ascii=False)}。

- 01：eff_frozen 的 avg_Jd 与 object_gain_retention；只画 prelock 五种 policy 的实测行。receiver (0,0) 是无 exchange 增量的代数参考，不是新增测量。frozen marginal cost 与 nonlinear 不混同。
- 02：eff_nonlinear 的全部 saved repetitions；x=child_wall_total_s，y=material_complex_relative 或 heldout_data_relative。完整 comparison_stratum、version、config SHA、driver SHA 显式分图并在 manifest 保留全文。PARTIAL measurement 用空心点；身份不合格用 x；没有隐藏最快/最慢重复。
- 03：closure_saved_nonlinear 原 18-update exchange total child wall 减 receiver total child wall，与同 run_id 的 cost_components / within_online_disjoint_stage / Jd_wall_s 并列。Jd 是 nested 子项，不能与 net 相加；net 包含不同 accepted-path 的全部 child 成本，并非 causal isolated Jd overhead。
- 04：同 stratum/object/method 的 matched_receiver_wall_ratio。仅三个唯一、MEASURED、身份合格 repeats 用 median 与 min/max；n<3只显示原始点并标 PARTIAL，不生成三重复统计。reuse_previous/reuse_reset3 始终标 secondary pilot，并不与 main repeats 合并。

没有伪造 P6/P7 或其他未运行点。内部对象 ID 不作为主图标签；主标签为 Smooth voxel / Shell Gaussian。表中无新结果的对象留缺失 panel。两对象三 timing repeats 不等于独立场景验证。没有判断 Gate 或科学接受。

PDF 使用 TrueType fonts，SVG 保留 editable text；每图另有 PNG。figure_manifest.json 记录图件 SHA-256；input_manifest.json 绑定所有消费表、prelock、旧绘图 reference 与新脚本；data_summary.json 保留 plotted values、strata、重复与局限。

复画命令：`Gaussian/.venv_nn/bin/python Gaussian/A17/EFFICIENCY_R1/code/plot_efficiency_final.py --data Gaussian/A17/EFFICIENCY_R1/analysis/derived_v2 --out Gaussian/A17/EFFICIENCY_R1/analysis/derived_v2/figures_final`。root 完整数据后重新绘图和视觉检查；当前图件只证明本地 PARTIAL 表可正常渲染。
'''
    (out / 'HANDOFF_PLOTS.md').write_text(handoff)
    print(json.dumps({'status': status, 'figures': len(figures), 'out': str(out), 'counts': summary['counts']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=EFF / 'analysis/derived_v2')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--policy', type=Path, default=EFF / 'configs/POLICY_PRELOCK_V1.json')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
