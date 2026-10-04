"""Owner-audit-only nonlinear plots. No physics imports, raw-array reads or gate assignment."""
from pathlib import Path
import argparse, collections, hashlib, json, math, statistics

EFF = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA = 'a17.efficiency.owner_trajectory_audit.v2'
ANALYSIS_SCHEMA = 'a17.efficiency.local_analysis.v3'
ROLES = {'a17.efficiency.nonlinear_transfer.v1': 'MAIN',
         'a17.efficiency.nonlinear_reuse_transfer.v1': 'REUSE_SECONDARY',
         'a17.efficiency.nonlinear_top1cap3_transfer.v1': 'TOP1CAP3_CONDITIONAL'}
NAMES = {'receiver_only': ('Receiver only', '#222222'),
         'original_top2_cap3': ('Original top 2 / up to 3 rounds', '#777777'),
         'top1_cap2': ('Top 1 / up to 2 rounds', '#009E73'),
         'top1_cap3': ('Top 1 / up to 3 rounds', '#0072B2'),
         'reuse_previous': ('Previous-space reuse', '#D55E00'),
         'reuse_reset3': ('Reuse / reset every 3', '#E69F00')}
OBJECTS = {2002: 'Smooth voxel / voxel material', 2009: 'Shell Gaussian / Gaussian material'}
OBJECTS_ZH = {2002: '平滑体素 / 体素材料', 2009: '壳层高斯 / 高斯材料'}
COUNTS = [('tangent_rhs', 'Full tangent RHS'), ('Jd_directions', 'Full Jd directions'),
          ('Jx_directions', 'Full Jx directions'), ('full_adjoint_rhs', 'Full adjoint RHS'),
          ('full_forward_rhs', 'Full forward RHS'), ('full_LU', 'Full LU factorizations')]


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def hash_string(s):
    return isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s)


def schema_check(audit, analysis):
    if audit.get('schema') != AUDIT_SCHEMA or analysis.get('schema') != ANALYSIS_SCHEMA:
        raise ValueError('Only owner trajectory audit v2 plus analyzer v3 are accepted.')


def array_check(row):
    raw = row.get('raw_array_check', {})
    eq = raw.get('final_matches_last_safe') or {}
    return (row.get('raw_binding_status') == 'HASH_BOUND_RAW_DIRECTORY'
            and raw.get('status') == 'MEASURED_ARRAY_CHECKS_ONLY'
            and not raw.get('issues') and eq.get('status') == 'MEASURED_ARRAY_COMPARISON'
            and eq.get('bitwise_equal') is True
            and all(r.get('status') == 'MEASURED_FEASIBLE' for r in raw.get('accepted_checkpoint_checks', [])))


def row_issues(row):
    issues = []
    if row.get('version') not in ROLES or ROLES.get(row.get('version')) != row.get('driver_role'):
        issues.append('UNKNOWN_VERSION_OR_ROLE_MISMATCH')
    if row.get('identity_eligible') is not True:
        issues.append('IDENTITY_NOT_ELIGIBLE')
    if any(not hash_string(row.get(k)) for k in ['result_sha256', 'config_sha256', 'physics_identity']):
        issues.append('MISSING_IDENTITY_HASH')
    if not array_check(row):
        issues.append('ARRAY_CHECK_FAILED_OR_NOT_MEASURED')
    if row.get('observable_label_checks') is not True or row.get('all_projection_solver_success') is not True or row.get('check_failures'):
        issues.append('OBSERVED_PATH_CHECK_OPEN')
    if row.get('object') not in OBJECTS or row.get('policy') not in NAMES:
        issues.append('UNKNOWN_OBJECT_OR_METHOD')
    if type(row.get('geometry_cache')) is not bool or type(row.get('repetition')) is not int or row.get('repetition') not in (0, 1, 2):
        issues.append('CACHE_OR_REPEAT_IDENTITY_MISSING')
    if any(not finite(row.get(k)) or row[k] < 0 for k in ['child_wall_s', 'material', 'heldout']):
        issues.append('QUALITY_OR_WALL_MISSING')
    return issues


def prepare_tables(audit, analysis, frozen):
    """Pure schema/identity/grouping; does not access any filesystem or plot library."""
    schema_check(audit, analysis)
    reports = collections.defaultdict(list)
    for row in audit.get('reports', []):
        reports[row.get('result_sha256')].append(row)
    accepted, opened = [], []
    for c in audit.get('comparisons', []):
        hashes = [c.get('source_sha256'), c.get('reference_sha256'), c.get('receiver_sha256')]
        reasons = []
        if any(not hash_string(h) or len(reports[h]) != 1 for h in hashes):
            reasons.append('HASH_BOUND_REPORT_MISSING_OR_AMBIGUOUS')
        if reasons:
            opened.append({'comparison': c, 'reasons': reasons}); continue
        row, original, receiver = [reports[h][0] for h in hashes]
        for role_name, r in zip(['source', 'original', 'receiver'], [row, original, receiver]):
            reasons.extend(role_name + ':' + issue for issue in row_issues(r))
        for r in [original, receiver]:
            if r.get('driver_role') != 'MAIN' or r.get('geometry_cache') is not False:
                reasons.append('REFERENCE_NOT_MAIN_UNCACHED')
        if original.get('policy') != 'original_top2_cap3' or receiver.get('policy') != 'receiver_only':
            reasons.append('REFERENCE_METHOD_MISMATCH')
        for field in ['object', 'repetition', 'physics_identity']:
            if not (c.get(field) == row.get(field) == original.get(field) == receiver.get(field)):
                reasons.append('SAME_PHYSICS_OBJECT_REP_BINDING_MISMATCH:' + field)
        for field in ['version', 'config_sha256', 'driver_role', 'policy', 'geometry_cache']:
            if c.get(field) != row.get(field):
                reasons.append('COMPARISON_REPORT_MISMATCH:' + field)
        if c.get('reference_version') != original.get('version') or c.get('reference_config_sha256') != original.get('config_sha256'):
            reasons.append('REFERENCE_VERSION_OR_CONFIG_BINDING_MISMATCH')
        if row.get('driver_role') == 'MAIN' and row.get('policy') == 'original_top2_cap3' and row.get('geometry_cache') is True:
            equivalents = [e for e in audit.get('cached_original_array_equivalence', [])
                           if e.get('cached_result_sha256') == row.get('result_sha256')
                           and e.get('uncached_result_sha256') == original.get('result_sha256')]
            if (len(equivalents) != 1 or equivalents[0].get('status') != 'MEASURED_ARRAY_COMPARISONS_ONLY'
                    or equivalents[0].get('issues') or not equivalents[0].get('comparisons')
                    or any(e.get('status') != 'MEASURED_ARRAY_COMPARISON' or e.get('bitwise_equal') is not True
                           for e in equivalents[0].get('comparisons', []))):
                reasons.append('CACHED_ORIGINAL_ARRAY_EQUIVALENCE_OPEN')
        expected = ('SAME_MAIN_VERSION_PHYSICS_REP' if row.get('driver_role') == 'MAIN'
                    else 'EXPLICIT_CROSS_VERSION_SAME_PHYSICS_REP_SECONDARY_NO_INDEPENDENT_VALIDATION')
        if c.get('driver_comparison') != expected or c.get('observed_path_checks') is not True:
            reasons.append('EXPLICIT_COMPARISON_SCOPE_OPEN')
        if row.get('driver_role') == 'MAIN' and row.get('version') != original.get('version'):
            reasons.append('MAIN_VERSION_MISMATCH')
        if reasons:
            opened.append({'comparison': c, 'reasons': sorted(set(reasons))})
        else:
            accepted.append({'comparison': c, 'report': row})
    groups = collections.defaultdict(list)
    for item in accepted:
        r = item['report']
        key = tuple(r[k] for k in ['object', 'policy', 'geometry_cache', 'driver_role',
                                   'physics_identity', 'version', 'config_sha256'])
        groups[key].append(item)
    table = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        reps = [m['report']['repetition'] for m in members]
        if len(set(reps)) != len(reps):
            opened.append({'group': list(key), 'reasons': ['DUPLICATE_REPEAT_IN_GROUP']}); continue
        obj, policy, cache, role, physics, version, config = key
        three = len(members) == 3 and set(reps) == {0, 1, 2} and role != 'REUSE_SECONDARY'
        mode = ('timing_three_median_range' if three else 'secondary_single_run' if role == 'REUSE_SECONDARY'
                and len(members) == 1 else 'partial_individual_runs')
        if role == 'REUSE_SECONDARY' and len(members) != 1:
            mode = 'secondary_individual_runs_no_pool'
        metrics = {}
        for field in ['child_wall_s', 'material', 'heldout'] + [x[0] for x in COUNTS]:
            vals = [m['report'].get(field) for m in members]
            available = all(finite(v) and v >= 0 for v in vals)
            metrics[field] = {'values': vals, 'median': statistics.median(vals) if three and available else None,
                              'min': min(vals) if three and available else None,
                              'max': max(vals) if three and available else None,
                              'status': 'MEASURED' if available else 'OPEN_MISSING_COUNT'}
        status_counts = dict(collections.Counter(m['report'].get('status', 'OPEN') for m in members))
        termination_summary = {'status_counts': status_counts,
                               'Armijo_stop_present': 'stopped_no_armijo_accept' in status_counts,
                               'normal_small_step_stop_present': 'stopped_small_step' in status_counts,
                               'wall_interpretation': 'Shorter executed wall after Armijo stop is not presumed acceleration.'}
        for field in ['outer_records', 'accepted_outer_updates']:
            values = [m['report'].get(field) for m in members]
            termination_summary[field + '_range'] = [min(values), max(values)] if all(finite(v) for v in values) else None
        table.append({'object': obj, 'policy': policy, 'geometry_cache': cache, 'driver_role': role,
                      'physics_identity': physics, 'version': version, 'config_sha256': config,
                      'repetitions': reps, 'n': len(members), 'mode': mode, 'metrics': metrics,
                      'result_sha256': [m['report']['result_sha256'] for m in members],
                      'termination_records': [{k: m['report'].get(k) for k in
                          ['result_sha256', 'repetition', 'status', 'outer_records', 'accepted_outer_updates']}
                          for m in members],
                      'termination_summary': termination_summary,
                      'comparisons': [m['comparison'] for m in members]})
    return {'schema': 'a17.owner.efficiency.pareto.plot_data.v1', 'groups': table,
            'open_comparisons': opened, 'frozen_context_only': frozen,
            'gate_assignment': 'ROOT_REVIEW_REQUIRED',
            'statistical_scope': 'Timing repeats of the same object; not independent scene validation.',
            'raw_array_policy': 'Only explicit hash-bound reports with measured array checks are plotted.'}


def label(group):
    name, color = NAMES[group['policy']]
    name += ' / cached' if group['geometry_cache'] else ' / uncached'
    if group['driver_role'] == 'REUSE_SECONDARY':
        name += ' / secondary single run' if group['n'] == 1 else ' / secondary runs (unpooled)'
    elif group['driver_role'] == 'TOP1CAP3_CONDITIONAL':
        name += ' / CONDITIONAL driver'
    name += ' / 3 timing repeats' if group['mode'] == 'timing_three_median_range' else ' / n=' + str(group['n'])
    stops = group.get('termination_records', [])
    status_set = {r.get('status') for r in stops}
    def update_range(field):
        values = [r.get(field) for r in stops]
        if not values or not all(finite(v) for v in values):
            return 'OPEN'
        low, high = min(values), max(values)
        return str(low) if low == high else str(low) + '–' + str(high)
    if 'stopped_no_armijo_accept' in status_set:
        name += '\nArmijo stop / outer ' + update_range('outer_records') + ' / accepted ' + update_range('accepted_outer_updates')
    if 'stopped_small_step' in status_set:
        name += '\nSmall-step stop (normal termination) / outer ' + update_range('outer_records') + ' / accepted ' + update_range('accepted_outer_updates')
    return name, color


def render(tables, out):
    # Lazy visualization imports; prepare_tables remains purely testable.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    chinese_font = None
    for family in ['Arial Unicode MS', 'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', 'Heiti SC']:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        chinese_font = family
        plt.rcParams['font.family'] = [family, 'DejaVu Sans']
        break
    object_titles = {obj: name + ('\n' + OBJECTS_ZH[obj] if chinese_font else '') for obj, name in OBJECTS.items()}
    outputs = []
    physical = sorted(set(g['physics_identity'] for g in tables['groups'])) or [None]
    def save(fig, name, metadata):
        fig.tight_layout(rect=(0, .04, 1, .93));artifacts=[]
        for ext in ['pdf', 'svg', 'png']:
            p = out / (name + '.' + ext);fig.savefig(p, dpi=180, bbox_inches='tight')
            artifacts.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        outputs.append({'name': name, 'metadata': metadata, 'artifacts': artifacts,
                        'object_labels_english': OBJECTS, 'object_labels_chinese': OBJECTS_ZH,
                        'chinese_font': chinese_font,
                        'chinese_render_status': 'ENABLED' if chinese_font else 'OPEN_NO_CJK_FONT_ENGLISH_FALLBACK'});plt.close(fig)
    for pi, identity in enumerate(physical, 1):
        groups = [g for g in tables['groups'] if g['physics_identity'] == identity]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for i, obj in enumerate(OBJECTS):
            local = [g for g in groups if g['object'] == obj]
            for j, metric in enumerate(['material', 'heldout']):
                ax = axes[i, j]
                if not local:
                    ax.text(.5, .5, 'OPEN / no eligible measured comparison', ha='center', va='center', transform=ax.transAxes)
                for g in local:
                    name, color = label(g);xs = g['metrics']['child_wall_s'];ys = g['metrics'][metric]
                    secondary = g['driver_role'] == 'REUSE_SECONDARY'
                    conditional = g['driver_role'] == 'TOP1CAP3_CONDITIONAL'
                    marker = 'D' if secondary else '^' if conditional else 's' if g['geometry_cache'] else 'o'
                    if g['mode'] == 'timing_three_median_range':
                        x, y = xs['median'], ys['median']
                        ax.errorbar(x, y, xerr=[[x-xs['min']], [xs['max']-x]],
                                    yerr=[[y-ys['min']], [ys['max']-y]], fmt=marker, color=color,
                                    capsize=3, markersize=5, label=name)
                    else:
                        for n, (x, y) in enumerate(zip(xs['values'], ys['values'])):
                            ax.scatter(x, y, marker=marker, s=45, facecolors='none', edgecolors=color,
                                       label=name if n == 0 else None)
                ax.set(title=object_titles[obj], xlabel='Total child reconstruction wall (s)',
                       ylabel='Final complex-material relative error' if metric == 'material' else 'Final held-out data relative error')
                ax.grid(alpha=.15)
                if local:ax.legend(fontsize=6, frameon=False)
        fig.suptitle('Owner-bound nonlinear quality–wall measurements / physics group ' + str(pi) + '\nTiming ranges; secondary single runs are hollow diamonds. OPEN excluded comparisons: ' + str(len(tables['open_comparisons'])), fontsize=11)
        save(fig, f'owner_quality_wall_physics_{pi:02d}', {'physics_identity': identity, 'groups': groups,
            'open_count': len(tables['open_comparisons']), 'scope': 'AUDIT.comparisons only'})
        for obj in OBJECTS:
            local = [g for g in groups if g['object'] == obj]
            fig, axes = plt.subplots(2, 3, figsize=(17, max(7, len(local)*.48)))
            for ax, (field, title) in zip(axes.flat, COUNTS):
                for pos, g in enumerate(local):
                    name, color = label(g);m = g['metrics'][field]
                    if m['status'] != 'MEASURED':
                        ax.text(.5, pos, 'OPEN / count not saved', transform=ax.get_yaxis_transform(), fontsize=7);continue
                    if g['mode'] == 'timing_three_median_range':
                        center=m['median'];ax.errorbar(center,pos,xerr=[[center-m['min']],[m['max']-center]],fmt='o',color=color,capsize=3)
                    else:
                        for k, value in enumerate(m['values']):
                            ax.scatter(value,pos+.07*k,marker='D' if g['driver_role']=='REUSE_SECONDARY' else '^' if g['driver_role']=='TOP1CAP3_CONDITIONAL' else 'o',facecolors='none',edgecolors=color,s=35)
                ax.set_yticks(range(len(local)),[label(g)[0]for g in local],fontsize=6)
                ax.set(title=title,xlabel='Complete run physical count');ax.grid(axis='x',alpha=.15)
                if not local:ax.text(.5,.5,'OPEN / no eligible runs',ha='center',transform=ax.transAxes)
            fig.suptitle(object_titles[obj] + ' / complete physical workload / physics group ' + str(pi),fontsize=11)
            save(fig,f'owner_physical_counts_{obj}_physics_{pi:02d}',{'object':obj,'physics_identity':identity,'groups':local,'counts':COUNTS})
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=EFF/'analysis/owner_nonlinear_v2/AUDIT.json')
    p.add_argument('--analysis',type=Path,default=EFF/'analysis/derived_v3')
    p.add_argument('--out',type=Path,default=EFF/'analysis/owner_nonlinear_v2/figures_owner')
    args=p.parse_args();inputs=[]
    def read(path):
        raw=path.read_bytes();inputs.append({'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)});return json.loads(raw)
    audit=read(args.audit);analysis=read(args.analysis/'analysis_receipt.json');frozen=read(args.analysis/'frozen_object_policy.json')
    tables=prepare_tables(audit,analysis,frozen);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'plot_data.json').write_text(json.dumps(tables,indent=2,allow_nan=False)+'\n')
    figures=render(tables,args.out)
    inputs.append({'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    manifest={'schema':'a17.owner.efficiency.figure_manifest.v1','source_inputs':inputs,
              'audit_declared_input_manifest':audit.get('input_manifest',[]),'audit_implementation_sha256':audit.get('source_sha256'),
              'figures':figures,'open_comparisons':tables['open_comparisons'],'gate_assignment':'ROOT_REVIEW_REQUIRED',
              'limits':['Frozen aggregates are context only; no frozen/nonlinear pooling.',
                        'Audit-declared source/config/array hashes are recorded, not independently reread by plotter.',
                        'No fastest-repeat selection, certification or scientific recommendation.',
                        'An Armijo stop shortens the executed trajectory; its shorter wall is not presumed to be acceleration.',
                        'Plots show measurements, not a computed Pareto-optimality certificate.']}
    (args.out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'groups':len(tables['groups']),'open_comparisons':len(tables['open_comparisons']),'figures':len(figures),'out':str(args.out)}))


if __name__=='__main__':main()
