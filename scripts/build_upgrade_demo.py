"""Execute a short evidence tour of the two VLMOP2 experiments; no optimizer runs."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})

    def code(text):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(True)})

    md('# ParEGO implementation fix and fixed-budget performance\n\nHow does the official fix change '
       'optimization performance, and how does corrected native ParEGO compare with HVPI, NDS and Random? '
       'Both experiments use VLMOP2. This notebook reads existing evidence without fitting models or collecting timings.')
    code('''from pathlib import Path
import json
root = Path.cwd()
if root.name == 'notebooks':
    root = root.parent
stats = json.loads((root / 'results/upgrade/statistics.json').read_text())
assert stats['runs_audited'] == 120
print('Retained evidence: 120 VLMOP2 trajectories / 6000 evaluations')
''')
    md('## Experiment 1 - Effect of the ParEGO implementation fix\n\nBoth operators run inside '
       'PHYSBO 3.2.1 with paired initial points, GP settings and an independently logged weight stream. '
       'The official correction belongs to PHYSBO maintainers. This isolates the reduction; it does not '
       'reproduce complete earlier releases.')
    code('''for method in ['ParEGO-max', 'ParEGO-min']:
    row = next(r for r in stats['summary'] if r['block'] == 'version' and r['method'] == method and r['evaluations'] == 50)
    print(method, 'final raw HV:', round(row['mean'], 4), '95% interval:', (round(row['low'], 4), round(row['high'], 4)))
pair = stats['paired_comparisons'][0]
print('Paired min-minus-max raw-HV difference:', round(pair['mean'], 4), '95% interval:', (round(pair['low'], 4), round(pair['high'], 4)))
''')
    md('![Controlled operator effect](../figures/upgrade/version_comparison.png)\n\nThe corrected min '
       'operator improves observed mean final quality in this controlled comparison.')
    md('## Experiment 2 - Corrected ParEGO under a fixed evaluation budget\n\nNative ParEGO, HVPI, '
       'NDS and Random share ten initial observations and a 50-evaluation cap. The native ParEGO weight '
       'mechanism differs from the diagnostic paired stream; the two conditions are not pooled.')
    code('''methods = ['ParEGO', 'HVPI', 'NDS', 'Random']
for method in methods:
    row = next(r for r in stats['summary'] if r['block'] == 'multi' and r['method'] == method and r['evaluations'] == 50)
    print(method, 'final raw HV:', round(row['mean'], 4), '95% interval:', (round(row['low'], 4), round(row['high'], 4)),
          '90% target hits:', f"{round(row['target_hit_rate']*20)}/20", 'decision seconds:', round(row['mean_decision_s'], 5))
''')
    md('![Fixed-budget strategies](../figures/upgrade/multi_quality.png)\n\nAt this fixed budget '
       'native ParEGO has a lower observed mean final HV than HVPI and a higher mean than NDS and Random. '
       'This statement is conditional on the current task and settings; it is not a new pairwise significance claim.')
    code('''for n in [10, 20, 50]:
    means = {m: round(next(r['mean'] for r in stats['summary'] if r['block'] == 'multi' and r['method'] == m and r['evaluations'] == n), 4) for m in methods}
    print('Queries:', n, 'mean raw HV:', means)
for target in [.8, .9, .95]:
    hits = {m: next(r['successes'] for r in stats['target_sensitivity'] if r['block'] == 'multi' and r['method'] == m and r['target'] == target) for m in methods}
    print('Grid-HV target:', target, 'hits out of 20:', hits)
''')
    md('## Evidence boundary\n\nThese are existing summaries of the same retained runs: 40 controlled '
       'operator trajectories and 80 native strategy trajectories. Original observations and timings '
       'remain unchanged. Full-grid HV and targets are used only for offline evaluation. '
       'Algorithm overhead is reported separately from quality; no physical experiment is performed.')

    namespace = {'__name__': '__evidence_tour__'}
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        count = 0
        for cell in cells:
            if cell['cell_type'] != 'code':
                continue
            count += 1
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(compile(''.join(cell['source']), '<evidence-tour>', 'exec'), namespace)
            cell['execution_count'] = count
            cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': buffer.getvalue().splitlines(True)}]
    finally:
        os.chdir(previous)
    for i, cell in enumerate(cells):
        cell['id'] = f'vlmop2-two-{i:02d}'
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': sys.version.split()[0]},
                             'execution': 'Four evidence-reading cells executed without optimizer runs.'}}
    path = ROOT / 'notebooks/upgrade_demo.ipynb'
    path.write_text(json.dumps(notebook, indent=2)+'\n', encoding='utf-8')
    print(f'Executed {count} evidence cells; no optimizer runs')


if __name__ == '__main__':
    main()
