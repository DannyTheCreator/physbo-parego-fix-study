"""Build the ParEGO fix-effect and fixed-budget strategy comparison report."""
from pathlib import Path
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
TITLE = 'ParEGO Implementation Fix and Fixed-Budget Performance'
QUESTION = ('How does the PHYSBO ParEGO implementation fix affect optimization performance, '
            'and how does corrected ParEGO compare with HVPI, NDS and Random under a fixed evaluation budget?')


def main():
    data = json.loads((ROOT / 'results/upgrade/statistics.json').read_text())
    if data['runs_audited'] != 120:
        raise ValueError('report requires 120 retained VLMOP2 trajectories')
    summary = data['summary']

    def get(block, method, n=50):
        rows = [r for r in summary if r['block'] == block and r['method'] == method and r['evaluations'] == n]
        if len(rows) != 1 or rows[0]['n_seeds'] != 20:
            raise ValueError('expected one result from 20 seeds')
        return rows[0]

    def ci(row):
        return f"{row['mean']:.4f} [{row['low']:.4f}, {row['high']:.4f}]"

    pair = next(r for r in data['paired_comparisons'] if r['comparison'] == 'current min minus legacy max')
    operator_rows = [get('version', m) for m in ['ParEGO-max', 'ParEGO-min']]
    methods = ['ParEGO', 'HVPI', 'NDS', 'Random']
    strategy_rows = [get('multi', m) for m in methods]
    operator_table = [['Controlled operator', 'Final raw HV mean [95% interval]']]
    operator_table += [[r['method'], ci(r)] for r in operator_rows]
    strategy_table = [['Strategy', 'Final raw HV mean [95% interval]', '90% target hits', 'Decision seconds']]
    strategy_table += [[r['method'], ci(r), f"{round(r['target_hit_rate']*20)}/20",
                        f"{r['mean_decision_s']:.3f}" if r['mean_decision_s'] >= .001 else '<0.001'] for r in strategy_rows]
    budgets = [['Queries', *methods]]
    budgets += [[str(n), *[f"{get('multi', m, n)['mean']:.4f}" for m in methods]] for n in [10, 20, 50]]
    sensitivity = [['Grid-HV target', *methods]]
    for target in [.8, .9, .95]:
        rows = [next(r for r in data['target_sensitivity'] if r['block'] == 'multi'
                     and r['method'] == m and r['target'] == target) for m in methods]
        sensitivity.append([f'{target:.0%}', *[f"{r['successes']}/{r['total']}" for r in rows]])

    setup = ('Both experiments use PHYSBO 3.2.1 and the fixed VLMOP2 two-objective function on a '
             '101 x 101 grid over [-2,2]^2, with reference (-1,-1). Each trajectory has 50 total '
             'evaluations including ten shared random initial candidates. Seeds are 10-29; exact GP '
             'is used and retrained every ten probes. Hypervolume (HV) is the dominated trade-off area '
             'relative to the fixed reference; larger is better. Full-grid HV is approximately 0.334518. '
             'The 90% target is approximately 0.301066. The full grid is evaluated only for offline reporting.')
    effect = ('Experiment 1 changes only the weighted Chebyshev reduction inside PHYSBO 3.2.1: '
              'diagnostic max versus corrected min. Initial points, normalization, GP settings and '
              'an independently logged weight stream are paired. For weights (0.5,0.5), extreme '
              'points (-1,0) and (0,-1) score -0.525 under min and -0.025 under max; balanced point '
              '(-0.4,-0.4) scores -0.220 under both. The ranking changes, followed by different search trajectories.')
    effect_result = ('At 50 evaluations, the paired raw-HV difference (min minus max) is ' + ci(pair) + '. '
                     'The corrected reduction improves mean final quality in this controlled experiment. '
                     'The official fix is upstream PHYSBO work. This is an operator-only diagnostic '
                     'within 3.2.1, not an end-to-end comparison of complete software releases.')
    comparison = ('Experiment 2 uses the corrected native library strategies. Native ParEGO has mean final '
                  'raw HV 0.2713, compared with 0.3270 for HVPI, 0.2516 for NDS and 0.1922 for Random. '
                  'The 90% target is reached in 4/20, 20/20, 6/20 and 0/20 runs, respectively. '
                  'Thus the fix improves controlled ParEGO behavior, while native ParEGO still has lower '
                  'observed mean final HV than HVPI at the same evaluation cap. ParEGO has a higher '
                  'observed mean than NDS and Random; no new pairwise significance test is claimed.')
    separation = ('The corrected diagnostic ParEGO-min mean in Experiment 1 is 0.2762, whereas native ParEGO '
                  'in Experiment 2 is 0.2713. These are separate conditions with different weight mechanisms: '
                  'the diagnostic uses an independent logged paired weight stream; native ParEGO uses the '
                  'library mechanism. Their records must not be pooled or treated as the same branch.')
    limits = ('The conclusions concern this VLMOP2 grid, 20 seeds, fixed model settings and evaluation cap. '
              'The existing 10/20/50-query summaries and 80%/90%/95% targets describe budget and target '
              'dependence within the same runs. Pointwise 95% quality intervals use 10,000 percentile '
              'seed-bootstrap draws with seed 20261015. Success denominators retain all runs. '
              'This evidence does not establish a universal ranking, a win in every seed, or physical '
              'experimental performance. Algorithm time is a separate local computation measurement.')
    provenance = ('The active dataset is a documented subset of the earlier acquisition: 120 retained '
                  'trajectories / 6,000 charged evaluations, comprising 40 controlled-operator and '
                  '80 native-strategy trajectories. Original retained per-run bytes and timestamps are '
                  'unchanged. Scope metadata and derived tables were updated without rerunning these '
                  'experiments. Original collector snapshots and historical validation are retained as '
                  'provenance, not relabeled as a new acquisition or review. The official algorithms and '
                  'fix belong to PHYSBO maintainers; project implementation and reporting were AI-assisted.')

    def md_table(rows):
        return ['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---']*len(rows[0])) + ' |',
                *['| ' + ' | '.join(row) + ' |' for row in rows[1:]]]

    lines = ['# ' + TITLE, '', '**Research question:** ' + QUESTION,
             '', '*Local scope revision: 16 October 2026. Two VLMOP2 experiments.*',
             '', '## Shared experimental setup', setup,
             '', '## Experiment 1 - Effect of the ParEGO implementation fix', effect,
             '', *md_table(operator_table), '', effect_result,
             '', '![Controlled ParEGO operator comparison](../figures/upgrade/version_comparison.png)',
             '', '## Experiment 2 - Corrected ParEGO under a fixed evaluation budget', comparison,
             '', *md_table(strategy_table), '', separation,
             '', '![Native strategies at fixed query counts](../figures/upgrade/multi_quality.png)',
             '', '### Existing query-budget summaries: mean raw HV', *md_table(budgets),
             '', '### Existing target sensitivity: successes out of 20 at 50 evaluations', *md_table(sensitivity),
             '', '## Interpretation and limits', limits,
             '', '## Provenance and reproduction', provenance,
             '', 'Verify with `python scripts/run_upgrade.py --verify-only`. For a new run use a fresh output '
             'directory. See [protocol](PROTOCOL.md), [reproduction](REPRODUCTION.md), '
             '`results/upgrade/statistics.json`, `summary.csv`, `paired_comparisons.csv` and `target_sensitivity.csv`.',
             '', '## Sources',
             '- [Official PHYSBO 3.2.1 correction](https://github.com/issp-center-dev/PHYSBO/releases/tag/v3.2.1)',
             '- [PHYSBO update paper, arXiv v1](https://arxiv.org/html/2603.01349v1)', '']
    (ROOT / 'docs/REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('StudyTitle', parent=styles['Title'], fontName='Helvetica-Bold',
                              fontSize=20, leading=23, alignment=TA_LEFT, spaceAfter=9,
                              textColor=colors.HexColor('#17324D')))
    styles.add(ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9, leading=12, spaceAfter=7))
    styles.add(ParagraphStyle('Small', parent=styles['BodyText'], fontSize=7.7, leading=10, spaceAfter=5))
    styles['Heading2'].textColor = colors.HexColor('#17324D')
    story = []

    def p(text, small=False):
        story.append(Paragraph(text, styles['Small' if small else 'Body']))

    def head(text):
        story.append(Paragraph(text, styles['Heading2']))

    def table(rows, widths):
        tab = Table([[Paragraph(escape(str(c)), styles['Small']) for c in row] for row in rows], colWidths=widths)
        tab.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E9F0F6')),
                                 ('LINEBELOW', (0,0), (-1,0), .5, colors.HexColor('#7891A8')),
                                 ('VALIGN', (0,0), (-1,-1), 'TOP'), ('BOTTOMPADDING', (0,0), (-1,-1), 4)]))
        story.append(tab)

    def image(name, width):
        from reportlab.lib.utils import ImageReader
        path = ROOT / 'figures/upgrade' / name
        w, h = ImageReader(str(path)).getSize()
        story.append(Image(str(path), width=width, height=width*h/w))

    story.append(Paragraph('ParEGO Implementation Fix<br/>and Fixed-Budget Performance', styles['StudyTitle']))
    p(QUESTION)
    p('PHYSBO 3.2.1 | VLMOP2 | 20 seeds | 50 evaluations per run | Local revision 2026-10-16', True)
    p(setup, True)
    head('Experiment 1 - Effect of the implementation fix')
    p(effect)
    table(operator_table, [170, 325])
    p(effect_result)
    image('version_comparison.png', 475)
    p(separation, True)
    p('The study has two experiments: controlled fix effect, then the corrected native strategy comparison.', True)
    story.append(PageBreak())
    head('Experiment 2 - Fixed-budget strategy comparison')
    p(comparison)
    table(strategy_table, [60, 225, 100, 110])
    image('multi_quality.png', 390)
    p('<b>Query-budget summaries - mean raw HV</b>', True)
    table(budgets, [95, 100, 100, 100, 100])
    p('<b>Target sensitivity - successes out of 20</b>', True)
    table(sensitivity, [95, 100, 100, 100, 100])
    p(limits, True)
    p('Retained evidence: 120 original trajectories / 6,000 evaluations. Verify with '
      'python scripts/run_upgrade.py --verify-only. Original per-run values were not changed or rerun. '
      'Source links, provenance, attribution and reproduction details are in docs/REPORT.md.', True)

    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#65717D'))
        canvas.drawString(50, 27, 'Independent PHYSBO study | Two VLMOP2 experiments | 2026-10-16')
        canvas.drawRightString(545, 27, str(doc.page))

    output = ROOT / 'output/pdf/physbo_upgrade_report.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=(595.28, 841.89), leftMargin=50, rightMargin=50,
                            topMargin=38, bottomMargin=40, title=TITLE, author='Independent PHYSBO study')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print('Wrote two-experiment Markdown/PDF report from 120 retained VLMOP2 trajectories')


if __name__ == '__main__':
    main()
