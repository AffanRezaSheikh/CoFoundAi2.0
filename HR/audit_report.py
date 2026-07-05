import re
from datetime import datetime


def _humanize(name):
    name = str(name).replace('.', ' ').replace('_', ' ')
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return name.strip().capitalize()


def _verdict_info(di):
    if di >= 0.8:
        return {
            'label': 'FAIR DECISION',
            'icon': '✅',
            'color': '#10b981',
            'bg': '#d1fae5',
            'summary': 'The selection appears fair across genders.',
            'plain': 'This shortlist meets the industry fairness standard (80% rule).'
        }
    elif di >= 0.6:
        return {
            'label': 'NEEDS REVIEW',
            'icon': '⚠️',
            'color': '#f59e0b',
            'bg': '#fef3c7',
            'summary': 'The selection shows some imbalance across genders.',
            'plain': 'This shortlist falls below the 80% fairness standard. Review before proceeding.'
        }
    else:
        return {
            'label': 'BIASED SELECTION',
            'icon': '❌',
            'color': '#ef4444',
            'bg': '#fee2e2',
            'summary': 'The selection shows significant imbalance across genders.',
            'plain': 'This shortlist shows strong bias. Do not proceed without remediation.'
        }


def _diagnosis_info(di, eo_di):
    if di < 0.8 and eo_di >= 0.8:
        return {
            'title': 'Disparity Originates From the Data, Not the AI Model',
            'icon': '📊',
            'color': '#0891b2',
            'bg': '#cffafe',
            'finding': (
                'The shortlist disparity is not introduced by the AI model.'
            ),
            'detail': (
                'When evaluating only candidates who meet the qualification criteria, '
                'the model selected men and women at comparable rates. The imbalance '
                'in the overall shortlist mirrors the imbalance in the qualified '
                'candidate pool itself — the historical data contains fewer qualified '
                'candidates from the underrepresented group.'
            ),
            'action': (
                'Remediation should focus on data sourcing and recruitment pipelines '
                'rather than retraining the model. Expanding the qualified candidate '
                'pool from the underrepresented group is the recommended path forward.'
            )
        }
    elif di < 0.8 and eo_di < 0.8:
        return {
            'title': 'Disparity Present in Both Data and Model',
            'icon': '⚠️',
            'color': '#ef4444',
            'bg': '#fee2e2',
            'finding': (
                'Bias is present at both layers — the data composition and the '
                'model behaviour.'
            ),
            'detail': (
                'The qualified candidate pool is imbalanced, AND the model selects '
                'qualified candidates from each group at materially different rates.'
            ),
            'action': (
                'Both the data pipeline and the model itself require remediation. '
                'Re-run reweighting with stronger fairness constraints and review '
                'recruitment sourcing.'
            )
        }
    elif di >= 0.8 and eo_di < 0.8:
        return {
            'title': 'Model Is Introducing Bias on Balanced Data',
            'icon': '🤖',
            'color': '#ef4444',
            'bg': '#fee2e2',
            'finding': (
                'While overall selection rates appear balanced, the model is treating '
                'equally qualified candidates differently across groups.'
            ),
            'detail': (
                'When matched on qualification level, the model selects candidates '
                'from one group at a higher rate than the other.'
            ),
            'action': (
                'The model requires retraining with stronger fairness constraints. '
                'The data composition is not the source of the issue.'
            )
        }
    else:
        return {
            'title': 'Selection Is Fair on Both Measures',
            'icon': '✅',
            'color': '#10b981',
            'bg': '#d1fae5',
            'finding': (
                'The model treats candidates fairly — in overall selection and when '
                'matched on qualification level.'
            ),
            'detail': (
                'Both the data composition and the model behaviour meet the fairness '
                'standard.'
            ),
            'action': (
                'No remediation required. Proceed with this shortlist and retain '
                'this report for compliance records.'
            )
        }


def _next_steps(di):
    if di >= 0.8:
        return [
            "Proceed with this shortlist — it meets fairness standards.",
            "Save this report in your compliance records.",
            "Re-run the audit if the candidate pool or role changes significantly."
        ]
    elif di >= 0.6:
        return [
            "Manually review the highest-scoring rejected candidates from the underrepresented group.",
            "Consider expanding the shortlist by 10–20% to reduce group disparity.",
            "Check whether any occupation-specific proxy columns were missed during auto-detection."
        ]
    else:
        return [
            "Do NOT proceed with this shortlist.",
            "Re-run the reweighting step with stronger fairness constraints.",
            "Inspect the flagged proxy columns — a strong proxy for gender may still be leaking through.",
            "If the pool is heavily imbalanced at the source, expand recruitment efforts before re-running."
        ]


def _bar(pct, color='#3b82f6', width_pct=None):
    w = width_pct if width_pct is not None else pct
    return (
        f'<div style="background:#e5e7eb;border-radius:4px;height:18px;'
        f'width:100%;overflow:hidden;">'
        f'<div style="background:{color};height:100%;width:{w}%;'
        f'border-radius:4px;"></div></div>'
    )


def generate_audit_report(
    output_path,
    dataset_name,
    model_name,
    target_occupation,
    is_occ_filtered,
    total_pool,
    gender_col,
    gender_breakdown,
    top_n,
    di,
    feature_importance,
    removed_cols,
    fairness_applied,
    eo_di=None,
    qualification_rates=None,
    eo_breakdown=None,
):
    verdict = _verdict_info(di)
    diagnosis = _diagnosis_info(di, eo_di) if eo_di is not None else None
    date_str = datetime.now().strftime("%B %d, %Y")
    role_label = target_occupation if is_occ_filtered else "all occupations"

    # Determine majority vs minority by applicant count
    sorted_groups = sorted(gender_breakdown.items(), key=lambda x: x[1]['applied'], reverse=True)
    majority_name, majority_stats = sorted_groups[0]
    minority_name, minority_stats = sorted_groups[-1]

    # Plain English DI explanation
    maj_rate_pct = round(majority_stats['rate'] * 100, 1)
    min_rate_pct = round(minority_stats['rate'] * 100, 1)

    # "For every 10 majority selected, X minority" (normalized to rates)
    if majority_stats['rate'] > 0:
        per_ten = round((minority_stats['rate'] / majority_stats['rate']) * 10, 1)
    else:
        per_ten = 0

    # Build "Who applied" bars
    max_applied = max(g['applied'] for g in gender_breakdown.values())
    applied_rows = ''
    for g, stats in gender_breakdown.items():
        pct_of_total = round(stats['applied'] / total_pool * 100, 1)
        bar_width = round(stats['applied'] / max_applied * 100, 1) if max_applied else 0
        applied_rows += f'''
        <tr>
            <td style="padding:10px 14px;font-weight:600;">{g}</td>
            <td style="padding:10px 14px;width:50%;">{_bar(bar_width, color='#ff8a3d')}</td>
            <td style="padding:10px 14px;text-align:right;">{stats['applied']} ({pct_of_total}%)</td>
        </tr>'''

    # Selection table
    selection_rows = ''
    for g, stats in gender_breakdown.items():
        rate_pct = round(stats['rate'] * 100, 1)
        selection_rows += f'''
        <tr>
            <td style="padding:10px 14px;font-weight:600;">{g}</td>
            <td style="padding:10px 14px;text-align:right;">{stats['applied']}</td>
            <td style="padding:10px 14px;text-align:right;">{stats['selected']}</td>
            <td style="padding:10px 14px;text-align:right;font-weight:600;">{rate_pct}%</td>
        </tr>'''

    # Feature importance bars (top 8)
    top_features = list(feature_importance.items())[:8]
    max_imp = max(imp for _, imp in top_features) if top_features else 1
    feature_rows = ''
    for i, (feat, imp) in enumerate(top_features, 1):
        pct = round(imp * 100, 1)
        bar_w = round(imp / max_imp * 100, 1) if max_imp else 0
        feature_rows += f'''
        <tr>
            <td style="padding:8px 14px;text-align:center;color:#6b7280;">{i}</td>
            <td style="padding:8px 14px;font-weight:500;">{_humanize(feat)}</td>
            <td style="padding:8px 14px;width:50%;">{_bar(bar_w, color='#ffae6b')}</td>
            <td style="padding:8px 14px;text-align:right;">{pct}%</td>
        </tr>'''

    top_feat_plain = _humanize(top_features[0][0]) if top_features else "experience"
    second_feat_plain = _humanize(top_features[1][0]) if len(top_features) > 1 else ""
    ai_focus_line = f"The AI mainly cared about <strong>{top_feat_plain.lower()}</strong>"
    if second_feat_plain:
        ai_focus_line += f" and <strong>{second_feat_plain.lower()}</strong>"
    ai_focus_line += " when deciding who to shortlist."

    # Removed columns list
    if removed_cols:
        removed_html = '<ul style="margin:12px 0;padding-left:24px;line-height:1.9;">'
        for col in removed_cols:
            removed_html += f'<li><strong>{_humanize(col)}</strong></li>'
        removed_html += '</ul>'
    else:
        removed_html = '<p><em>No proxy columns were flagged for removal.</em></p>'

    fairness_line = (
        "Fairness reweighting was also applied, giving historically underrepresented "
        "groups equal weight during training."
        if fairness_applied else
        "<em>Note: fairness reweighting was not applied for this run.</em>"
    )

    # Next steps
    steps_html = '<ul style="margin:12px 0;padding-left:24px;line-height:1.9;">'
    for step in _next_steps(di):
        steps_html += f'<li>{step}</li>'
    steps_html += '</ul>'

    total_selected = sum(s['selected'] for s in gender_breakdown.values())

    # Diagnosis section HTML
    diagnosis_html = ''
    if diagnosis is not None and qualification_rates and eo_breakdown:
        # Qualification rate rows
        qual_rows = ''
        for g, rate in qualification_rates.items():
            qualified_n = eo_breakdown[g]['qualified']
            total_n = gender_breakdown[g]['applied']
            rate_pct = round(rate * 100, 1)
            qual_rows += f'''
            <tr>
                <td style="padding:10px 14px;font-weight:600;">{g}</td>
                <td style="padding:10px 14px;text-align:right;">{qualified_n} / {total_n}</td>
                <td style="padding:10px 14px;text-align:right;font-weight:600;">{rate_pct}%</td>
            </tr>'''

        # Equal opportunity rows
        eo_rows = ''
        for g, stats in eo_breakdown.items():
            eo_pct = round(stats['eo_rate'] * 100, 1)
            eo_rows += f'''
            <tr>
                <td style="padding:10px 14px;font-weight:600;">{g}</td>
                <td style="padding:10px 14px;text-align:right;">{stats['qualified']}</td>
                <td style="padding:10px 14px;text-align:right;">{stats['qualified_selected']}</td>
                <td style="padding:10px 14px;text-align:right;font-weight:600;">{eo_pct}%</td>
            </tr>'''

        diagnosis_html = f'''
  <section>
    <h2>🔍 Source of Disparity — Data or Model?</h2>

    <div style="background:{diagnosis['bg']};border-left:6px solid {diagnosis['color']};
                padding:20px 24px;margin:14px 0;border-radius:6px;">
      <div style="font-size:18px;font-weight:700;color:{diagnosis['color']};margin-bottom:10px;">
        {diagnosis['icon']} {diagnosis['title']}
      </div>
      <p style="margin:6px 0;font-size:15px;color:#1f2937;"><strong>Finding:</strong> {diagnosis['finding']}</p>
      <p style="margin:6px 0;color:#374151;">{diagnosis['detail']}</p>
      <p style="margin:10px 0 0 0;color:#1f2937;"><strong>Recommended action:</strong> {diagnosis['action']}</p>
    </div>

    <h3 style="font-size:15px;color:#374151;margin-top:24px;">Qualification Rate by Group</h3>
    <p style="color:#4b5563;font-size:14px;">
      Percentage of each group in this candidate pool who meet the qualification criteria
      (this reflects what the historical data looks like, before the AI sees it):
    </p>
    <table>
      <thead>
        <tr>
          <th>Group</th>
          <th style="text-align:right;">Qualified / Total</th>
          <th style="text-align:right;">Qualification Rate</th>
        </tr>
      </thead>
      <tbody>
        {qual_rows}
      </tbody>
    </table>

    <h3 style="font-size:15px;color:#374151;margin-top:24px;">Equal Opportunity — Among Qualified Candidates Only</h3>
    <p style="color:#4b5563;font-size:14px;">
      Of the candidates who actually meet the qualification criteria,
      what percentage of each group did the AI select? This isolates the model's behaviour
      from the data composition:
    </p>
    <table>
      <thead>
        <tr>
          <th>Group</th>
          <th style="text-align:right;">Qualified</th>
          <th style="text-align:right;">Qualified & Selected</th>
          <th style="text-align:right;">Selection Rate</th>
        </tr>
      </thead>
      <tbody>
        {eo_rows}
      </tbody>
    </table>

    <div class="callout" style="margin-top:18px;">
      <strong>Equal Opportunity DI: {eo_di}</strong> &nbsp;|&nbsp;
      <strong>Overall DI: {di}</strong><br>
      <span style="font-size:13px;color:#1e3a8a;">
        Equal Opportunity DI compares only equally qualified candidates across groups.
        A high score here means the model is fair, even if the overall pool is imbalanced.
      </span>
    </div>
  </section>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hiring Fairness Report — {role_label}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
    background: #fff8f1;
    color: #1d1d1f;
    margin: 0;
    padding: 40px 20px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{
    max-width: 860px;
    margin: 0 auto;
    background: white;
    border-radius: 24px;
    box-shadow: 0 8px 40px rgba(217, 106, 31, 0.08);
    overflow: hidden;
  }}
  header {{
    background: linear-gradient(135deg, #ff8a3d 0%, #ffae6b 100%);
    color: white;
    padding: 32px 40px;
  }}
  header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
  header .meta {{ font-size: 14px; opacity: 0.9; }}
  .verdict {{
    padding: 28px 40px;
    background: {verdict['bg']};
    border-left: 6px solid {verdict['color']};
  }}
  .verdict .label {{
    font-size: 22px;
    font-weight: 700;
    color: {verdict['color']};
    margin-bottom: 8px;
  }}
  .verdict .summary {{ font-size: 16px; color: #1f2937; margin: 0; }}
  section {{ padding: 28px 40px; border-bottom: 1px solid #e5e7eb; }}
  section:last-of-type {{ border-bottom: none; }}
  h2 {{
    font-size: 18px;
    color: #111827;
    margin: 0 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e5e7eb;
  }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0; }}
  th {{
    background: #f3f4f6;
    text-align: left;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 13px;
    text-transform: uppercase;
    color: #6b7280;
    letter-spacing: 0.5px;
  }}
  tr {{ border-bottom: 1px solid #e5e7eb; }}
  tr:last-child {{ border-bottom: none; }}
  .callout {{
    background: #fff8f1;
    border-left: 4px solid #ff8a3d;
    padding: 14px 18px;
    margin: 14px 0;
    border-radius: 12px;
    font-size: 14px;
    color: #d96a1f;
  }}
  footer {{
    padding: 20px 40px;
    background: #f9fafb;
    color: #6b7280;
    font-size: 13px;
    text-align: center;
  }}
  @media print {{
    body {{ background: white; padding: 0; }}
    .container {{ box-shadow: none; border-radius: 0; }}
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>🎯 Hiring Fairness Report</h1>
    <div class="meta">
      Role: <strong>{role_label}</strong> &nbsp;|&nbsp; Date: {date_str}
    </div>
  </header>

  <div class="verdict">
    <div class="label">{verdict['icon']} {verdict['label']}</div>
    <p class="summary">
      Out of <strong>{total_pool}</strong> people who applied for the
      <strong>{role_label}</strong> role, the AI selected <strong>{total_selected}</strong> candidates.
      {verdict['summary']}
    </p>
  </div>

  <section>
    <h2>📊 Who Applied</h2>
    <p>Total applicants for this role: <strong>{total_pool}</strong></p>
    <table>
      {applied_rows}
    </table>
  </section>

  <section>
    <h2>✅ Who the AI Selected</h2>
    <p>
      Out of every 100 <strong>{majority_name}</strong> who applied, about <strong>{maj_rate_pct}</strong> were selected.<br>
      Out of every 100 <strong>{minority_name}</strong> who applied, about <strong>{min_rate_pct}</strong> were selected.
    </p>
    <table>
      <thead>
        <tr>
          <th>Group</th>
          <th style="text-align:right;">Applied</th>
          <th style="text-align:right;">Selected</th>
          <th style="text-align:right;">Selection Rate</th>
        </tr>
      </thead>
      <tbody>
        {selection_rows}
      </tbody>
    </table>
  </section>

  <section>
    <h2>⚖️ Was the Selection Fair?</h2>
    <div class="callout">
      <strong>Industry rule (80% Rule):</strong> For every 10 people selected from the
      majority group, at least 8 from the minority group should also be selected.
      This is the standard used by the U.S. Equal Employment Opportunity Commission.
    </div>
    <p>
      <strong>Your result:</strong> For every 10 {majority_name} selected (proportionally),
      <strong>{per_ten}</strong> {minority_name} were selected.
    </p>
    <p style="font-size:16px;font-weight:600;color:{verdict['color']};">
      Verdict: {verdict['plain']}
    </p>
  </section>
{diagnosis_html}
  <section>
    <h2>🤖 What Did the AI Look At?</h2>
    <p>The AI ranked each candidate based mostly on these factors:</p>
    <table>
      <thead>
        <tr>
          <th style="text-align:center;width:40px;">#</th>
          <th>Factor</th>
          <th>Weight</th>
          <th style="text-align:right;">%</th>
        </tr>
      </thead>
      <tbody>
        {feature_rows}
      </tbody>
    </table>
    <p style="margin-top:14px;color:#4b5563;">
      <strong>In plain terms:</strong> {ai_focus_line}
    </p>
  </section>

  <section>
    <h2>🛡️ What Was Hidden From the AI (to Prevent Bias)</h2>
    <p>The following details were <strong>not shown</strong> to the AI, so they could not
    influence the decision:</p>
    {removed_html}
    <p style="color:#4b5563;margin-top:14px;">{fairness_line}</p>
  </section>

  <section>
    <h2>📝 What You Should Do Next</h2>
    {steps_html}
  </section>

  <footer>
    Generated on {date_str} &nbsp;|&nbsp; Model: {model_name} &nbsp;|&nbsp; Dataset: {dataset_name}
  </footer>

</div>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path
