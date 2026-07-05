"""
Interview Report — generates a recruiter-facing HTML report from the
output of Interview pipeline.py (cv_data + questions + scoring + aggregated_scores).

Visually matches the HireMatrix theme: white background, light-orange accents,
rounded Apple-style edges. Print-friendly so the recruiter can save as PDF.
"""

from datetime import datetime
from typing import Dict, Optional


THEME = {
    "accent": "#ff8a3d",            # primary orange
    "accent_soft": "#ffe9d6",       # soft orange for callouts
    "accent_dark": "#d96a1f",       # for headings
    "ink": "#1d1d1f",               # near-black Apple text
    "muted": "#6e6e73",
    "border": "#f0e3d3",
    "surface": "#ffffff",
    "panel": "#fff8f1",             # very-soft orange tint
}


def _verdict_info(rec: str, overall: float):
    rec = (rec or "").lower()
    if rec == "strong_hire":
        return {
            "label": "STRONG HIRE",
            "color": "#16a34a",
            "bg": "#dcfce7",
            "summary": "Exceptional performance across the interview. Recommended for immediate next round.",
        }
    if rec == "hire":
        return {
            "label": "HIRE",
            "color": "#16a34a",
            "bg": "#dcfce7",
            "summary": "Solid performance overall. Recommended to proceed.",
        }
    if rec == "maybe":
        return {
            "label": "BORDERLINE — REVIEW",
            "color": "#d97706",
            "bg": "#fef3c7",
            "summary": "Candidate shows partial fit. Manual review of weak areas suggested before proceeding.",
        }
    return {
        "label": "DO NOT PROCEED",
        "color": "#dc2626",
        "bg": "#fee2e2",
        "summary": "Performance does not meet the bar for this role. Not recommended.",
    }


def _ai_risk_badge(level: str):
    level = (level or "low").lower()
    if level == "high":
        return ("AI-Generated Risk: HIGH", "#dc2626", "#fee2e2")
    if level == "medium":
        return ("AI-Generated Risk: Medium", "#d97706", "#fef3c7")
    return ("AI-Generated Risk: Low", "#16a34a", "#dcfce7")


def _bar(pct: float, color: str = None) -> str:
    color = color or THEME["accent"]
    pct = max(0, min(100, pct))
    return (
        f'<div style="background:{THEME["border"]};border-radius:999px;'
        f'height:10px;width:100%;overflow:hidden;">'
        f'<div style="background:{color};height:100%;width:{pct}%;'
        f'border-radius:999px;"></div></div>'
    )


def _question_block(idx: int, q_score: Dict) -> str:
    qtext = q_score.get("question", "")
    skill = q_score.get("skill_tested", "—")
    aggregate = q_score.get("aggregate", 0.0)
    feedback = q_score.get("feedback", "")
    flags = q_score.get("red_flags", []) or []
    keywords_used = q_score.get("keyword_hits", []) or []

    dims = ["technical_accuracy", "depth", "relevance", "clarity"]
    dim_rows = ""
    for d in dims:
        v = q_score.get(d, 0)
        try:
            v = float(v)
        except Exception:
            v = 0
        dim_rows += f"""
        <tr>
            <td style="padding:6px 0;color:{THEME['muted']};font-size:13px;width:40%;">
                {d.replace('_', ' ').title()}
            </td>
            <td style="padding:6px 0;width:50%;">{_bar(v * 10)}</td>
            <td style="padding:6px 0;text-align:right;font-weight:600;font-size:13px;">{v:.1f} / 10</td>
        </tr>"""

    ai_v = q_score.get("ai_likeness", 0)
    try:
        ai_v = float(ai_v)
    except Exception:
        ai_v = 0
    ai_color = "#dc2626" if ai_v >= 7 else ("#d97706" if ai_v >= 5 else "#16a34a")

    flag_html = ""
    if flags:
        chips = "".join(
            f'<span style="background:#fee2e2;color:#dc2626;padding:3px 10px;'
            f'border-radius:999px;font-size:12px;margin-right:6px;display:inline-block;'
            f'margin-bottom:4px;">{f}</span>'
            for f in flags
        )
        flag_html = f'<div style="margin-top:10px;">{chips}</div>'

    kw_html = ""
    if keywords_used:
        kw_html = (
            f'<div style="margin-top:10px;color:{THEME["muted"]};font-size:13px;">'
            f'<strong style="color:{THEME["ink"]};">Keywords mentioned:</strong> '
            f'{", ".join(keywords_used)}</div>'
        )

    return f"""
    <div style="background:{THEME['surface']};border:1px solid {THEME['border']};
                border-radius:18px;padding:22px 26px;margin-bottom:18px;
                box-shadow:0 1px 2px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  gap:16px;margin-bottom:14px;">
        <div style="flex:1;">
          <div style="font-size:12px;color:{THEME['accent_dark']};font-weight:600;
                      letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;">
              Question {idx} • {skill}
          </div>
          <div style="font-size:15px;color:{THEME['ink']};line-height:1.5;">{qtext}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:24px;font-weight:700;color:{THEME['accent_dark']};">
              {aggregate:.1f}<span style="font-size:14px;color:{THEME['muted']};font-weight:500;"> / 10</span>
          </div>
          <div style="font-size:11px;color:{ai_color};font-weight:600;
                      letter-spacing:0.3px;text-transform:uppercase;">
              AI-likeness {ai_v:.1f}
          </div>
        </div>
      </div>

      <table style="width:100%;border-collapse:collapse;">
        {dim_rows}
      </table>

      {f'<div style="margin-top:14px;padding:12px 16px;background:{THEME["panel"]};border-radius:12px;font-size:14px;color:{THEME["ink"]};line-height:1.5;"><strong style="color:{THEME["accent_dark"]};">Feedback:</strong> {feedback}</div>' if feedback else ''}
      {kw_html}
      {flag_html}
    </div>
    """


def generate_interview_report(
    output_path: str,
    report: Dict,
    candidate_metadata: Optional[Dict] = None,
) -> str:
    """
    Build an HTML report from a `report` dict produced by InterviewPipeline.process_candidate().
    """
    cv = report.get("cv_data", {}) or {}
    contact = cv.get("contact_info", {}) or {}
    candidate_name = contact.get("name") or report.get("candidate_id") or "Candidate"

    aggregated = report.get("aggregated_scores", {}) or {}
    overall = float(aggregated.get("overall_score", 0.0))
    technical = float(aggregated.get("technical_score", 0.0))
    communication = float(aggregated.get("communication_score", 0.0))
    rec = aggregated.get("recommendation", "reject")
    rec_reason = aggregated.get("recommendation_reason", "")
    coverage = float(aggregated.get("coverage", 0.0))
    consistency = float(aggregated.get("consistency", 0.0))
    ai_risk = aggregated.get("ai_risk", "low")
    red_flags = aggregated.get("red_flags", []) or []
    by_skill = aggregated.get("by_skill", {}) or {}
    by_difficulty = aggregated.get("by_difficulty", {}) or {}

    questions = report.get("questions", []) or []
    per_q = (report.get("scoring", {}) or {}).get("per_question", []) or []

    verdict = _verdict_info(rec, overall)
    ai_label, ai_color, ai_bg = _ai_risk_badge(ai_risk)
    date_str = datetime.now().strftime("%B %d, %Y")

    skills = cv.get("skills", {})
    if isinstance(skills, dict):
        skill_chips = skills.get("all_skills", [])[:12]
    elif isinstance(skills, list):
        skill_chips = skills[:12]
    else:
        skill_chips = []
    skill_chip_html = "".join(
        f'<span style="background:{THEME["panel"]};color:{THEME["accent_dark"]};'
        f'padding:5px 12px;border-radius:999px;font-size:12px;margin:0 6px 6px 0;'
        f'display:inline-block;border:1px solid {THEME["border"]};">{s}</span>'
        for s in skill_chips
    )

    skill_rows = ""
    for skill_name, stats in by_skill.items():
        avg = stats.get("avg_aggregate", 0.0)
        skill_rows += f"""
        <tr style="border-bottom:1px solid {THEME['border']};">
            <td style="padding:12px 16px;font-weight:500;">{skill_name}</td>
            <td style="padding:12px 16px;color:{THEME['muted']};text-align:center;">{stats.get('count', 0)}</td>
            <td style="padding:12px 16px;width:40%;">{_bar(avg * 10)}</td>
            <td style="padding:12px 16px;text-align:right;font-weight:600;">{avg:.1f}</td>
        </tr>"""

    diff_rows = ""
    for diff_name, stats in by_difficulty.items():
        avg = stats.get("avg_aggregate", 0.0)
        diff_rows += f"""
        <tr style="border-bottom:1px solid {THEME['border']};">
            <td style="padding:12px 16px;font-weight:500;text-transform:capitalize;">{diff_name}</td>
            <td style="padding:12px 16px;color:{THEME['muted']};text-align:center;">{stats.get('count', 0)}</td>
            <td style="padding:12px 16px;width:40%;">{_bar(avg * 10)}</td>
            <td style="padding:12px 16px;text-align:right;font-weight:600;">{avg:.1f}</td>
        </tr>"""

    questions_html = ""
    for i, q_score in enumerate(per_q, 1):
        questions_html += _question_block(i, q_score)
    if not questions_html:
        questions_html = (
            f'<div style="padding:24px;text-align:center;color:{THEME["muted"]};'
            f'background:{THEME["panel"]};border-radius:18px;">'
            f'No question-level scoring available.</div>'
        )

    flag_summary = ""
    if red_flags:
        chips = "".join(
            f'<span style="background:#fee2e2;color:#dc2626;padding:5px 12px;'
            f'border-radius:999px;font-size:12px;margin:0 6px 6px 0;display:inline-block;">{f}</span>'
            for f in red_flags
        )
        flag_summary = f"""
        <section>
          <h2>Red Flags Across the Interview</h2>
          <div style="margin-top:10px;">{chips}</div>
        </section>"""

    role = ""
    if candidate_metadata:
        role = candidate_metadata.get("role") or candidate_metadata.get("job_role", "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HireMatrix — Interview Report — {candidate_name}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
    background: {THEME['panel']};
    color: {THEME['ink']};
    margin: 0;
    padding: 40px 20px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{
    max-width: 920px;
    margin: 0 auto;
    background: {THEME['surface']};
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(217, 106, 31, 0.08);
  }}
  header {{
    padding: 36px 44px 28px 44px;
    background: linear-gradient(135deg, {THEME['accent']} 0%, #ffae6b 100%);
    color: white;
    position: relative;
  }}
  header h1 {{ margin: 0 0 6px 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }}
  header .meta {{ font-size: 14px; opacity: 0.95; }}
  .verdict {{
    padding: 24px 44px;
    background: {verdict['bg']};
    border-left: 6px solid {verdict['color']};
  }}
  .verdict .label {{
    font-size: 22px;
    font-weight: 700;
    color: {verdict['color']};
    margin-bottom: 6px;
    letter-spacing: -0.3px;
  }}
  .verdict .summary {{ font-size: 15px; color: {THEME['ink']}; margin: 0; }}
  .verdict .reason {{ font-size: 13px; color: {THEME['muted']}; margin-top: 8px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }}

  section {{ padding: 28px 44px; border-bottom: 1px solid {THEME['border']}; }}
  section:last-of-type {{ border-bottom: none; }}
  h2 {{
    font-size: 17px;
    font-weight: 700;
    color: {THEME['ink']};
    margin: 0 0 16px 0;
    letter-spacing: -0.2px;
  }}

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }}
  .stat-card {{
    background: {THEME['panel']};
    border: 1px solid {THEME['border']};
    border-radius: 16px;
    padding: 18px 20px;
  }}
  .stat-card .label {{
    font-size: 11px;
    color: {THEME['muted']};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
    margin-bottom: 6px;
  }}
  .stat-card .value {{
    font-size: 24px;
    font-weight: 700;
    color: {THEME['accent_dark']};
    letter-spacing: -0.5px;
  }}
  .stat-card .value .unit {{ font-size: 14px; color: {THEME['muted']}; font-weight: 500; }}

  table {{ width: 100%; border-collapse: collapse; margin: 4px 0; }}
  th {{
    background: {THEME['panel']};
    text-align: left;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    color: {THEME['muted']};
    letter-spacing: 0.5px;
    border-bottom: 1px solid {THEME['border']};
  }}
  th:first-child {{ border-top-left-radius: 12px; }}
  th:last-child {{ border-top-right-radius: 12px; }}

  .ai-badge {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
    background: {ai_bg};
    color: {ai_color};
  }}

  footer {{
    padding: 22px 44px;
    background: {THEME['panel']};
    color: {THEME['muted']};
    font-size: 12px;
    text-align: center;
    border-top: 1px solid {THEME['border']};
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
    <h1>HireMatrix Interview Report</h1>
    <div class="meta">
      Candidate: <strong>{candidate_name}</strong>
      {f' &nbsp;|&nbsp; Role: <strong>{role}</strong>' if role else ''}
      &nbsp;|&nbsp; Date: {date_str}
    </div>
  </header>

  <div class="verdict">
    <div class="label">{verdict['label']}</div>
    <p class="summary">{verdict['summary']}</p>
    <div class="reason">{rec_reason}</div>
  </div>

  <section>
    <h2>Performance Overview</h2>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">Overall</div>
        <div class="value">{overall:.1f}<span class="unit"> / 10</span></div>
      </div>
      <div class="stat-card">
        <div class="label">Technical</div>
        <div class="value">{technical:.1f}<span class="unit"> / 10</span></div>
      </div>
      <div class="stat-card">
        <div class="label">Communication</div>
        <div class="value">{communication:.1f}<span class="unit"> / 10</span></div>
      </div>
      <div class="stat-card">
        <div class="label">Coverage</div>
        <div class="value">{int(coverage * 100)}<span class="unit">%</span></div>
      </div>
    </div>
    <div style="margin-top:16px;">
      <span class="ai-badge">{ai_label}</span>
      <span style="margin-left:10px;color:{THEME['muted']};font-size:13px;">
        Consistency {consistency * 100:.0f}% • {aggregated.get('scored_count', 0)} of {aggregated.get('total_questions', 0)} questions scored
      </span>
    </div>
  </section>

  <section>
    <h2>Candidate Profile</h2>
    <p style="color:{THEME['muted']};font-size:14px;margin:0 0 14px 0;">
      Extracted automatically from the uploaded resume.
    </p>
    <div style="background:{THEME['panel']};border-radius:16px;padding:18px 22px;border:1px solid {THEME['border']};">
      <div style="margin-bottom:8px;"><strong>Email:</strong> <span style="color:{THEME['muted']};">{contact.get('email') or '—'}</span></div>
      <div style="margin-bottom:8px;"><strong>Phone:</strong> <span style="color:{THEME['muted']};">{contact.get('phone') or '—'}</span></div>
      <div style="margin-bottom:14px;"><strong>Location:</strong> <span style="color:{THEME['muted']};">{contact.get('location') or '—'}</span></div>
      <div><strong style="display:block;margin-bottom:8px;">Skills:</strong>{skill_chip_html or f'<span style="color:{THEME["muted"]};">None detected</span>'}</div>
    </div>
  </section>

  <section>
    <h2>Performance by Skill</h2>
    {f'<table><thead><tr><th>Skill</th><th style="text-align:center;">Q&apos;s</th><th>Average Score</th><th style="text-align:right;">Score</th></tr></thead><tbody>{skill_rows}</tbody></table>' if skill_rows else f'<p style="color:{THEME["muted"]};">No per-skill breakdown available.</p>'}
  </section>

  <section>
    <h2>Performance by Difficulty</h2>
    {f'<table><thead><tr><th>Difficulty</th><th style="text-align:center;">Q&apos;s</th><th>Average Score</th><th style="text-align:right;">Score</th></tr></thead><tbody>{diff_rows}</tbody></table>' if diff_rows else f'<p style="color:{THEME["muted"]};">No per-difficulty breakdown available.</p>'}
  </section>

  <section>
    <h2>Question-by-Question Breakdown</h2>
    {questions_html}
  </section>

  {flag_summary}

  <footer>
    Generated by HireMatrix on {date_str} · This report is intended for internal hiring use.
  </footer>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
