from finance.database.finance_repository import (
    get_total_income,
    get_total_expense,
    get_recent_transactions,
    get_transactions_by_category,
    get_monthly_totals,
)


def calculate_cash_balance():
    income = get_total_income()
    expense = get_total_expense()
    return income - expense


def calculate_burn_rate():
    rows = get_monthly_totals()
    expense_months = [r for r in rows if r["direction"] == "Expense"]
    if not expense_months:
        return 0.0
    total_expense = sum(float(r["total"]) for r in expense_months)
    num_months = len(expense_months)
    return total_expense / num_months


def calculate_runway():
    cash = calculate_cash_balance()
    burn = calculate_burn_rate()
    if burn == 0:
        return float("inf")
    return round(cash / burn, 2)


def calculate_monthly_profit():
    rows = get_monthly_totals()
    months = {}
    for r in rows:
        key = f"{int(r['yr'])}-{int(r['mo']):02d}"
        if key not in months:
            months[key] = {"income": 0.0, "expense": 0.0}
        months[key][r["direction"].lower()] += float(r["total"])

    result = []
    for month, vals in sorted(months.items()):
        profit = vals["income"] - vals["expense"]
        margin = (profit / vals["income"] * 100) if vals["income"] > 0 else 0.0
        result.append({
            "month": month,
            "income": vals["income"],
            "expense": vals["expense"],
            "profit": round(profit, 2),
            "margin": round(margin, 2),
        })
    return result


def top_expense_categories(limit=5):
    rows = get_transactions_by_category(direction="Expense")
    return [{"category": r["category"], "total": float(r["total"])} for r in rows[:limit]]


def generate_finance_summary():
    income = get_total_income()
    expense = get_total_expense()
    balance = income - expense
    burn = calculate_burn_rate()
    runway = calculate_runway()
    top_expenses = top_expense_categories()
    recent = get_recent_transactions(5)

    return {
        "total_income": income,
        "total_expense": expense,
        "cash_balance": balance,
        "monthly_burn_rate": round(burn, 2),
        "runway_months": runway,
        "top_expense_categories": top_expenses,
        "recent_transactions": recent,
    }
