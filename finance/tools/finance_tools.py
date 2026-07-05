from finance.services.finance_service import (
    calculate_cash_balance,
    calculate_burn_rate,
    calculate_runway,
    calculate_monthly_profit,
    top_expense_categories,
    generate_finance_summary,
)


FINANCE_TOOLS = [
    {
        "name": "get_cash_balance",
        "description": "Get the current cash balance (total income minus total expenses)",
        "function": calculate_cash_balance,
    },
    {
        "name": "get_burn_rate",
        "description": "Get the average monthly burn rate",
        "function": calculate_burn_rate,
    },
    {
        "name": "get_runway",
        "description": "Get how many months of runway the company has left based on current cash and burn rate",
        "function": calculate_runway,
    },
    {
        "name": "get_monthly_profit",
        "description": "Get profit and margin breakdown for each month",
        "function": calculate_monthly_profit,
    },
    {
        "name": "get_top_expenses",
        "description": "Get the top expense categories ranked by total spend",
        "function": top_expense_categories,
    },
    {
        "name": "get_finance_summary",
        "description": "Get a complete financial overview: balance, burn rate, runway, top expenses, and recent transactions",
        "function": generate_finance_summary,
    },
]
