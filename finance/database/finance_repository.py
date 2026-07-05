from finance.database.db import get_db


def get_total_income():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT COALESCE(SUM(amount), 0) AS total FROM transactions WHERE direction='Income'")
    result = cursor.fetchone()["total"]
    db.close()
    return float(result)


def get_total_expense():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT COALESCE(SUM(amount), 0) AS total FROM transactions WHERE direction='Expense'")
    result = cursor.fetchone()["total"]
    db.close()
    return float(result)


def get_recent_transactions(limit=10):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM transactions ORDER BY transaction_date DESC LIMIT %s", (limit,))
    rows = cursor.fetchall()
    db.close()
    return rows


def get_transactions_by_month(year, month):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM transactions WHERE YEAR(transaction_date) = %s AND MONTH(transaction_date) = %s ORDER BY transaction_date",
        (year, month)
    )
    rows = cursor.fetchall()
    db.close()
    return rows


def get_transactions_by_category(direction=None):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    if direction:
        cursor.execute(
            "SELECT category, SUM(amount) AS total FROM transactions WHERE direction = %s GROUP BY category ORDER BY total DESC",
            (direction,)
        )
    else:
        cursor.execute("SELECT category, direction, SUM(amount) AS total FROM transactions GROUP BY category, direction ORDER BY total DESC")
    rows = cursor.fetchall()
    db.close()
    return rows


def get_monthly_totals():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT
            YEAR(transaction_date) AS yr,
            MONTH(transaction_date) AS mo,
            direction,
            SUM(amount) AS total
        FROM transactions
        GROUP BY yr, mo, direction
        ORDER BY yr, mo
    """)
    rows = cursor.fetchall()
    db.close()
    return rows


def get_transaction_count():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) AS cnt FROM transactions")
    result = cursor.fetchone()["cnt"]
    db.close()
    return result


def insert_transaction(data):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO transactions
        (transaction_id, transaction_date, direction, category, description, vendor, customer, amount, payment_method, status, department)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        data["transaction_id"], data["transaction_date"], data["direction"],
        data["category"], data["description"], data.get("vendor"),
        data.get("customer"), data["amount"], data["payment_method"],
        data["status"], data["department"]
    ))
    db.commit()
    db.close()
    return data["transaction_id"]


def delete_transaction(transaction_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM transactions WHERE transaction_id = %s", (transaction_id,))
    affected = cursor.rowcount
    db.commit()
    db.close()
    return affected
