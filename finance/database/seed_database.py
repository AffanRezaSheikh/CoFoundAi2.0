import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from finance.database.db import get_db

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.json")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   VARCHAR(32) PRIMARY KEY,
    transaction_date DATE,
    direction        VARCHAR(16),
    category         VARCHAR(64),
    description      VARCHAR(255),
    vendor           VARCHAR(128),
    customer         VARCHAR(128),
    amount           DECIMAL(15,2),
    payment_method   VARCHAR(64),
    status           VARCHAR(32),
    department       VARCHAR(64)
)
"""

INSERT = """
INSERT INTO transactions
(transaction_id, transaction_date, direction, category, description,
 vendor, customer, amount, payment_method, status, department)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""


def seed():
    db = get_db()
    cursor = db.cursor()

    cursor.execute(CREATE_TABLE)
    cursor.execute("DELETE FROM transactions")  # clean re-seed

    with open(DATA_PATH, "r") as f:
        transactions = json.load(f)

    for t in transactions:
        cursor.execute(INSERT, (
            t["transaction_id"], t["transaction_date"], t["direction"],
            t["category"], t["description"], t["vendor"], t["customer"],
            t["amount"], t["payment_method"], t["status"], t["department"],
        ))

    db.commit()
    cursor.execute("SELECT COUNT(*) FROM transactions")
    count = cursor.fetchone()[0]
    db.close()
    print(f"Seeded {count} transactions successfully.")


if __name__ == "__main__":
    seed()
