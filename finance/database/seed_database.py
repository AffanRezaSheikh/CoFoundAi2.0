import json
from database.db import get_db

db = get_db()
cursor = db.cursor()

with open("../data/transactions.json", "r") as f:
    transactions = json.load(f)

query = """
INSERT INTO transactions
(
transaction_id,
transaction_date,
direction,
category,
description,
vendor,
customer,
amount,
payment_method,
status,
department
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

for t in transactions:
    cursor.execute(query, (
        t["transaction_id"],
        t["transaction_date"],
        t["direction"],
        t["category"],
        t["description"],
        t["vendor"],
        t["customer"],
        t["amount"],
        t["payment_method"],
        t["status"],
        t["department"]
    ))

db.commit()
db.close()

print("✅ Transactions seeded successfully!")