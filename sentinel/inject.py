import sqlite3
import datetime

db = sqlite3.connect('data/sentinel.db')
cursor = db.cursor()

# Создаем тестовую открытую позицию LONG
cursor.execute('''
    INSERT INTO positions (symbol, side, entry_price, quantity, current_price, status, opened_at, is_paper)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', ('BTCUSDT', 'LONG', 85000.0, 0.015, 85000.0, 'OPEN', datetime.datetime.now(datetime.timezone.utc).isoformat(), 1))

db.commit()
db.close()
print("Position injected!")
