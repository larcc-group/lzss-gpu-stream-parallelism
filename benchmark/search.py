import sqlite3
db = sqlite3.connect("benchmark.db")
# Get a cursor object
cursor = db.cursor()
cursor.execute("select * from result")
for row in cursor.fetchall():
    # for i in range(len(row)):
    print(row)