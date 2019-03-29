import sqlite3

def main():
    db = sqlite3.connect("example.db")
    # Get a cursor object
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE result(id INTEGER PRIMARY KEY, name TEXT,
                        phone TEXT, email TEXT unique, password TEXT)
    ''')
    db.commit()

    for i in range(100):
        cursor.execute(''' 
        INSERT INTO result VALUES (?,?,?,?,?)
        ''',(i,i,i,i,i))
    pass
    db.commit()
    db.close()

if __name__ == "__main__":
    main()