import sqlite3

conn = sqlite3.connect("Data/Users.db")

c = conn.cursor()
#Datatypes: NULL, INTEGER, REAL, TEXT, BLOB
SQLStr = """
SELECT * 
FROM Users
"""

SQLStr2 = """
INSERT INTO Users (first_name, last_name, username, password, email, admin, id) 
VALUES ('yaakov', 'Levy', 'y123', 'yyy', 'levy@gmail.com', 0, 023534175)
"""
c.execute(SQLStr2)
conn.commit()
users = c.fetchall()
conn.close()
print(type(users))
for user in users:
    print(type(user))
print(str(users))

