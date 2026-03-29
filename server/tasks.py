import sqlite3
import re

class Task:
    def __init__(self, task_id: int):
        self.task_id = task_id
        
    def setup_db(self, conn: sqlite3.Connection):
        raise NotImplementedError
        
    def get_goal(self) -> str:
        raise NotImplementedError
        
    def grade(self, conn: sqlite3.Connection) -> float:
        raise NotImplementedError

class EasyTask(Task):
    def __init__(self):
        super().__init__(1)

    def setup_db(self, conn: sqlite3.Connection):
        c = conn.cursor()
        c.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, total_spent REAL)")
        c.executemany("INSERT INTO customers (name, total_spent) VALUES (?, ?)", [
            ("Alice", 500.0),
            ("Bob", 1200.0),
            ("Charlie", 50.0),
            ("Diana", 3000.0),
            ("Eve", 1000.01) # over 1000
        ])
        conn.commit()

    def get_goal(self) -> str:
        return "Create a view named 'high_value_customers' containing all customers who have a 'total_spent' greater than 1000.0. The view should contain the exact same columns as the customers table."

    def grade(self, conn: sqlite3.Connection) -> float:
        c = conn.cursor()
        try:
            # Check if view exists
            c.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='high_value_customers'")
            if not c.fetchone():
                return 0.0
            
            # Check rows
            c.execute("SELECT name, total_spent FROM high_value_customers ORDER BY name")
            rows = c.fetchall()
            
            if len(rows) != 3:
                return 0.5 # partially correct, exists but wrong rows
                
            expected = [("Bob", 1200.0), ("Diana", 3000.0), ("Eve", 1000.01)]
            if rows == expected:
                return 1.0
            return 0.5
        except Exception:
            return 0.0


class MediumTask(Task):
    def __init__(self):
        super().__init__(2)

    def setup_db(self, conn: sqlite3.Connection):
        c = conn.cursor()
        c.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price TEXT)")
        c.executemany("INSERT INTO products (name, category, price) VALUES (?, ?, ?)", [
            ("Laptop", "Electronics", "$999.99"),
            ("Mouse", "electronics", "25.50 USD"),
            ("Desk", "FURNITURE", "150.0"),
            ("Chair", "furniture", "$85.00"),
            ("Headphones", "ELEC", "€45.00") # We'll just ask them to remove letters/symbols
        ])
        conn.commit()

    def get_goal(self) -> str:
        return (
            "The 'products' table is messy. "
            "1. Standardize the 'category' column to be fully UPPERCASE. (Hint: treat 'ELEC' as 'ELECTRONICS'). "
            "2. Create a new column 'price_usd' of type REAL. Extract the numeric value from the 'price' string and populate 'price_usd'. "
            "Do not drop any original columns."
        )

    def grade(self, conn: sqlite3.Connection) -> float:
        score = 0.0
        c = conn.cursor()
        try:
            # Check column exists
            c.execute("PRAGMA table_info(products)")
            columns = [row[1] for row in c.fetchall()]
            if 'price_usd' in columns:
                score += 0.3
                
                # Check data accuracy for price
                c.execute("SELECT price_usd FROM products ORDER BY id")
                prices = [row[0] for row in c.fetchall()]
                expected_prices = [999.99, 25.50, 150.0, 85.0, 45.0]
                
                # allow small float diffs
                correct_prices = sum(1 for p, e in zip(prices, expected_prices) if p is not None and abs(p - e) < 0.01)
                score += (correct_prices / 5.0) * 0.4 # up to 0.4 for correct prices

            # Check category uppercase
            c.execute("SELECT category FROM products ORDER BY id")
            categories = [row[0] for row in c.fetchall()]
            expected_cats = ["ELECTRONICS", "ELECTRONICS", "FURNITURE", "FURNITURE", "ELECTRONICS"]
            
            correct_cats = sum(1 for c, e in zip(categories, expected_cats) if c == e)
            score += (correct_cats / 5.0) * 0.3 # up to 0.3 for correct categories
            
            return min(1.0, score)
        except Exception:
            return score


class HardTask(Task):
    def __init__(self):
        super().__init__(3)

    def setup_db(self, conn: sqlite3.Connection):
        c = conn.cursor()
        c.execute("""
            CREATE TABLE hospital_records (
                patient_name TEXT, 
                patient_dob TEXT, 
                doctor_name TEXT, 
                doctor_specialty TEXT, 
                appointment_date TEXT, 
                diagnosis TEXT
            )
        """)
        records = [
            ("John Doe", "1980-01-01", "Dr. Smith", "Cardiology", "2023-10-01", "Hypertension"),
            ("Jane Roe", "1992-05-15", "Dr. Jones", "Neurology", "2023-10-02", "Migraine"),
            ("John Doe", "1980-01-01", "Dr. Smith", "Cardiology", "2023-11-01", "Follow-up"),
            ("Bob Guy", "1975-11-20", "Dr. Smith", "Cardiology", "2023-10-05", "Checkup")
        ]
        c.executemany("INSERT INTO hospital_records VALUES (?, ?, ?, ?, ?, ?)", records)
        conn.commit()

    def get_goal(self) -> str:
        return (
            "Normalize the flat 'hospital_records' table into 3 tables: "
            "'patients' (id INTEGER PRIMARY KEY, name TEXT, dob TEXT), "
            "'doctors' (id INTEGER PRIMARY KEY, name TEXT, specialty TEXT), and "
            "'appointments' (id INTEGER PRIMARY KEY, patient_id INTEGER, doctor_id INTEGER, date TEXT, diagnosis TEXT). "
            "Migrate all data from 'hospital_records' correctly without duplication. "
            "Ensure foreign keys are correctly pointing to the new IDs."
        )

    def grade(self, conn: sqlite3.Connection) -> float:
        score = 0.0
        c = conn.cursor()
        try:
            # Check tables exist
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in c.fetchall()]
            
            if 'patients' in tables: score += 0.1
            if 'doctors' in tables: score += 0.1
            if 'appointments' in tables: score += 0.2
            
            if score < 0.4:
                return score
                
            # Check data counts (3 unique patients, 2 unique doctors, 4 appointments)
            c.execute("SELECT COUNT(*) FROM patients")
            if c.fetchone()[0] == 3: score += 0.1
            
            c.execute("SELECT COUNT(*) FROM doctors")
            if c.fetchone()[0] == 2: score += 0.1
            
            c.execute("SELECT COUNT(*) FROM appointments")
            if c.fetchone()[0] == 4: score += 0.1
            
            # Check referential integrity (can we reconstruct the original view?)
            query = """
                SELECT p.name, p.dob, d.name, d.specialty, a.date, a.diagnosis
                FROM appointments a
                JOIN patients p ON a.patient_id = p.id
                JOIN doctors d ON a.doctor_id = d.id
                ORDER BY p.name, a.date
            """
            c.execute(query)
            reconstructed = c.fetchall()
            if len(reconstructed) == 4:
                score += 0.3
                
            return min(1.0, score)
        except Exception:
            return score

TASKS = {
    1: EasyTask(),
    2: MediumTask(),
    3: HardTask()
}
