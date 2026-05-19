import sqlite3
import json

db_path = 'database.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print('Tables in database:')
for table in tables:
    print(f'  - {table[0]}')

# Check event 10
print('\nEvent 10 details:')
cursor.execute("SELECT id, event_name FROM events WHERE id = 10")
event = cursor.fetchone()
if event:
    print(f'  Event: {event[1]} (ID: {event[0]})')
    
    # Check if layouts table exists and has data for this event
    try:
        cursor.execute("SELECT zones_json FROM layouts WHERE event_id = 10")
        layout = cursor.fetchone()
        if layout:
            zones_json = layout[0]
            if zones_json:
                zones = json.loads(zones_json)
                print(f'  Zones defined: {len(zones.get("features", []))} zones')
            else:
                print('  Zones: NO (zones_json is empty)')
        else:
            print('  Zones: NO (no layout record)')
    except Exception as e:
        print(f'  Zones: ERROR ({e})')
else:
    print('  Event 10 not found')

conn.close()
