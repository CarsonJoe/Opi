from typing import List
from pathlib import Path

class SqliteMemoryStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        
    async def initialize(self):
        print('[Memory] Initializing memory store...')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        print('[Memory] ✅ Memory store initialized')
    
    def get_tools(self) -> List:
        return []  # Simplified - no memory tools for now
    
    async def close(self):
        print('[Memory] ✅ Memory store closed')
