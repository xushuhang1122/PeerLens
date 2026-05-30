import sys; sys.path.insert(0,'.')
from src.peerlens.agent.tools_remote import _call_mcp_tool
import time
t = time.time()
result = _call_mcp_tool('http://43.134.60.58:8765/mcp', 'search_papers', {'query': 'machine learning', 'top_k': 20, 'decision_filter': ['oral','spotlight','poster','accepted']})
elapsed = round(time.time()-t, 1)
n = len(result.get('results', []))
print('took', elapsed, 's, results=', n)