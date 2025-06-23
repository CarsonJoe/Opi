#!/usr/bin/env python3
"""
Debug MCP Connection Issues
Comprehensive testing for MCP server connectivity
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastmcp import Client
    FASTMCP_AVAILABLE = True
    print("✅ FastMCP available")
except ImportError:
    FASTMCP_AVAILABLE = False
    print("❌ FastMCP not available - install with: pip install fastmcp>=2.0.0")

def load_config():
    """Load configuration file."""
    config_path = "config.json"
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded config from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

async def test_http_server(url):
    """Test HTTP MCP server connection."""
    print(f"\n🌐 Testing HTTP MCP server: {url}")
    
    if not FASTMCP_AVAILABLE:
        print("❌ Cannot test - FastMCP not available")
        return False
    
    try:
        print(f"   Connecting to {url}...")
        
        async with Client(url) as client:
            print("   ✅ Connected successfully!")
            
            # List tools
            print("   📋 Listing available tools...")
            tools = await client.list_tools()
            
            if not tools:
                print("   ⚠️  No tools available")
                return True
            
            print(f"   ✅ Found {len(tools)} tools:")
            for tool in tools:
                if hasattr(tool, 'name'):
                    name = tool.name
                    desc = getattr(tool, 'description', 'No description')
                else:
                    name = tool.get('name', 'Unknown')
                    desc = tool.get('description', 'No description')
                print(f"      🔧 {name}: {desc}")
            
            # Test a simple tool call
            if tools:
                test_tool_name = None
                
                # Look for a simple test tool
                for tool in tools:
                    tool_name = tool.name if hasattr(tool, 'name') else tool.get('name')
                    if tool_name in ['get_secret_message', 'count_secrets', 'echo_message']:
                        test_tool_name = tool_name
                        break
                
                if test_tool_name:
                    print(f"   🧪 Testing tool: {test_tool_name}")
                    try:
                        if test_tool_name == 'echo_message':
                            result = await client.call_tool(test_tool_name, {"message": "Hello MCP!"})
                        else:
                            result = await client.call_tool(test_tool_name, {})
                        
                        print(f"   ✅ Tool call successful!")
                        print(f"   📄 Result: {result}")
                        
                    except Exception as e:
                        print(f"   ⚠️  Tool call failed: {e}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

async def test_server_startup():
    """Test if we can start our own server."""
    print(f"\n🚀 Testing server startup...")
    
    try:
        from mcp_tools.secret_server import mcp
        print("   ✅ Secret server module imported successfully")
        
        # Try to get server info
        print(f"   📋 Server has tools and resources configured")
        return True
        
    except Exception as e:
        print(f"   ❌ Server startup test failed: {e}")
        return False

def check_config_mcp_section():
    """Check MCP configuration section."""
    print(f"\n⚙️  Checking MCP configuration...")
    
    config = load_config()
    if not config:
        return False
    
    mcp_config = config.get('mcp', {})
    if not mcp_config:
        print("   ❌ No 'mcp' section in config.json")
        return False
    
    servers = mcp_config.get('servers', {})
    if not servers:
        print("   ❌ No 'servers' in mcp config")
        return False
    
    print(f"   ✅ Found {len(servers)} configured servers:")
    
    for server_name, server_config in servers.items():
        print(f"      📡 {server_name}:")
        
        enabled = server_config.get('enabled', True)
        print(f"         Enabled: {enabled}")
        
        if 'url' in server_config:
            url = server_config['url']
            print(f"         URL: {url}")
            
            if not url.startswith('http://') and not url.startswith('https://'):
                print(f"         ⚠️  URL should start with http:// or https://")
        
        if 'command' in server_config:
            cmd = server_config['command']
            print(f"         Command: {cmd}")
    
    return True

async def comprehensive_test():
    """Run comprehensive MCP testing."""
    print("🔍 MCP Connection Debugger")
    print("=" * 50)
    
    # Check config
    config_ok = check_config_mcp_section()
    
    # Test server startup
    startup_ok = await test_server_startup()
    
    # Test HTTP connections from config
    if config_ok:
        config = load_config()
        mcp_config = config.get('mcp', {})
        servers = mcp_config.get('servers', {})
        
        http_tests_passed = 0
        total_http_tests = 0
        
        for server_name, server_config in servers.items():
            if server_config.get('enabled', True) and 'url' in server_config:
                url = server_config['url']
                if url.startswith('http'):
                    total_http_tests += 1
                    success = await test_http_server(url)
                    if success:
                        http_tests_passed += 1
        
        if total_http_tests > 0:
            print(f"\n📊 HTTP Tests: {http_tests_passed}/{total_http_tests} passed")
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"🎯 SUMMARY:")
    print(f"   Config loaded: {'✅' if config_ok else '❌'}")
    print(f"   Server module: {'✅' if startup_ok else '❌'}")
    print(f"   FastMCP available: {'✅' if FASTMCP_AVAILABLE else '❌'}")
    
    if config_ok and startup_ok and FASTMCP_AVAILABLE:
        print(f"\n✅ MCP setup looks good!")
        print(f"💡 Next steps:")
        print(f"   1. Start server: python start_server.py --http")
        print(f"   2. Test connection: python debug_mcp.py")
        print(f"   3. Run Opi: python main.py --test-mcp")
    else:
        print(f"\n❌ Issues detected. Check the errors above.")

def print_suggested_config():
    """Print a suggested config.json MCP section."""
    print(f"\n📋 Suggested config.json MCP section:")
    
    suggested_config = {
        "mcp": {
            "servers": {
                "secret_server": {
                    "enabled": True,
                    "url": "http://127.0.0.1:8000",
                    "description": "Secret message server with system tools"
                }
            }
        }
    }
    
    print(json.dumps(suggested_config, indent=2))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--suggest-config":
        print_suggested_config()
    else:
        asyncio.run(comprehensive_test())
