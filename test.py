#!/usr/bin/env python3
"""
Simple test of MCP manager to debug the timeout issue
"""
import asyncio
import json
from pathlib import Path

# Import your MCP manager
from llm.mcp_manager import MCPManager

async def test_mcp_manager():
    """Test the MCP manager directly."""
    print("🧪 Testing MCP Manager...")
    
    # Load config
    config_path = Path("config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    mcp_config = config.get("mcp", {}).get("servers", {})
    
    # Create MCP manager
    manager = MCPManager(mcp_config)
    
    try:
        # Connect
        print("🔌 Connecting...")
        await manager.connect_all(verbose=True)
        
        # List tools
        tools = manager.get_tools()
        print(f"🛠️  Found {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")
        
        if tools:
            # Test tool call
            tool_name = tools[0].name
            print(f"\n🧪 Testing tool: {tool_name}")
            
            start_time = asyncio.get_event_loop().time()
            result = await manager.call_tool(tool_name, {})
            end_time = asyncio.get_event_loop().time()
            
            print(f"⏱️  Tool call took {end_time - start_time:.3f} seconds")
            print(f"✅ Result: {result.get_text_content()}")
            
            # Test second call to verify connection reuse
            print(f"\n🧪 Testing second call...")
            start_time = asyncio.get_event_loop().time()
            result2 = await manager.call_tool(tool_name, {})
            end_time = asyncio.get_event_loop().time()
            
            print(f"⏱️  Second call took {end_time - start_time:.3f} seconds")
            print(f"✅ Result: {result2.get_text_content()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(test_mcp_manager())
