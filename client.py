import asyncio
import os
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack

# 安裝必要的套件:
# pip install mcp-client openai python-dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 改為從 openai 庫導入 AsyncAzureOpenAI
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# 從 .env 文件載入環境變數
load_dotenv()

class MCPClient:
    def __init__(self):
        # 初始化 session 和 client 物件
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # --- 修改開始: 初始化 Azure OpenAI Client ---
        # 從環境變數讀取 Azure OpenAI 的設定
        self.azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not self.azure_deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME 環境變數未設定")

        # 使用非同步的 Azure OpenAI Client
        self.openai_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )
        # --- 修改結束 ---

    async def load_server_config(self):
        self.server_config = {
                "mcpServers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {
                            "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
                        }
                    }
                }
            }

    async def connect_to_server(self, server_name: str = "github"):
        """連接到 MCP 伺服器"""
        if not self.server_config:
            await self.load_server_config()

        server_info = self.server_config.get("mcpServers", {}).get(server_name)
        if not server_info:
            raise ValueError(f"No configuration found for server: {server_name}")

        command = server_info.get("command")
        args = server_info.get("args", [])
        env = server_info.get("env", None)

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

        ########### 本地MCP server ###########
        # is_python = server_script_path.endswith('.py')
        # is_js = server_script_path.endswith('.js')
        # if not (is_python or is_js):
        #     raise ValueError("伺服器腳本必須是 .py 或 .js 文件")

        # command = "python" if is_python else "node"
        # server_params = StdioServerParameters(
        #     command=command,
        #     args=[server_script_path],
        #     env=None
        # )
        ########### 本地MCP server ###########

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\n已連接到伺服器，可用工具:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """使用 Azure OpenAI 和可用工具來處理查詢"""
        messages = [
            {"role": "user", "content": query}
        ]

        # 取得 MCP Server 提供的工具列表
        response = await self.session.list_tools()
        # 轉換為 OpenAI API 接受的格式
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # --- 修改開始: 使用 Azure OpenAI API 進行 Tool-Calling ---
        while True:
            # 第一次 (或後續) API 呼叫
            print("...正在呼叫 Azure OpenAI...")
            response = await self.openai_client.chat.completions.create(
                model=self.azure_deployment_name, # 注意：這裡是你在 Azure 上的「部署名稱」
                max_tokens=1500,
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # 步驟 1: 檢查模型是否要求呼叫工具
            if not tool_calls:
                # 如果沒有工具要呼叫，直接返回模型的文字回覆
                return response_message.content

            # 步驟 2: 執行工具呼叫
            # 先將 assistant 的回覆 (包含 tool_calls) 加入歷史紀錄
            messages.append(response_message)
            
            # 平行處理所有工具呼叫
            tool_results = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                # OpenAI 的參數是 JSON 字串，需要解析
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"...正在執行工具 '{function_name}'，參數: {function_args}...")
                
                # 呼叫 MCP Server 上的工具
                result = await self.session.call_tool(function_name, function_args)
                
                # 將工具的執行結果準備好，以便回傳給模型
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result.content, # 假設 result.content 是字串
                })

            # 步驟 3: 將所有工具的執行結果加入歷史紀錄
            messages.extend(tool_results)

            # 進入下一個循環，將帶有工具結果的 messages 再次發送給模型進行總結
        # --- 修改結束 ---

    async def chat_loop(self):
        """運行互動式聊天循環"""
        print("\nMCP 客戶端已啟動！")
        print("請輸入您的查詢，或輸入 'quit' 退出。")

        while True:
            try:
                query = input("\n查詢: ").strip()

                if query.lower() == 'quit':
                    break
                
                if not query:
                    continue

                response = await self.process_query(query)
                print("\n模型回覆:\n" + response)

            except Exception as e:
                print(f"\n發生錯誤: {str(e)}")

    async def cleanup(self):
        """清理資源"""
        await self.exit_stack.aclose()

async def main():

    client = MCPClient()
    try:
        # Load configuration and connect to GitHub MCP server
        await client.load_server_config()
        await client.connect_to_server("github")
        await client.chat_loop()
    finally:
        await client.cleanup()
    ########### 本地MCP server ###########
    # if len(sys.argv) < 2:
    #     print("用法: python client.py <path_to_server_script>")
    #     sys.exit(1)

    # client = MCPClient()
    # try:
    #     await client.connect_to_server(sys.argv[1])
    #     await client.chat_loop()
    # finally:
    #     await client.cleanup()
    ########### 本地MCP server ###########
if __name__ == "__main__":
    asyncio.run(main())