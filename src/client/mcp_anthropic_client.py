import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("anthropic_mcp_client")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
            self, name: str, description: str, input_schema: Dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema
        self.server_name: str = ""  # Track which server this tool belongs to

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""Tool: {self.name} (Server: {self.server_name})
                Description: {self.description}
                Arguments:
                {chr(10).join(args_desc)}
                """


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        import shutil

        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            logger.info(f"Server {self.name} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Tool]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    t = Tool(tool.name, tool.description, tool.inputSchema)
                    t.server_name = self.name
                    tools.append(t)

        return tools

    async def execute_tool(
            self,
            tool_name: str,
            arguments: Dict[str, Any],
            retries: int = 2,
            delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        last_error = None

        while attempt <= retries:
            try:
                logger.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result

            except Exception as e:
                last_error = e
                attempt += 1
                logger.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries + 1}."
                )
                if attempt <= retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        logger.error("Max retries reached. Failing.")
        raise last_error

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                logger.info(f"Server {self.name} cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup of server {self.name}: {e}")


class AnthropicClient:
    """Client for the Anthropic API."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key
        self.client = anthropic.Anthropic(api_key=api_key)

    def get_response(self, messages: List[Dict[str, str]], system: str) -> str:
        """Get a response from the Anthropic model.

        Args:
            messages: A list of message dictionaries.
            system: System prompt for Anthropic.

        Returns:
            The model's response as a string.
        """
        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-latest",
                messages=messages,
                system=system,
                max_tokens=4096,
                temperature=0.7
            )
            return response.content[0].text

        except Exception as e:
            error_message = f"Error getting Anthropic response: {str(e)}"
            logger.error(error_message)
            return f"I encountered an error: {error_message}. Please try again."


class ChatSession:
    """Orchestrates the interaction between user, Anthropic, and tools."""

    def __init__(self, servers: List[Server], anthropic_client: AnthropicClient) -> None:
        self.servers: List[Server] = servers
        self.anthropic_client: AnthropicClient = anthropic_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        # Direct cleanup without creating separate tasks
        for server in self.servers:
            try:
                await server.cleanup()
            except Exception as e:
                logger.warning(f"Warning during cleanup of server {server.name}: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logger.info(f"Executing tool: {tool_call['tool']}")
                logger.info(f"With arguments: {tool_call['arguments']}")

                # Find which server has this tool
                server_name = tool_call.get("server")

                # If server is specified, try that one first
                if server_name:
                    for server in self.servers:
                        if server.name == server_name:
                            try:
                                result = await server.execute_tool(
                                    tool_call["tool"], tool_call["arguments"]
                                )
                                return f"Tool execution result: {result}"
                            except Exception as e:
                                error_msg = f"Error executing tool on specified server: {str(e)}"
                                logger.error(error_msg)
                                return error_msg

                # Try all servers if server not specified or not found
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )
                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logger.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize server {server.name}: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_prompt = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "server": "server-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

            messages = []

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ["quit", "exit"]:
                        logger.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})

                    llm_response = self.anthropic_client.get_response(messages, system=system_prompt)
                    logger.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)

                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        # Use a different system prompt for tool result interpretation
                        tool_system_prompt = "You are a helpful assistant explaining tool results to the user."
                        messages.append({"role": "user",
                                         "content": f"Tool result: {result}\nPlease explain this result in a helpful way."})

                        final_response = self.anthropic_client.get_response(messages, system=tool_system_prompt)
                        logger.info("\nFinal response: %s", final_response)
                        messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logger.info("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    # Load configuration
    with open("servers_config.json", "r") as f:
        server_config = json.load(f)

    # Get API key from environment
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    # Initialize servers
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]

    # Initialize Anthropic client
    anthropic_client = AnthropicClient(anthropic_api_key)

    # Create and start chat session
    chat_session = ChatSession(servers, anthropic_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
