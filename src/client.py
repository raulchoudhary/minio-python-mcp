from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os
import sys
import tempfile


async def main():
    # Get the absolute path to the server script
    server_script = os.path.join(os.path.dirname(__file__), "minio_mcp_server/server.py")

    # Create server parameters for stdio connection using python interpreter
    server_params = StdioServerParameters(
        command=sys.executable,  # Use Python interpreter
        args=[server_script],  # Path to your server script
        env=None  # Optional environment variables
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Example: Create a temporary file and upload it
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write("Hello MinIO MCP!")
                temp_file_path = temp_file.name

            try:
                # Upload the file
                print("\nUploading test file...")
                result = await session.call_tool(
                    "PutObject",
                    {
                        "bucket_name": "test-bucket",
                        "object_name": "hello.txt",
                        "file_path": temp_file_path
                    }
                )
                print(f"Upload result: {result}")
                # List available resources
                print("\nListing resources...")
                resources_result = await session.list_resources()
                for resource in resources_result.resources:
                    print(f"Resource: {resource}")

            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)


if __name__ == "__main__":
    asyncio.run(main())
