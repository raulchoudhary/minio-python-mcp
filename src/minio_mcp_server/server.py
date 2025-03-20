import asyncio
import base64
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from dotenv import load_dotenv
import logging
import os
from typing import List, Optional
from mcp.types import Resource, LoggingLevel, EmptyResult, Tool, TextContent, ImageContent, EmbeddedResource, \
    BlobResourceContents, ReadResourceResult
from resources.minio_resource import MinioResource, is_text_file
from pydantic import AnyUrl

# Initialize server
server = Server("minio_service")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp_minio_server")

# Get max buckets from environment or use default
max_buckets = int(os.getenv('MINIO_MAX_BUCKETS', '5'))

# Initialize MinIO resource
minio_resource = MinioResource(max_buckets=max_buckets)


@server.set_logging_level()
async def set_logging_level(level: LoggingLevel) -> EmptyResult:
    logger.setLevel(level.lower())
    await server.request_context.session.send_log_message(
        level="info",
        data=f"Log level set to {level}",
        logger="mcp_minio_server"
    )
    return EmptyResult()


@server.list_resources()
async def list_resources(start_after: Optional[str] = None) -> List[Resource]:
    """
    List MinIO buckets and their contents as resources with pagination
    Args:
        start_after: Start listing after this bucket name
    """
    resources = []
    logger.debug("Starting to list resources")
    logger.debug(f"Configured buckets: {minio_resource.configured_buckets}")

    try:
        # Get limited number of buckets
        buckets = await minio_resource.list_buckets(start_after)

        # limit concurrent operations
        async def process_bucket(bucket):
            bucket_name = bucket['Name']
            logger.debug(f"Processing bucket: {bucket_name}")
            try:
                # List objects in the bucket with a reasonable limit
                objects = await minio_resource.list_objects(bucket_name, max_keys=1000)
                for obj in objects:
                    if 'Key' in obj and not obj['Key'].endswith('/'):
                        object_key = obj['Key']
                        mime_type = "text/plain" if is_text_file(
                            object_key) else "application/octet-stream"
                        resource = Resource(
                            uri=f"minio://{bucket_name}/{object_key}",
                            name=object_key,
                            mimeType=mime_type
                        )
                        resources.append(resource)
                        logger.debug(f"Added resource: {resource.uri}")
            except Exception as ex:
                logger.error(f"Error listing objects in bucket {bucket_name}: {str(ex)}")

        # Use semaphore to limit concurrent bucket processing
        semaphore = asyncio.Semaphore(3)  # Limit concurrent bucket processing

        async def process_bucket_with_semaphore(bucket):
            async with semaphore:
                await process_bucket(bucket)

        # Process buckets concurrently
        await asyncio.gather(*[process_bucket_with_semaphore(bucket) for bucket in buckets])
    except Exception as e:
        logger.error(f"Error listing buckets: {str(e)}")
        raise

    return resources


@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """
    Read content from a MinIO resource and return structured response
    """
    uri_str = str(uri)
    logger.debug(f"Reading resource: {uri_str}")

    if not uri_str.startswith("minio://"):
        raise ValueError("Invalid MinIO URI")

    # Parse the MinIO URI
    from urllib.parse import unquote
    path = uri_str[8:]  # Remove "minio://"
    path = unquote(path)  # Decode URL-encoded characters
    parts = path.split("/", 1)

    if len(parts) < 2:
        raise ValueError("Invalid MinIO URI format")

    bucket_name = parts[0]
    key = parts[1]
    logger.debug(f"Attempting to read - Bucket: {bucket_name}, Key: {key}")

    try:
        response = await minio_resource.get_object(bucket_name, key)
        content_type = response.get("ContentType", "application/octet-stream")
        logger.debug(f"Read MIMETYPE response: {content_type}")

        if 'Body' in response:
            data = response['Body']
            # Process the data based on file type
            if is_text_file(key):
                text_content = base64.b64encode(data).decode('utf-8')
                return text_content
            else:
                text_content = str(base64.b64encode(data))
                result = ReadResourceResult(
                    contents=[
                        BlobResourceContents(
                            blob=text_content,
                            uri=uri_str,
                            mimeType=content_type
                        )
                    ]
                )
                logger.debug(result)
                return text_content
        else:
            raise ValueError("No data in response body")
    except Exception as e:
        logger.error(f"Error reading object {key} from bucket {bucket_name}: {str(e)}")
        raise ValueError(f"Error reading resource: {str(e)}")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="ListBuckets",
            description="Returns a list of all buckets owned by the authenticated sender of the request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_after": {"type": "string", "description": "Start listing after this bucket name"},
                    "max_buckets": {"type": "integer", "description": "Maximum number of buckets to return"}
                },
                "required": [],
            },
        ),
        Tool(
            name="ListObjects",
            description="Returns some or all (up to 1,000) of the objects in a bucket with each request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_name": {"type": "string", "description": "Name of the bucket"},
                    "prefix": {"type": "string",
                               "description": "Limits the response to keys that begin with the specified prefix"},
                    "max_keys": {"type": "integer", "description": "Maximum number of keys to return"}
                },
                "required": ["bucket_name"],
            },
        ),
        Tool(
            name="GetObject",
            description="Retrieves an object from MinIO.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_name": {"type": "string", "description": "Name of the bucket"},
                    "object_name": {"type": "string", "description": "Name of the object to retrieve"}
                },
                "required": ["bucket_name", "object_name"]
            }
        ),
        Tool(
            name="PutObject",
            description="Uploads a file to MinIO bucket using fput method.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_name": {"type": "string", "description": "Name of the bucket"},
                    "object_name": {"type": "string", "description": "Name to give the object in MinIO"},
                    "file_path": {"type": "string", "description": "Local file path to upload"}
                },
                "required": ["bucket_name", "object_name", "file_path"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
        name: str, arguments: dict | None
) -> list[TextContent | ImageContent | EmbeddedResource]:
    try:
        match name:
            case "ListBuckets":
                buckets = await minio_resource.list_buckets(
                    start_after=arguments.get("start_after") if arguments else None
                )
                return [
                    TextContent(
                        type="text",
                        text=str(buckets)
                    )
                ]
            case "ListObjects":
                if not arguments or "bucket_name" not in arguments:
                    raise ValueError("bucket_name is required")

                objects = await minio_resource.list_objects(
                    bucket_name=arguments["bucket_name"],
                    prefix=arguments.get("prefix", ""),
                    max_keys=arguments.get("max_keys", 1000)
                )
                return [
                    TextContent(
                        type="text",
                        text=str(objects)
                    )
                ]
            case "GetObject":
                if not arguments or "bucket_name" not in arguments or "object_name" not in arguments:
                    raise ValueError("bucket_name and object_name are required")

                response = await minio_resource.get_object(
                    bucket_name=arguments["bucket_name"],
                    object_name=arguments["object_name"]
                )
                return [
                    TextContent(
                        type="text",
                        text=str(response)
                    )
                ]
            case "PutObject":
                if (not arguments or
                        "bucket_name" not in arguments or
                        "object_name" not in arguments or
                        "file_path" not in arguments):
                    raise ValueError("bucket_name, object_name, and file_path are required")

                response = await minio_resource.put_object(
                    bucket_name=arguments["bucket_name"],
                    object_name=arguments["object_name"],
                    file_path=arguments["file_path"]
                )
                return [
                    TextContent(
                        type="text",
                        text=str(response)
                    )
                ]
    except Exception as error:
        return [
            TextContent(
                type="text",
                text=f"Error: {str(error)}"
            )
        ]


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="minio-mcp-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
