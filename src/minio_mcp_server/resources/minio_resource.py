from minio import Minio
from typing import List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger("mcp_minio_server")


class MinioResource:
    def __init__(self, max_buckets: int = 5):
        self.max_buckets = max_buckets
        self.client = Minio(
            endpoint=os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
            secure=os.getenv('MINIO_SECURE', 'false').lower() == 'true'
        )
        self.configured_buckets = []

    async def list_buckets(self, start_after: Optional[str] = None) -> List[Dict[str, Any]]:
        """List buckets with pagination support"""
        try:
            logger.debug(f"Request to list buckets with start_after: {start_after}")
            buckets = self.client.list_buckets()
            if start_after:
                buckets = [b for b in buckets if b.name > start_after]

            # Limit the number of buckets returned
            limited_buckets = buckets[:self.max_buckets]
            self.configured_buckets = [b.name for b in limited_buckets]

            return [{"Name": b.name, "CreationDate": b.creation_date} for b in limited_buckets]
        except Exception as e:
            logger.error(f"Error listing buckets: {str(e)}")
            raise

    async def list_objects(self, bucket_name: str, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List objects in a bucket with pagination support"""
        try:
            logger.error(f"Request to list objects in bucket {bucket_name}")
            objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=True)
            result = []
            count = 0

            for obj in objects:
                if count >= max_keys:
                    break
                result.append({
                    "Key": obj.object_name,
                    "LastModified": obj.last_modified,
                    "ETag": obj.etag,
                    "Size": obj.size,
                    "StorageClass": "STANDARD"
                })
                count += 1

            return result
        except Exception as e:
            logger.error(f"Error listing objects in bucket {bucket_name}: {str(e)}")
            raise

    async def get_object(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """Get object data and metadata"""
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()

            # Get object stats for metadata
            stats = self.client.stat_object(bucket_name, object_name)

            return {
                "Body": data,
                "ContentType": stats.content_type,
                "ContentLength": stats.size,
                "LastModified": stats.last_modified,
                "ETag": stats.etag,
                "Metadata": stats.metadata
            }
        except Exception as e:
            logger.error(f"Error getting object {object_name} from bucket {bucket_name}: {str(e)}")
            raise

    async def put_object(self, bucket_name: str, object_name: str, file_path: str) -> Dict[str, Any]:
        """Put object into MinIO bucket using fput method"""
        try:
            # Check if bucket exists, create if it doesn't
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")

            # Get content type based on file extension
            content_type = _get_content_type(file_path)

            # Upload the file
            result = self.client.fput_object(
                bucket_name,
                object_name,
                file_path,
                content_type=content_type
            )

            return {
                "ETag": result.etag,
                "VersionId": result.version_id,
                "Location": f"minio://{bucket_name}/{object_name}"
            }
        except Exception as e:
            logger.error(f"Error putting object {object_name} to bucket {bucket_name}: {str(e)}")
            raise


def _get_content_type(file_path: str) -> str:
    """Determine content type based on file extension"""
    import mimetypes
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type or 'application/octet-stream'


def is_text_file(filename: str) -> bool:
    """Check if the file is likely to be a text file based on extension"""
    text_extensions = {
        '.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.csv',
        '.log', '.conf', '.ini', '.py', '.js', '.html', '.css'
    }
    return any(filename.lower().endswith(ext) for ext in text_extensions)
