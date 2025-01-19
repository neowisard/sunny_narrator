# Подключение к базе данных PostgreSQL и хранилищу S3 MinIO
db_engine = create_engine(postgresql_config)
s3_client = boto3.client(
    's3',
    endpoint_url=minio_config.endpoint,
    aws_access_key_id=minio_config.access_key,
    aws_secret_access_key=minio_config.secret_key
)
