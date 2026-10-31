import boto3

s3 = boto3.client("s3")

def upload_to_s3(local_file, bucket, key):
    s3.upload_file(
        local_file,
        bucket,
        key,
        ExtraArgs={
            "ContentType": "image/jpeg"
        }
    )

    return f"s3://{bucket}/{key}"