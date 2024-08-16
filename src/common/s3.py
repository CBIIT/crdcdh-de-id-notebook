#!/usr/bin/env python
import os
import shutil
import botocore
from botocore.exceptions import ClientError


BUCKET_OWNER_ACL = 'bucket-owner-full-control'
SINGLE_PUT_LIMIT = 4_500_000_000

class S3Bucket:
    def __init__(self, bucket, session):
        self.bucket_name = bucket
        self.client = session.client('s3')
        self.s3 = session.resource('s3')
        self.bucket = self.s3.Bucket(bucket)

    def _put_file_obj(self, key, data):
        return self.bucket.put_object(Key=key,
                                      Body=data,
                                      ACL= BUCKET_OWNER_ACL)

    def _upload_file_obj(self, key, data, config=None, extra_args={'ACL': BUCKET_OWNER_ACL}):
        return self.bucket.upload_fileobj(data, key, ExtraArgs=extra_args, Config=config)

    def get_object_size(self, key):
        try:
            res = self.client.head_object(Bucket=self.bucket_name, Key=key)
            return res['ContentLength']

        except ClientError as e:
            return None


    def download_file(self, key, filename):
        return self.bucket.download_file(key, filename)

    def download_file_obj(self, key, obj):
        self.bucket.download_fileobj(key, obj)

    def delete_file(self, key):
        response = self.bucket.delete_objects(
            Delete={
                'Objects': [
                    {
                        'Key': key
                    }
                ]
            }
        )
        if 'Errors' in response:
            self.log.error('S3: delete file {} failed!'.format(key))
            return False
        else:
            return True

    def file_exists_on_s3(self, key):
        '''
        Check if file exists in S3, return True only if file exists

        :param key: file path
        :return: boolean
        '''
        try:
            self.client.head_object(Bucket=self.bucket.name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] in ['404', '412']:
                return False
            else:
                self.log.error('Unknown S3 client error!')
                self.log.exception(e)
                return False

    def same_file_exists_on_s3(self, key, md5):
        '''
        Check if same file already exists in S3, return True only if file with same MD5 exists

        :param key: file path
        :param md5: file MD5
        :return: boolean
        '''
        try:
            self.client.head_object(Bucket=self.bucket.name, Key=key, IfMatch=md5)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] in ['404', '412']:
                return False
            else:
                self.log.error('Unknown S3 client error!')
                self.log.exception(e)

    def same_size_file_exists(self, key, file_size):
        return file_size == self.get_object_size(key)

    def upload_file(self, key, file_name):
        with open(file_name, 'rb') as data:
            file_size = os.path.getsize(file_name)
            if file_size > SINGLE_PUT_LIMIT:
                try:
                    self._upload_file_obj(key, data)
                    return {'bucket': self.bucket.name, 'key': key}
                except ClientError as e:
                    self.log.exception(e)
                    return None
            else:
                obj = self._put_file_obj(key, data)

                if obj:
                    return {'bucket': self.bucket.name, 'key': key,}
                else:
                    message = "Upload file {} to S3 failed!".format(file_name)
                    self.log.error(message)
                    return None

    def download_files_in_folder(self, folder, local_path):
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            result = self.client.list_objects_v2(Bucket=self.bucket_name,  Prefix=folder)
            for file in result.get('Contents', []):
                if file['Size'] > 0:
                    key = file['Key']
                    base_name = os.path.relpath(key, folder)
                    file_name = os.path.join(local_path, base_name)
                    dir_name = os.path.dirname(file_name)
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    self.log.info('Downloading "{}" from AWS S3'.format(base_name))
                    self.download_file(key, file_name)
            return True
        except botocore.exceptions.ClientError as e:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the bucket does not exist.
            error_code = int(e.response['Error']['Code'])
            if error_code == 403:
                self.log.error('Don\'t have permission to access for Bucket: "{}"'.format(self.bucket_name))
            elif error_code == 404:
                self.log.error('Bucket: "{}" does NOT exist!'.format(self.bucket_name))
            return False
        