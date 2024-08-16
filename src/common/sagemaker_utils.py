import os
import boto3
from sagemaker.session import Session
from sagemaker.config import load_sagemaker_config, validate_sagemaker_config
from sagemaker.local import LocalSession
from sagemaker import get_execution_role
from botocore.exceptions import ClientError
import numpy as np
import common.sagemaker_config as config

def get_sagemaker_session(boto3_session, run_Locally = False):
    """
    Get sagemaker session
    @param is_Local: is local
    @return: sagemaker session
    """
    if run_Locally == True: 
        # Create a configuration dictionary manually
        custom_sagemaker_config = load_sagemaker_config(
            additional_config_paths=[
                os.path.join(config.ROOT_DIR, "configs/sagemaker/config.yaml")
            ]
        )
        # Then validate that the dictionary adheres to the configuration schema
        validate_sagemaker_config(custom_sagemaker_config)
        # Then initialize the Session object with the configuration dictionary
        return Session(
            boto_session = boto3_session,
            sagemaker_config = custom_sagemaker_config
        ) 
    else:
        return Session()
    
def get_local_session():
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True, 'region_name': "us-east-1" }}
    return sagemaker_session

def delete_endpoint(endpoint_name, boto3_session):
    """
    Delete endpoint and endpoint configuration
    @param endpoint_name: endpoint name
    """
    sagemaker_client = None
    try:
        sagemaker_client = boto3_session.client("sagemaker")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    except ClientError as ce:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(ce)
    finally:
        if sagemaker_client:
            sagemaker_client.close()
            sagemaker_client = None

def get_sagemaker_execute_role(sagemaker_session):
    """
    Get sagemaker execute role
    @param sagemaker_session: sagemaker session
    @return: sagemaker execute role
    """
    if config.RUN_LOCAL == True:
        return config.SAGEMAKER_EXECUTE_ROLE
    else:
        return get_execution_role(sagemaker_session)
        


def create_local_output_dirs(output_dirs):
    """
    Create local output dirs
    @param output_dirs: output dir list
    """
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)