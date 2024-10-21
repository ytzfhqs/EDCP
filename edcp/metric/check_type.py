from typing import List, Dict, Any, Union, Optional


from loguru import logger
from pydantic import BaseModel, ValidationError


class FileOrData(BaseModel):
    data_or_filepath: Union[str, List[Dict[str, Any]]]


class FilePath(BaseModel):
    file_path: str


class LDictData(BaseModel):
    data: List[Dict[str, Any]]


def check_path_data(path_or_data) -> str:
    try:
        FilePath(file_path=path_or_data)
        if path_or_data.endswith(".json"):
            logger.info('The input is the file path and the file is being read.')
        else:
            raise Exception("Unable to read non-json files!")
        return "path"
    except ValidationError as e:
        try:
            LDictData(data=path_or_data)
            logger.info('The input is a List[Dict[str, Any]] variable, passing the variable.')
            return "data"
        except ValidationError as e:
            raise
