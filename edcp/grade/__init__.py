from .piplines import *
from loguru import logger
from ..tool import _is_package_available


def is_openai_available():
    return _is_package_available("openai")


def is_zhipuai_available():
    return _is_package_available("zhipuai")


def is_transformers_available():
    return _is_package_available("transformers")


if is_openai_available():
    from .chatmodel import ChatQwen, ChatGPT4

    logger.info("The ChatGPT API or Qwen API can be called.")

if is_zhipuai_available():
    from .chatmodel import ChatGLM4

    logger.info("The ChatGLM4 API can be called.")

if is_transformers_available():
    logger.info("The open source model Qwen2.5 can be called.")

if (
    (is_openai_available() is None)
    and (is_zhipuai_available() is None)
    and (is_transformers_available() is None)
):
    logger.info(
        "If you want to call Qwen or ChatGPT, please install openai package first, 'pip install openai'."
    )
    logger.info(
        "If you want to call ChatGLM, please install zhipuai package first, 'pip install zhipuai'."
    )
    logger.info(
        "If you want to call open source model Qwen2.5, please install transformers package first, 'pip install transformers'."
    )
    raise RuntimeError("Package Missing!")
