"""LangChain chat-model boundary for locally executed, downloaded weights."""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class LocalWeightsChatModel(BaseChatModel):
    backend: Any = Field(exclude=True, repr=False)
    model_id: str
    revision: str

    @property
    def _llm_type(self):
        return 'huggingface-downloaded-weights'

    @property
    def _identifying_params(self):
        return {'model_id': self.model_id, 'revision': self.revision, 'inference': 'self_hosted_downloaded_weights'}

    def _get_ls_params(self, stop=None, **kwargs):
        return {'ls_provider': 'huggingface-local', 'ls_model_name': self.model_id, 'ls_model_type': 'chat'}

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError('Stop sequences are not supported by this bounded research adapter')
        if len(messages) != 1 or not isinstance(messages[0].content, str):
            raise ValueError('This text-only research adapter expects one string prompt')
        text = self.backend._generate_text(messages[0].content)
        measurement = self.backend.measurements[-1]
        usage = {'input_tokens': measurement['input_tokens'], 'output_tokens': measurement['output_tokens'], 'total_tokens': measurement['input_tokens'] + measurement['output_tokens']}
        message = AIMessage(content=text, usage_metadata=usage, response_metadata={'model_id': self.model_id, 'revision': self.revision})
        return ChatResult(generations=[ChatGeneration(message=message)])
