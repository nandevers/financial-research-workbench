from workbench.local_chat_model import LocalWeightsChatModel


def test_local_chat_adapter_preserves_prompt_and_reports_actual_usage(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    class FixtureBackend:
        measurements = []
        def _generate_text(self, prompt):
            assert prompt == 'A fixture prompt, unchanged.'
            self.measurements.append({'input_tokens': 7, 'output_tokens': 3})
            return 'Fixture response'
    adapter = LocalWeightsChatModel(backend=FixtureBackend(), model_id='fixture/not-a-benchmark-model', revision='fixture-revision')
    response = adapter.invoke('A fixture prompt, unchanged.')
    assert response.content == 'Fixture response'
    assert response.usage_metadata == {'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10}
    assert 'backend' not in adapter.model_dump()
