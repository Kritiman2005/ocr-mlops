from api.model_loader import model, processor

def test_model_loaded():
    assert model is not None
    assert processor is not None
