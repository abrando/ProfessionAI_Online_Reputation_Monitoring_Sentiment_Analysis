from src.predict import predict_sentiment

def test_prediction_output():
    out = predict_sentiment(["hello"])
    assert isinstance(out, list)
    assert "label" in out[0]
    assert "score" in out[0]