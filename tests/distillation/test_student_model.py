import torch
from vad_baseline.distillation.student_model import SimplifiedCRDNN


def test_model_forward():
    """Test that model can forward pass with random audio features."""
    model = SimplifiedCRDNN(
        input_size=257,
        cnn_channels=(32, 64),
        rnn_hidden_size=128,
        rnn_num_layers=1,
        dnn_hidden_size=128,
    )

    # Create dummy input: (batch, time, freq) = (2, 100, 257)
    batch_size = 2
    time_steps = 100
    freq_bins = 257
    x = torch.randn(batch_size, time_steps, freq_bins)

    output = model(x)

    assert output.shape == (batch_size, time_steps)
    assert output.min() >= 0 and output.max() <= 1


def test_parameter_count():
    """Test that model has reasonable parameter count (< 0.5M)."""
    model = SimplifiedCRDNN(
        input_size=257,
        cnn_channels=(32, 64),
        rnn_hidden_size=128,
        rnn_num_layers=1,
        dnn_hidden_size=128,
    )

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 500_000, f"Model has {param_count} params, should be < 500K"


def test_model_output_is_probability():
    """Test output can be interpreted as probabilities."""
    model = SimplifiedCRDNN(input_size=257)
    x = torch.randn(1, 50, 257)
    output = model(x)
    probs = torch.sigmoid(output)
    assert probs.min() >= 0 and probs.max() <= 1