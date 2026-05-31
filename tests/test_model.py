import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ImportError:
    pytest = None
import torch
import torch.nn as nn
from transformers import XLMRobertaConfig, TrainingArguments, Trainer
from src.models.classifier import XLMRobertaTextCNN
from src.training.trainer import _custom_loss_trainer_class

def test_model_init_weights():
    # Verify custom Conv1d and Linear weight initialization
    config = XLMRobertaConfig(
        vocab_size=100,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=3,
        initializer_range=0.02
    )
    model = XLMRobertaTextCNN(config)
    
    # Check that conv weights are initialized with std around 0.02 and biases are zero
    for conv in model.convs:
        assert conv.weight.data.std().item() < 0.1  # with small layers, std won't be exactly 0.02 but should be reasonably close
        assert conv.bias.data.sum().item() == 0.0
        
    # Check that fc weights are initialized with std around 0.02 and biases are zero
    assert model.fc.weight.data.std().item() < 0.1
    assert model.fc.bias.data.sum().item() == 0.0

def test_sequence_length_guard():
    # Verify that sequence length guard correctly pads short sequences
    config = XLMRobertaConfig(
        vocab_size=100,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=3,
        initializer_range=0.02
    )
    model = XLMRobertaTextCNN(config)
    model.eval()

    # Create dummy short sequence input (batch_size=2, seq_len=3)
    input_ids = torch.randint(0, 100, (2, 3))
    attention_mask = torch.ones((2, 3), dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        assert logits.shape == (2, 3)
        assert not torch.isnan(logits).any()

def test_llrd_parameter_grouping():
    # Verify that LLRD parameter grouping is leak-proof, has zero overlaps, and handles bias/LayerNorm correctly
    config = XLMRobertaConfig(
        vocab_size=100,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=4,
        num_labels=3,
        initializer_range=0.02
    )
    model = XLMRobertaTextCNN(config)
    
    args = TrainingArguments(
        output_dir="tmp_output",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=False,
    )
    
    CustomTrainer = _custom_loss_trainer_class(Trainer)
    trainer = CustomTrainer(
        model=model,
        args=args,
        class_weights=[1.0, 1.0, 1.0],
        loss_name="focal",
        focal_gamma=2.0,
        layerwise_lr_decay=0.8,
    )
    
    optimizer = trainer.create_optimizer()
    param_groups = optimizer.param_groups
    
    # 1. Check no duplicate parameters in optimizer
    all_param_ids = []
    for group in param_groups:
        for p in group["params"]:
            all_param_ids.append(id(p))
            
    # Verify set size equals list size (no overlaps)
    assert len(all_param_ids) == len(set(all_param_ids))
    
    # Verify all trainable parameters are assigned (no omissions)
    trainable_param_ids = [id(p) for p in model.parameters() if p.requires_grad]
    assert set(all_param_ids) == set(trainable_param_ids)
    
    # 2. Check no_decay weight_decay values are strictly 0.0
    # Let's map parameter ids to weight decay and learning rate in optimizer
    param_id_to_decay = {}
    param_id_to_lr = {}
    for group in param_groups:
        wd = group["weight_decay"]
        lr = group["lr"]
        for p in group["params"]:
            param_id_to_decay[id(p)] = wd
            param_id_to_lr[id(p)] = lr
            
    no_decay_names = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if any(nd in name for nd in no_decay_names):
            assert param_id_to_decay[pid] == 0.0, f"{name} should have 0.0 weight decay"
        else:
            assert param_id_to_decay[pid] == 0.01, f"{name} should have 0.01 weight decay"

    # 3. Check learning rate layers decay structure
    # base_lr = 2e-5, decay = 0.8
    # classifier (convs, fc) should have 2e-5
    # encoder layer 3 should have 2e-5
    # encoder layer 2 should have 1.6e-5 (2e-5 * 0.8)
    # encoder layer 1 should have 1.28e-5 (2e-5 * 0.8^2)
    # encoder layer 0 should have 1.024e-5 (2e-5 * 0.8^3)
    # embeddings should have 2e-5 * 0.8^4 = 8.192e-6
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        lr = param_id_to_lr[pid]
        if any(kw in name for kw in ["classifier", "pooler", "convs", "fc"]):
            assert lr == pytest.approx(2e-5)
        elif ".layer.3." in name:
            assert lr == pytest.approx(2e-5)
        elif ".layer.2." in name:
            assert lr == pytest.approx(1.6e-5)
        elif ".layer.1." in name:
            assert lr == pytest.approx(1.28e-5)
        elif ".layer.0." in name:
            assert lr == pytest.approx(1.024e-5)
        elif "embedding" in name.lower():
            assert lr == pytest.approx(8.192e-6)

def test_focal_loss_clamping():
    # Verify that focal loss compute_loss is numerically stable and uses correct clamping
    config = XLMRobertaConfig(
        vocab_size=100,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=3,
        initializer_range=0.02
    )
    model = XLMRobertaTextCNN(config)
    
    args = TrainingArguments(
        output_dir="tmp_output",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=False,
    )
    
    CustomTrainer = _custom_loss_trainer_class(Trainer)
    trainer = CustomTrainer(
        model=model,
        args=args,
        class_weights=[1.0, 1.0, 1.0],
        loss_name="focal",
        focal_gamma=2.0,
    )
    
    # Create extreme inputs to test clamping stability
    inputs = {
        # High logit for class 0, extremely low for others -> pt for class 0 will be extremely close to 1.0
        "input_ids": torch.randint(0, 100, (2, 5)),
        "attention_mask": torch.ones((2, 5), dtype=torch.long),
        "labels": torch.tensor([0, 0], dtype=torch.long)
    }
    
    # Mock model outputs to have extreme values (e.g. logit of class 0 is 1000.0, others are -1000.0)
    class MockOutput:
        def __init__(self):
            self.logits = torch.tensor([[1000.0, -1000.0, -1000.0],
                                        [1000.0, -1000.0, -1000.0]], dtype=torch.float32)
            self.hidden_states = None
            self.attentions = None
            
    # Mock model to return MockOutput
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
        def forward(self, **kwargs):
            return MockOutput()
            
    mock_model = MockModel()
    
    # Compute loss and assert it is stable (not NaN or Inf)
    loss = trainer.compute_loss(mock_model, inputs)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0.0

def test_dynamic_hidden_states_and_attentions():
    # Verify that XLMRobertaTextCNN only returns hidden_states and attentions if requested
    config = XLMRobertaConfig(
        vocab_size=100,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=3,
        initializer_range=0.02
    )
    model = XLMRobertaTextCNN(config)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 5))
    attention_mask = torch.ones((2, 5), dtype=torch.long)
    
    # Case 1: output_hidden_states=False, output_attentions=False (default behavior during training/eval evaluation loop)
    with torch.no_grad():
        outputs_default = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False
        )
        assert outputs_default.hidden_states is None
        assert outputs_default.attentions is None

    # Case 2: output_hidden_states=True, output_attentions=True
    with torch.no_grad():
        outputs_all = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        assert outputs_all.hidden_states is not None
        assert len(outputs_all.hidden_states) == 3  # embedding + 2 encoder layers
        assert outputs_all.attentions is not None
        assert len(outputs_all.attentions) == 2  # 2 encoder layers

if __name__ == "__main__":
    import sys
    class pytest_approx:
        def __init__(self, expected, rel=1e-6):
            self.expected = expected
            self.rel = rel
        def __eq__(self, actual):
            return abs(actual - self.expected) <= self.expected * self.rel
    
    # Mock pytest approx inside our runner if pytest is not imported
    class ApproxMock:
        def approx(self, expected, rel=1e-6):
            return pytest_approx(expected, rel)
    pytest = ApproxMock()
    
    print("Running test_model_init_weights...")
    test_model_init_weights()
    print("test_model_init_weights passed!")
    
    print("Running test_sequence_length_guard...")
    test_sequence_length_guard()
    print("test_sequence_length_guard passed!")
    
    print("Running test_llrd_parameter_grouping...")
    test_llrd_parameter_grouping()
    print("test_llrd_parameter_grouping passed!")
    
    print("Running test_focal_loss_clamping...")
    test_focal_loss_clamping()
    print("test_focal_loss_clamping passed!")
    
    print("Running test_dynamic_hidden_states_and_attentions...")
    test_dynamic_hidden_states_and_attentions()
    print("test_dynamic_hidden_states_and_attentions passed!")
    
    print("ALL TESTS PASSED SUCCESSFULLY!")
