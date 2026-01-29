def test_modular_imports():
    from config.default import default_config
    from data.synthetic import QuarticData
    from circuits.encoding import EncodingCircuit
    from autograd.psr import PSRGradient
    from loss.mse import MSELoss
    from training.trainer import Trainer

    config = default_config
    data_source = QuarticData(n_data=config['n_data'], sigma_noise=config['sigma_noise'])
    circuit = EncodingCircuit()
    grad_method = PSRGradient()
    loss_fn = MSELoss()
    trainer = Trainer(data_source, circuit, grad_method, loss_fn, config)
    assert trainer.data_source is not None
    assert trainer.circuit is not None
    assert trainer.grad_method is not None
    assert trainer.loss_fn is not None
    print("All modular components imported and instantiated successfully.")

if __name__ == "__main__":
    test_modular_imports() 