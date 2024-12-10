import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import random as rd
from datetime import datetime
from tqdm import tqdm, trange
import pickle


from src.plotting import plot_training_results, plot_predictions_new
from src.utils import log_training_loss
from src.logger import ExperimentLogger
from src.quantum import MemristorCircuit, MemristorMegaCircuit

def MSEloss(y_train, prob_state_001):
    """ Compute MSE loss objective 001 state output.
    """
    loss = tf.square(tf.abs(y_train - prob_state_001))

    return loss

def NLLloss(y_train, prob_state_001, prob_state_010):
    """ Compute NLL loss objective 001 state output as mean and 010 state output as variance.
    """
    eps = 0.0001
    loss = tf.fraction(1, tf.pow(prob_state_010,2)+eps)*tf.square(tf.abs(y_train - prob_state_001))+ tf.log(tf.pow(prob_state_010,2)+eps)

    return loss

#def KLdivloss(y_train, prob_state_001, prob_state_010, ):
#    """ Compute KL div loss objective 001 state output as mean and 010 state output as variance and fit stochastic params.
#    """
#    eps = 0.0001
#    KLdiv = rewrite cuircuit with reparametrization trick and stochastic params
#    loss = tf.fraction(1, tf.pow(prob_state_010,2)+eps)*tf.square(tf.abs(y_train - prob_state_001))+ tf.log(tf.pow(prob_state_010,2)+eps)
#
#    return loss

#def kl_div(ciruit, mu_q, sigma_q, mu_p, sigma_p):
#        
#        """Compute kl divergence between two gaussians (Q || P).
#
#        Args:
#            mu_q: mu parameter of distribution Q
#            sigma_q: sigma parameter of distribution Q
#            mu_p: mu parameter of distribution P
#            sigma_p: sigma parameter of distribution P
#
#        Returns:
#            kl divergence
#        """
#        kl = (
#            tf.log(sigma_p)
#            - tf.log(sigma_q)
#            + (tf.pow(sigma_q,2) + tf.pow((mu_q - mu_p),2)) / (2 * (tf.pow(sigma_p,2)))
#            - 0.5
#        )
#        return kl.mean()
    
def train_mve_memristor(X_train, 
                    y_train, 
                    memory_depth, 
                    training_steps, 
                    learning_rate, 
                    cutoff_dim, 
                    logger: ExperimentLogger,
                    log_filepath: str = None,
                    log_path: str = None,
                    param_id: str = None, 
                    plot = True,
                    plot_path: str = None):
    """
    Trains the MVE memristor model using the training data. This means that the circuit models the target data
    as a Gaussian distribution, by outputting the mean and variance of a 1d Gaussian distribution.
    This is an extension of 
    * https://ieeexplore.ieee.org/document/374138
    to integrated phonic quantum circuits.

    Args:
        x_train: Training input data.
        y_train: Training target data.
        memory_depth: Memory depth of the memristor.

    Returns:
        res_mem: Dictionary containing the training loss and parameters over iterations.
        phase1: Trained phase parameter 1.
        phase3: Trained phase parameter 3.
        memristor_weight: Trained weight parameter for the memristor update function.
    """

    # Initialize variables and optimizer
    np.random.seed(42)
    tf.random.set_seed(42)
    rd.seed(42)

    phase1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter

    
    logger.log_initial_training_phase(phase1, phase3, memristor_weight)
    
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    res_mem = {}

    encoded_phases = tf.constant(2 * np.arccos(X_train), dtype=tf.float64)
    num_samples = len(encoded_phases)

    # Initialize memory variables
    #memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    #memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    #let's try this with the 100 state for memory
    memory_p3 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    # Training loop
    pbar = trange(training_steps, desc='Training', unit='step')
    for step in pbar:
        # Reset the engine if it has already been executed
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss = 0.0

            for i in tqdm(range(num_samples), 
                         desc=f'Step {step+1}/{training_steps}',
                         leave=False,
                         unit='sample'):
                time_step = i - cycle_index * memory_depth
                if time_step == memory_depth - 1:
                    cycle_index += 1

                if i == 0:
                    memristor_phase = tf.acos(tf.sqrt(0.5))

                    # Init Memristor Circuit
                    memristor_circuit = MemristorCircuit(phase1, memristor_phase, phase3, encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())
                else:
                    memristor_phase = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p3) / memory_depth
                    ))
                    memristor_circuit = MemristorCircuit(phase1, memristor_phase, phase3, encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())

                # Get probabilities from the circuit results
                prob = results.state.all_fock_probs()
                prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float64)
                prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float64)
                prob_state_100 = tf.cast(tf.math.real(prob[1, 0, 0]), dtype=tf.float64)

                # Update memory variables
                #memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [[time_step % memory_depth]], [prob_state_010])
                #memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [[time_step % memory_depth]], [prob_state_001])
                memory_p3 = tf.tensor_scatter_nd_update(memory_p3, [[time_step % memory_depth]], [prob_state_100])

                # Compute the loss
                loss += NLLloss(y_train[i],prob_state_001,prob_state_010)

            # Compute gradients and update variables
            gradients = tape.gradient(loss, [phase1, phase3, memristor_weight])
            optimizer.apply_gradients(zip(gradients, [phase1, phase3, memristor_weight]))

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{float(loss):.4f}'})
                    
            logger.log_training_step(step, loss, phase1, phase3, memristor_weight)


            res_mem[('loss', 'tr', step)] = [loss.numpy(), phase1.numpy(), phase3.numpy(), memristor_weight.numpy()]


    final_metrics = {
        'final_loss': float(loss),
        'final_phase1': float(phase1),
        'final_phase3': float(phase3),
        'final_memristor_weight': float(memristor_weight),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'learning_rate': learning_rate,
        'cutoff_dim': cutoff_dim
    }
    logger.log_final_results(final_metrics)

    # Save trained model parameters
    trained_params = {
        'phase1': phase1.numpy(),
        'phase3': phase3.numpy(),
        'memristor_weight': memristor_weight.numpy(),
        'final_loss': loss.numpy(),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'res_mem': res_mem
    }
    logger.save_model_artifact(trained_params, 'trained_parameters.pkl')

    # Plot training results
    plot_training_results(res_mem, f"{logger.base_dir}/plots/training_results_+{param_id}.png")
    
        
    return res_mem, phase1, phase3, memristor_weight

def predict_mve_memristor(X_test: np.ndarray, 
                      y_test: np.ndarray, 
                      memory_depth: int, 
                      phase1: float, 
                      phase3: float, 
                      memristor_weight: float, 
                      stochastic: bool, 
                      samples: int, 
                      var: float, 
                      cutoff_dim: int, 
                      logger: ExperimentLogger,
                      param_id: str = None):
    """
    Uses the trained mve memristor model to make predictions on test data.     
    This means that the circuit models the target data as a Gaussian distribution, by outputting the mean and variance of a 1d Gaussian distribution.
    This is an extension of 
    * https://ieeexplore.ieee.org/document/374138
    to integrated phonic quantum circuits.

    Additionally the cicuit parameters can be modelled as stochastic variables, e.g., by applying AC voltage during experiments.
    This results in a circuit modelling epistemic uncertainty (by the model parameters) and aleatoric uncertainty (with the assumption that the targets are Gaussian).

    final_predictions: mean prediction of mve circuit model. Either mean over ensemble of sampled circuits if stochastic = True (or just mean) of Gaussian model.
    targets: targets from dataloader of target function.
    predictive_uncertainty: total predictive uncertainty computed according to eq. in section 2.4 of (https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) 
                            where ensemble members are the circuit for one sample of parameters. 
    all_predictions_mean: samples of all mean predictions. 
    all_predictions_sigma: sampes of all sigma predictions (root of variance of Gaussian) 
    epistemic_uncertainty: model/parameter uncertainty computed according to section 2.4 of (https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) 
    aleatoric_uncertainty: data uncertainty computed according to section 2.4 of (https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) 

    """

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    encoded_phases = tf.constant(2 * np.arccos(X_test), dtype=tf.float64)

    # Initialize lists to store predictions and targets
    all_predictions_mean = []
    all_predictions_sigma = []
    targets = []

    # Initialize memory variables
    #memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    #memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    #let's try this with the 100 state for memory
    memory_p3 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    if stochastic:
        print(f"Running {samples} samples with variance {var}...")
    else:
        print("Running deterministic prediction...")
        samples = 1

    # print("Predicting on test data...")
    sample_pbar = trange(samples, desc='Prediction Samples', unit='sample')
    for sample in sample_pbar:
        sample_predictions_mean = []
        sample_predictions_sigma = []

        if stochastic:
            phase1_sample = np.random.normal(phase1, var)
            # memristor_sample = np.random.normal(memristor_weight, var)
            phase3_sample = np.random.normal(phase3, var)
        else:
            phase1_sample = phase1
            # memristor_sample = memristor_weight
            phase3_sample = phase3


        # Inner progress bar for phases
        phase_pbar = tqdm(range(len(encoded_phases)), 
                         desc=f'Sample {sample+1}/{samples}',
                         leave=False,
                         unit='phase')

        for i in phase_pbar:
            time_step = i - cycle_index * memory_depth
            if time_step == memory_depth - 1:
                cycle_index += 1
            if i == 0:
                memristor_phase = tf.acos(tf.sqrt(0.5))
            else:
                memristor_phase = tf.acos(tf.sqrt(
                    memristor_weight * tf.reduce_sum(memory_p3) / memory_depth
                ))


            memristor_circuit = MemristorCircuit(phase1_sample, memristor_phase, phase3_sample, encoded_phases[i])
            results = eng.run(memristor_circuit.build_circuit())

            # Get probabilities from the circuit results
            prob = results.state.all_fock_probs()
            prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)
            prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)
            prob_state_100 = tf.cast(tf.math.real(prob[1, 0, 0]), dtype=tf.float32)

            # Update memory variables
            memory_p3 = tf.tensor_scatter_nd_update(memory_p3, [[time_step % memory_depth]], [prob_state_100])

            sample_predictions_mean.append(prob_state_001.numpy())
            sample_predictions_sigma.append(prob_state_010.numpy())
            if sample == 0:
                targets.append(float(y_test[i].numpy()))

            # Update inner progress bar with current probabilities
            phase_pbar.set_postfix({
                'prob_001': f'{float(prob_state_001):.4f}',
                'prob_010': f'{float(prob_state_010):.4f}'
            })
            # Compute the loss
            prob_state_001 = tf.cast(prob_state_001, dtype=tf.float64)
            loss = tf.square(tf.abs(y_test[i] - prob_state_001))
            logger.log_prediction_step(i, loss, phase1_sample, phase3_sample, memristor_weight)


        all_predictions_mean.append(sample_predictions_mean)
        all_predictions_sigma.append(sample_predictions_sigma)

    # Convert all_predictions to a NumPy array for easier manipulation
    all_predictions_mean = np.array(all_predictions_mean)
    all_predictions_sigma = np.array(all_predictions_sigma)

    if stochastic:
        # Calculate mean and standard deviation along the column axis
        final_predictions = np.mean(all_predictions_mean, axis=0)
        epistemic_uncertainty = np.std(all_predictions_mean, axis=0)
        aleatoric_uncertainty = np.sqrt(np.mean(np.pow(all_predictions_sigma,2)))
        predictive_uncertainty = np.std(np.pow(epistemic_uncertainty,2)+np.pow(aleatoric_uncertainty,2))
        logger.log_prediction(final_predictions, predictive_uncertainty, samples)
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, f"{logger.base_dir}/plots/prediction_results_sample{samples}_{param_id}.png")
    else:
        final_predictions = all_predictions_mean[0]
        aleatoric_uncertainty = all_predictions_sigma[0]
        predictive_uncertainty = np.array([])
        targets = np.array(targets)
        logger.log_prediction(final_predictions)
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, f"{logger.base_dir}/plots/prediction_results_deterministic_{param_id}.png")


    return final_predictions, targets, predictive_uncertainty, all_predictions_mean, all_predictions_sigma, epistemic_uncertainty, aleatoric_uncertainty


def train_mve_mega_memristor(X_train, 
                    y_train, 
                    memory_depth, 
                    training_steps, 
                    learning_rate, 
                    cutoff_dim, 
                    logger: ExperimentLogger,
                    log_filepath: str = None,
                    log_path: str = None,
                    param_id: str = None, 
                    plot = True,
                    plot_path: str = None):
    """
    Trains the longer mve memristor model to make predictions on training data.     
    This means that the circuit models the target data as a Gaussian distribution, by outputting the mean and variance of a 1d Gaussian distribution.
    This is an extension of 
    * https://ieeexplore.ieee.org/document/374138
    to integrated phonic quantum circuits.

    Additionally the cicuit parameters can be modelled as stochastic variables, e.g., by applying AC voltage during experiments.
    This results in a circuit modelling epistemic uncertainty (by the model parameters) and aleatoric uncertainty (with the assumption that the targets are Gaussian).
    """

    # Initialize variables and optimizer
    np.random.seed(42)
    tf.random.set_seed(42)
    rd.seed(42)

    phase1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter
    phase4 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase6 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    memristor2_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter

    
    logger.log_initial_training_phase(phase1, phase3, memristor_weight, phase4, phase6, memristor2_weight)
    
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    res_mem = {}

    encoded_phases = tf.constant(2 * np.arccos(X_train), dtype=tf.float64)
    num_samples = len(encoded_phases)

    # Initialize memory variables
    memory_p3 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    # Training loop
    pbar = trange(training_steps, desc='Training', unit='step')
    for step in pbar:
        # Reset the engine if it has already been executed
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss = 0.0

            for i in tqdm(range(num_samples), 
                         desc=f'Step {step+1}/{training_steps}',
                         leave=False,
                         unit='sample'):
                time_step = i - cycle_index * memory_depth
                if time_step == memory_depth - 1:
                    cycle_index += 1

                if i == 0:
                    memristor_phase = tf.acos(tf.sqrt(0.5))

                    # Init Memristor Circuit
                    memristor_circuit = MemristorMegaCircuit(phase1, memristor_phase, phase3, phase4, memristor2_weight, phase6, encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())
                else:
                    #we can choose which output we feed back here?
                    memristor_phase = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p3) / memory_depth
                    ))
                    memristor_circuit = MemristorMegaCircuit(phase1, memristor_phase, phase3, phase4, memristor2_weight, phase6, encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())

                # Get probabilities from the circuit results
                prob = results.state.all_fock_probs()
                prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float64)
                prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float64)
                prob_state_100 = tf.cast(tf.math.real(prob[1, 0, 0]), dtype=tf.float64)

                # Update memory variables
                memory_p3 = tf.tensor_scatter_nd_update(memory_p3, [[time_step % memory_depth]], [prob_state_100])

                # Compute the loss
                loss += NLLloss(y_train[i],prob_state_001,prob_state_010)

            # Compute gradients and update variables
            gradients = tape.gradient(loss, [phase1, phase3, memristor_weight, phase4, memristor2_weight, phase6])
            optimizer.apply_gradients(zip(gradients, [phase1, phase3, memristor_weight, phase4, memristor2_weight, phase6]))

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{float(loss):.4f}'})
                    
            logger.log_training_step(step, loss, phase1, phase3, memristor_weight, phase4, memristor2_weight, phase6)


            res_mem[('loss', 'tr', step)] = [loss.numpy(), phase1.numpy(), phase3.numpy(), memristor_weight.numpy(), phase4.numpy(), memristor2_weight.numpy(), phase6.numpy()]


    final_metrics = {
        'final_loss': float(loss),
        'final_phase1': float(phase1),
        'final_phase3': float(phase3),
        'final_memristor_weight': float(memristor_weight),
        'final_phase4': float(phase4),
        'final_phase6': float(phase6),
        'final_memristor2_weight': float(memristor2_weight),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'learning_rate': learning_rate,
        'cutoff_dim': cutoff_dim
    }
    logger.log_final_results(final_metrics)

    # Save trained model parameters
    trained_params = {
        'phase1': phase1.numpy(),
        'phase3': phase3.numpy(),
        'memristor_weight': memristor_weight.numpy(),
        'phase4': phase4.numpy(),
        'phase6': phase6.numpy(),
        'memristor2_weight': memristor2_weight.numpy(),
        'final_loss': loss.numpy(),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'res_mem': res_mem
    }
    logger.save_model_artifact(trained_params, 'trained_parameters.pkl')

    # Plot training results
    plot_training_results(res_mem, f"{logger.base_dir}/plots/training_results_+{param_id}.png")
    
        
    return res_mem, phase1, phase3, memristor_weight, phase4, phase6, memristor2_weight

def predict_mve_mega_memristor(X_test: np.ndarray, 
                      y_test: np.ndarray, 
                      memory_depth: int, 
                      phase1: float, 
                      phase3: float, 
                      memristor_weight: float, 
                      phase4: float, 
                      phase6: float, 
                      memristor2_weight: float,
                      stochastic: bool, 
                      samples: int, 
                      var: float, 
                      cutoff_dim: int, 
                      logger: ExperimentLogger,
                      param_id: str = None):
    """
    Uses the trained mve longer memristor model to make predictions on test data.     
    This means that the circuit models the target data as a Gaussian distribution, by outputting the mean and variance of a 1d Gaussian distribution.
    This is an extension of 
    * https://ieeexplore.ieee.org/document/374138
    to integrated phonic quantum circuits.

    Additionally the cicuit parameters can be modelled as stochastic variables, e.g., by applying AC voltage during experiments.
    This results in a circuit modelling epistemic uncertainty (by the model parameters) and aleatoric uncertainty (with the assumption that the targets are Gaussian).

    final_predictions: mean prediction of mve circuit model. Either mean over ensemble of sampled circuits if stochastic = True (or just mean) of Gaussian model.
    targets: targets from dataloader of target function.
    predictive_uncertainty: total predictive uncertainty computed according to eq. in section 2.4 of (https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) 
                            where ensemble members are the circuit for one sample of parameters. 
    all_predictions_mean: samples of all mean predictions. 
    all_predictions_sigma: sampes of all sigma predictions (root of variance of Gaussian) 
    epistemic_uncertainty: model/parameter uncertainty computed according to section 2.4 of (https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) 
    aleatoric_uncertainty: data uncertainty computed according to section 2.4 of (https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) 

    """

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    encoded_phases = tf.constant(2 * np.arccos(X_test), dtype=tf.float64)

    # Initialize lists to store predictions and targets
    all_predictions_mean = []
    all_predictions_sigma = []
    targets = []

    # Initialize memory variables
    memory_p3 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    if stochastic:
        print(f"Running {samples} samples with variance {var}...")
    else:
        print("Running deterministic prediction...")
        samples = 1

    # print("Predicting on test data...")
    sample_pbar = trange(samples, desc='Prediction Samples', unit='sample')
    for sample in sample_pbar:
        sample_predictions_mean = []
        sample_predictions_sigma = []

        if stochastic:
            phase4_sample = np.random.normal(phase4, var)
            # memristor_sample = np.random.normal(memristor_weight, var)
            phase6_sample = np.random.normal(phase6, var)
        else:
            phase4_sample = phase4
            # memristor_sample = memristor_weight
            phase6_sample = phase6


        # Inner progress bar for phases
        phase_pbar = tqdm(range(len(encoded_phases)), 
                         desc=f'Sample {sample+1}/{samples}',
                         leave=False,
                         unit='phase')

        for i in phase_pbar:
            time_step = i - cycle_index * memory_depth
            if time_step == memory_depth - 1:
                cycle_index += 1
            if i == 0:
                memristor_phase = tf.acos(tf.sqrt(0.5))
            else:
                # let us use the 100 output for the memristor?
                memristor_phase = tf.acos(tf.sqrt(
                    memristor_weight * tf.reduce_sum(memory_p3) / memory_depth
                ))


            memristor_circuit = MemristorMegaCircuit(phase1, memristor_phase, phase3, phase4_sample, memristor_phase, phase6_sample, encoded_phases[i])
            results = eng.run(memristor_circuit.build_circuit())

            # Get probabilities from the circuit results
            prob = results.state.all_fock_probs()
            prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)
            prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)
            prob_state_100 = tf.cast(tf.math.real(prob[1, 0, 0]), dtype=tf.float32)

            # Update memory variables
            memory_p3 = tf.tensor_scatter_nd_update(memory_p3, [[time_step % memory_depth]], [prob_state_100])

            sample_predictions_mean.append(prob_state_001.numpy())
            sample_predictions_sigma.append(prob_state_010.numpy())
            if sample == 0:
                targets.append(float(y_test[i].numpy()))

            # Update inner progress bar with current probabilities
            phase_pbar.set_postfix({
                'prob_001': f'{float(prob_state_001):.4f}',
                'prob_010': f'{float(prob_state_010):.4f}'
            })
            # Compute the loss
            prob_state_001 = tf.cast(prob_state_001, dtype=tf.float64)
            loss = tf.square(tf.abs(y_test[i] - prob_state_001))
            logger.log_prediction_step(i, loss, phase1, phase3, memristor_weight, phase4_sample, phase6_sample, memristor2_weight)

        all_predictions_mean.append(sample_predictions_mean)
        all_predictions_sigma.append(sample_predictions_sigma)

    # Convert all_predictions to a NumPy array for easier manipulation
    all_predictions_mean = np.array(all_predictions_mean)
    all_predictions_sigma = np.array(all_predictions_sigma)

    if stochastic:
        # Calculate mean and standard deviation along the column axis
        final_predictions = np.mean(all_predictions_mean, axis=0)
        epistemic_uncertainty = np.std(all_predictions_mean, axis=0)
        aleatoric_uncertainty = np.sqrt(np.mean(np.pow(all_predictions_sigma,2)))
        predictive_uncertainty = np.std(np.pow(epistemic_uncertainty,2)+np.pow(aleatoric_uncertainty,2))
        logger.log_prediction(final_predictions, predictive_uncertainty, samples)
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, f"{logger.base_dir}/plots/prediction_results_sample{samples}_{param_id}.png")
    else:
        final_predictions = all_predictions_mean[0]
        aleatoric_uncertainty = all_predictions_sigma[0]
        predictive_uncertainty = np.array([])
        targets = np.array(targets)
        logger.log_prediction(final_predictions)
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, f"{logger.base_dir}/plots/prediction_results_deterministic_{param_id}.png")


    return final_predictions, targets, predictive_uncertainty, all_predictions_mean, all_predictions_sigma, epistemic_uncertainty, aleatoric_uncertainty
