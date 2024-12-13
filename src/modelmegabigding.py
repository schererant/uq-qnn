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
from src.quantum import MemristorMegaBigCircuit

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
    
def train_megabigmemristor(X_train, 
                    y_train, 
                    memory_depth, 
                    training_steps, 
                    learning_rate, 
                    cutoff_dmegabigim, 
                    cutoff_dim,
                    logger: ExperimentLogger,
                    log_filepath: str = None,
                    log_path: str = None,
                    param_id: str = None, 
                    plot = True,
                    plot_path: str = None):
    """
    Trains the mega big memristor model using the training data. 
    Here the memristors are phase3 with input 000 001, phase5 000 010, phase7 with input 010 000, phase6 with input 100 000

    Args:
        x_train: Training input data.
        y_train: Training target data.
        memory_depth: Memory depth of the memristor.

    Returns:
        res_mem: Dictionary containing the training loss and parameters over iterations.
        phasei: Trained phase parameter i in (1,...,12)
        memristor_weight: Trained weight parameter for the memristor update function.
    """

    # Initialize variables and optimizer
    np.random.seed(42)
    tf.random.set_seed(42)
    rd.seed(42)

    phase1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase2 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    #phase3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
    #                   constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase4 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    #phase5 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
    #                   constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    #phase6 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
    #                   constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    #phase7 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
    #                   constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase8 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase9 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase10 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase11 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase12 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter

    
    logger.log_initial_training_phase(phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12)
    
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    res_mem = {}

    encoded_phases = tf.constant(2 * np.arccos(X_train), dtype=tf.float64)
    num_samples = len(encoded_phases)

    # Initialize memory variables
    memory_p3 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p5 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p6 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p7 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
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
                    phase3 = tf.acos(tf.sqrt(0.5))
                    phase5 = tf.acos(tf.sqrt(0.5))
                    phase6 = tf.acos(tf.sqrt(0.5))
                    phase7 = tf.acos(tf.sqrt(0.5))
                    
                    # Init Memristor Circuit
                    memristor_circuit = MemristorMegaBigCircuit(phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())
                else:
                    phase3 = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p3) / memory_depth
                    ))
                    phase5 = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p5) / memory_depth
                    ))
                    phase6 = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p6) / memory_depth
                    ))
                    phase7 = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p7) / memory_depth
                    ))

                    memristor_circuit = MemristorMegaBigCircuit(phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())

                # Get probabilities from the circuit results
                prob = results.state.all_fock_probs()
                prob_state_000100 = tf.cast(tf.math.real(prob[0, 0, 0, 1, 0, 0]), dtype=tf.float64)
                # prob_state_001000 can be used for training with NLL
                prob_state_001000 = tf.cast(tf.math.real(prob[0, 0, 1, 0, 0, 0]), dtype=tf.float64)
                prob_state_010000 = tf.cast(tf.math.real(prob[0, 1, 0, 0, 0, 0]), dtype=tf.float64)
                prob_state_100000 = tf.cast(tf.math.real(prob[1, 0, 0, 0, 0, 0]), dtype=tf.float64)
                prob_state_000010 = tf.cast(tf.math.real(prob[0, 0, 0, 0, 1, 0]), dtype=tf.float64)
                prob_state_000001 = tf.cast(tf.math.real(prob[0, 0, 0, 0, 0, 1]), dtype=tf.float64)

                # Update memory variables
                memory_p3 = tf.tensor_scatter_nd_update(memory_p3, [[time_step % memory_depth]], [prob_state_000001])
                memory_p5 = tf.tensor_scatter_nd_update(memory_p5, [[time_step % memory_depth]], [prob_state_100000])
                memory_p6 = tf.tensor_scatter_nd_update(memory_p6, [[time_step % memory_depth]], [prob_state_000010])
                memory_p7 = tf.tensor_scatter_nd_update(memory_p7, [[time_step % memory_depth]], [prob_state_010000])

                # Compute the loss
                loss += MSEloss(y_train[i],prob_state_000100)

            # Compute gradients and update variables
            gradients = tape.gradient(loss, [phase1, phase2, phase4, phase8, phase9, phase10, phase11, phase12, memristor_weight])
            optimizer.apply_gradients(zip(gradients, [phase1, phase2, phase4, phase8, phase9, phase10, phase11, phase12, memristor_weight]))

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{float(loss):.4f}'})
                    
            logger.log_training_step(step, loss, phase1, phase2, phase4, phase8, phase9, phase10, phase11, phase12, memristor_weight)


            res_mem[('loss', 'tr', step)] = [loss.numpy()] #, phase1.numpy(), phase2.numpy(), phase4.numpy(), phase8.numpy(), phase9.numpy(), phase10.numpy(), phase11.numpy(), phase12.numpy(), memristor_weight.numpy()]


    final_metrics = {
        'final_loss': float(loss),
        #'final_phase1': float(phase1),
        #'final_phase3': float(phase3),
        'final_memristor_weight': float(memristor_weight),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'learning_rate': learning_rate,
        'cutoff_dim': cutoff_dim
    }
    logger.log_final_results(final_metrics)

    # Save trained model parameters
    trained_params = {
        #'phase1': phase1.numpy(),
        #'phase3': phase3.numpy(),
        'memristor_weight': memristor_weight.numpy(),
        'final_loss': loss.numpy(),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'res_mem': res_mem
    }
    logger.save_model_artifact(trained_params, 'trained_parameters.pkl')

    # Plot training results
    plot_training_results(res_mem, f"{logger.base_dir}/plots/training_results_+{param_id}.png")
    
        
    return res_mem, phase1, phase2, phase4, phase8, phase9, phase10, phase11, phase12

def predict_megabigmemristor(X_test: np.ndarray, 
                      y_test: np.ndarray, 
                      memory_depth: int, 
                      phase1: float, 
                      phase2: float, 
                      phase3: float,
                      phase4: float,
                      phase5: float,
                      phase6: float,
                      phase7: float, 
                      phase8: float,
                      phase9: float,
                      phase10: float, 
                      phase11: float, 
                      phase12: float,
                      memristor_weight: float, 
                      stochastic: bool, 
                      samples: int, 
                      var: float, 
                      cutoff_dim: int, 
                      logger: ExperimentLogger,
                      param_id: str = None):
    """
    Uses the trained mega big memristor model to make predictions on test data.
    """

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    encoded_phases = tf.constant(2 * np.arccos(X_test), dtype=tf.float64)

    # Initialize lists to store predictions and targets
    all_predictions = []
    targets = []

    # Initialize memory variables
    memory_p3 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p5 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p6 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p7 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    if stochastic:
        print(f"Running {samples} samples with variance {var}...")
    else:
        print("Running deterministic prediction...")
        samples = 1

    # print("Predicting on test data...")
    sample_pbar = trange(samples, desc='Prediction Samples', unit='sample')
    for sample in sample_pbar:
        sample_predictions = []

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
                phase3 = tf.acos(tf.sqrt(0.5))
                phase5 = tf.acos(tf.sqrt(0.5))
                phase6 = tf.acos(tf.sqrt(0.5))
                phase7 = tf.acos(tf.sqrt(0.5))
                    
                # Init Memristor Circuit
                memristor_circuit = MemristorMegaBigCircuit(phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, encoded_phases[i])
                results = eng.run(
                        memristor_circuit.build_circuit())
            else:
                phase3 = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p3) / memory_depth
                    ))
                phase5 = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p5) / memory_depth
                    ))
                phase6 = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p6) / memory_depth
                    ))
                phase7 = tf.acos(tf.sqrt(
                        memristor_weight * tf.reduce_sum(memory_p7) / memory_depth
                    ))


            memristor_circuit = MemristorMegaBigCircuit(phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, encoded_phases[i])
            results = eng.run(memristor_circuit.build_circuit())

            # Get probabilities from the circuit results
            prob = results.state.all_fock_probs()
            prob_state_000100 = tf.cast(tf.math.real(prob[0, 0, 0, 1, 0, 0]), dtype=tf.float64)
            # prob_state_001000 can be used for training with NLL
            prob_state_001000 = tf.cast(tf.math.real(prob[0, 0, 1, 0, 0, 0]), dtype=tf.float64)
            prob_state_010000 = tf.cast(tf.math.real(prob[0, 1, 0, 0, 0, 0]), dtype=tf.float64)
            prob_state_100000 = tf.cast(tf.math.real(prob[1, 0, 0, 0, 0, 0]), dtype=tf.float64)
            prob_state_000010 = tf.cast(tf.math.real(prob[0, 0, 0, 0, 1, 0]), dtype=tf.float64)
            prob_state_000001 = tf.cast(tf.math.real(prob[0, 0, 0, 0, 0, 1]), dtype=tf.float64)

            # Update memory variables
            memory_p3 = tf.tensor_scatter_nd_update(memory_p3, [[time_step % memory_depth]], [prob_state_000001])
            memory_p5 = tf.tensor_scatter_nd_update(memory_p5, [[time_step % memory_depth]], [prob_state_100000])
            memory_p6 = tf.tensor_scatter_nd_update(memory_p6, [[time_step % memory_depth]], [prob_state_000010])
            memory_p7 = tf.tensor_scatter_nd_update(memory_p7, [[time_step % memory_depth]], [prob_state_010000])


            sample_predictions.append(prob_state_000100.numpy())
            if sample == 0:
                targets.append(float(y_test[i].numpy()))

            # Update inner progress bar with current probabilities - do we need this?
            #phase_pbar.set_postfix({
            #    'prob_001': f'{float(prob_state_001):.4f}',
            #    'prob_010': f'{float(prob_state_010):.4f}'
            #})
            # Compute the loss
            prob_state_000100 = tf.cast(prob_state_000100, dtype=tf.float64)
            loss = tf.square(tf.abs(y_test[i] - prob_state_000100))
            logger.log_prediction_step(i, loss) #phase1_sample, phase3_sample, memristor_weight)

        all_predictions.append(sample_predictions)

    # Convert all_predictions to a NumPy array for easier manipulation
    all_predictions = np.array(all_predictions)

    if stochastic:
        # Calculate mean and standard deviation along the column axis
        final_predictions = np.mean(all_predictions, axis=0)
        predictive_uncertainty = np.std(all_predictions, axis=0)
        logger.log_prediction(final_predictions, predictive_uncertainty, samples)
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, f"{logger.base_dir}/plots/prediction_results_sample{samples}_{param_id}.png")
    else:
        final_predictions = all_predictions[0]
        predictive_uncertainty = np.array([])
        targets = np.array(targets)
        logger.log_prediction(final_predictions)
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, f"{logger.base_dir}/plots/prediction_results_deterministic_{param_id}.png")


    return final_predictions, targets, predictive_uncertainty, all_predictions


