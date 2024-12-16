import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import random as rd
from tqdm import tqdm, trange
import uncertainty_toolbox as uct

from src.plotting import plot_training_results, plot_predictions_new
from src.logger import ExperimentLogger
from src.quantum import MemristorCircuit


# Supresses TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class MemristorModel:
    def __init__(self, training_steps, memory_depth, learning_rate, cutoff_dim, stochastic, samples, variance,
                 logger: ExperimentLogger = None, param_id = None):
        
        # Logger
        self.logger = logger
        
        # Training parameters
        self.training_steps = training_steps
        self.memory_depth = memory_depth
        self.learning_rate = learning_rate
        self.cutoff_dim = cutoff_dim
        self.logger = logger
        self.param_id = param_id
        
        # Prediction parameters
        self.stochastic = stochastic
        self.samples = samples
        self.var = variance

        # Initialize model parameters
        np.random.seed(42)
        tf.random.set_seed(42)
        rd.seed(42)

        self.phase1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                  constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        self.phase3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                  constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        self.memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                                            constraint=lambda z: tf.clip_by_value(z, 0.01, 1))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # predict var
        self.final_predictions = None
        self.targets = []
        self.predictive_uncertainty = None
        self.all_predictions = []
        
        # Evaluation
        self.uq_metrics = None
        self.uq_metric_categories = None
        

    def train(self, X_train, y_train, plot=False):
        if self.logger is not None:
            self.logger.log_initial_training_phase(self.phase1, self.phase3, self.memristor_weight)
        res_mem = {}
        encoded_phases = tf.constant(2 * np.arccos(X_train), dtype=tf.float64)
        num_samples = len(encoded_phases)

        # Initialize memory variables
        # TODO: can we use just numpy?
        memory_p1 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
        memory_p2 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
        cycle_index = 0

        # Training loop
        pbar = trange(self.training_steps, desc='Training', unit='step')
        for step in pbar:
            # TODO: eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
            
            if self.eng.run_progs:
                self.eng.reset()

            with tf.GradientTape() as tape:
                loss = 0.0

                for i in tqdm(range(num_samples),
                              desc=f'Step {step+1}/{self.training_steps}',
                              leave=False,
                              unit='sample'):
                    # TODO: time_step = i % self.memory_depth?
                    time_step = i - cycle_index * self.memory_depth
                    if time_step == self.memory_depth - 1:
                        cycle_index += 1

                    if i == 0:
                        memristor_phase = tf.acos(tf.sqrt(0.5))
                    else:
                        memristor_phase = tf.acos(tf.sqrt(
                            tf.reduce_sum(memory_p1) / self.memory_depth +
                            self.memristor_weight * tf.reduce_sum(memory_p2) / self.memory_depth
                        ))

                    memristor_circuit = MemristorCircuit(self.phase1, memristor_phase, self.phase3, encoded_phases[i])
                    results = self.eng.run(memristor_circuit.build_circuit())

                    prob = results.state.all_fock_probs()
                    prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float64)
                    prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float64)

                    memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [[time_step % self.memory_depth]], [prob_state_010])
                    memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [[time_step % self.memory_depth]], [prob_state_001])

                    loss += tf.square(tf.abs(y_train[i] - prob_state_001))

                gradients = tape.gradient(loss, [self.phase1, self.phase3, self.memristor_weight])
                self.optimizer.apply_gradients(zip(gradients, [self.phase1, self.phase3, self.memristor_weight]))

                pbar.set_postfix({'loss': f'{float(loss):.4f}'})
                if self.logger is not None:
                    self.logger.log_training_step(step, loss, self.phase1, self.phase3, self.memristor_weight)
                res_mem[('loss', 'tr', step)] = [loss.numpy(), self.phase1.numpy(), self.phase3.numpy(), self.memristor_weight.numpy()]

        # TODO: Use config for that
        final_metrics = {
            'final_loss': float(loss),
            'final_phase1': float(self.phase1),
            'final_phase3': float(self.phase3),
            'final_memristor_weight': float(self.memristor_weight),
            'memory_depth': self.memory_depth,
            'training_steps': self.training_steps,
            'learning_rate': self.learning_rate,
            'cutoff_dim': self.cutoff_dim
        }
        if self.logger is not None:
            self.logger.log_final_results(final_metrics)

        trained_params = {
            'phase1': self.phase1.numpy(),
            'phase3': self.phase3.numpy(),
            'memristor_weight': self.memristor_weight.numpy(),
            'final_loss': loss.numpy(),
            'memory_depth': self.memory_depth,
            'training_steps': self.training_steps,
            'res_mem': res_mem
        }
        
        if self.logger is not None:
            self.logger.save_model_artifact(trained_params, 'trained_parameters.pkl')

        # TODO: change self.logger to config or so
        if plot and self.logger is not None:
            plot_training_results(res_mem, f"{self.logger.base_dir}/plots/training_results_{self.param_id}.png")
        elif plot:
            plot_training_results(res_mem)
        return res_mem

    def predict(self, X_test, y_test, plot=False):
        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        encoded_phases = tf.constant(2 * np.arccos(X_test), dtype=tf.float64)


        if self.stochastic:
            print(f"Running {self.samples} samples with variance {self.var}...")
        else:
            print("Running deterministic prediction...")
            self.samples = 1

        sample_pbar = trange(self.samples, desc='Prediction Samples', unit='sample')
        for sample in sample_pbar:
            sample_predictions = []
            memory_p1 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
            memory_p2 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
            cycle_index = 0

            if self.stochastic:
                phase1_sample = np.random.normal(self.phase1.numpy(), self.var)
                phase3_sample = np.random.normal(self.phase3.numpy(), self.var)
            else:
                phase1_sample = self.phase1.numpy()
                phase3_sample = self.phase3.numpy()

            phase_pbar = tqdm(range(len(encoded_phases)),
                              desc=f'Sample {sample+1}/{self.samples}',
                              leave=False,
                              unit='phase')

            for i in phase_pbar:
                time_step = i - cycle_index * self.memory_depth
                if time_step == self.memory_depth - 1:
                    cycle_index += 1
                if i == 0:
                    memristor_phase = tf.acos(tf.sqrt(0.5))
                else:
                    memristor_phase = tf.acos(tf.sqrt(
                        tf.reduce_sum(memory_p1) / self.memory_depth +
                        self.memristor_weight * tf.reduce_sum(memory_p2) / self.memory_depth
                    ))

                memristor_circuit = MemristorCircuit(phase1_sample, memristor_phase, phase3_sample, encoded_phases[i])
                results = eng.run(memristor_circuit.build_circuit())

                prob = results.state.all_fock_probs()
                prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)
                prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)

                memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [[time_step % self.memory_depth]], [prob_state_010])
                memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [[time_step % self.memory_depth]], [prob_state_001])

                sample_predictions.append(prob_state_001.numpy())
                if sample == 0:
                    self.targets.append(float(y_test[i].numpy()))

                phase_pbar.set_postfix({
                    'prob_001': f'{float(prob_state_001):.4f}',
                    'prob_010': f'{float(prob_state_010):.4f}'
                })
                prob_state_001 = tf.cast(prob_state_001, dtype=tf.float64)
                loss = tf.square(tf.abs(y_test[i] - prob_state_001))
                
                if self.logger is not None:
                    self.logger.log_prediction_step(i, loss, phase1_sample, phase3_sample, self.memristor_weight)

            self.all_predictions.append(sample_predictions)

        self.all_predictions = np.array(self.all_predictions)

        if self.stochastic:
            self.final_predictions = np.mean(self.all_predictions, axis=0)
            self.predictive_uncertainty = np.std(self.all_predictions, axis=0)
            
            if self.logger is not None:
                self.logger.log_prediction(self.final_predictions, self.predictive_uncertainty, self.samples)
                
            if plot and self.logger is not None:
                plot_predictions_new(X_test, y_test, self.final_predictions, self.predictive_uncertainty)
                
            elif plot and self.logger is None:  
                plot_predictions_new(X_test, y_test, self.final_predictions, self.predictive_uncertainty,
                                    f"{self.logger.base_dir}/plots/prediction_results_sample{self.samples}_{self.param_id}.png")
                
        else:
            self.final_predictions = self.all_predictions[0]
            self.predictive_uncertainty = np.array([])
            self.targets = np.array(self.targets)
            
            if self.logger is not None:
                self.logger.log_prediction(self.final_predictions)
                
            if plot and self.logger is not None:
                plot_predictions_new(X_test, y_test, self.final_predictions, self.predictive_uncertainty)
            
            elif plot and self.logger is None:
                plot_predictions_new(X_test, y_test, self.final_predictions, self.predictive_uncertainty,
                                        f"{self.logger.base_dir}/plots/prediction_results_deterministic_{self.param_id}.png")

    def evaluate(self):
        #idea compute eval metrics for selective prediction and full version
        """ Copmutes UQ metrics
        
        Interpretation:
        - final_predictions: predictions from circuit, np.array
        - targets: data points, np.array
        - predictive_uncertainty: predictive uncertainty from ciruit, np.array 

        Returns:
            Dictionary containing all metrics. Accuracy metrics:  Mean average error ('mae'), Root mean squared
            error ('rmse'), Median absolute error ('mdae'),  Mean absolute
            relative percent difference ('marpd'), r^2 ('r2'), and Pearson's
            correlation coefficient ('corr').
        """

        # Convert to numpy arrays
        final_predictions = np.array(self.final_predictions)
        targets = np.array(self.targets)
        predictive_uncertainty = np.array(self.predictive_uncertainty)
        param_id = self.param_id


        if len(predictive_uncertainty) > 0:
        
            if len(final_predictions) > 0:
                self.uq_metrics = uct.metrics.get_all_metrics(
                    final_predictions,
                    predictive_uncertainty,
                    targets,
                    verbose=False,
                    )
            else:
                self.uq_metrics = [] # TODO: define empty result
            # categories when predictive uncertainty is present
            self.uq_metric_categories = [
                "scoring_rule",
                "avg_calibration",
                "sharpness",
                "accuracy",
            ]

        else:
            # categories when no predictive uncertainty is present
            self.uq_metric_categories = ["accuracy"]
            self.uq_metrics = {
                "accuracy": uct.metrics.get_all_accuracy_metrics(
                    final_predictions, targets
                )
            }

        if self.logger is not None:
            self.logger.log_evaluation_metrics(self.uq_metrics, param_id)
        else:
            print(self.uq_metrics)

        return self.uq_metrics, self.uq_metric_categories