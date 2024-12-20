# python
# filepath: /Users/anani/Code/uq-qnn/src/models/uq_6x6.py
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
from tqdm import tqdm, trange
import random as rd
import uncertainty_toolbox as uct

from src.plotting import plot_training_results, plot_predictions_new
from src.logger import ExperimentLogger
from src.quantum import MemristorMegaBigCircuit

# Suppresses TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class MegaBigMemristorModel:
    def __init__(self, training_steps, memory_depth, learning_rate, cutoff_dim,
                 stochastic, samples, variance, logger: ExperimentLogger = None, param_id=None):
        # Logger
        self.logger = logger

        # Training parameters
        self.training_steps = training_steps
        self.memory_depth = memory_depth
        self.learning_rate = learning_rate
        self.cutoff_dim = cutoff_dim
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
        self.phase2 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                  constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        #
        self.phase4 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                  constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        #
        #
        #
        self.phase8 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                  constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        self.phase9 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                  constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        self.phase10 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                   constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        self.phase11 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                   constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        self.phase12 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                                   constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        self.memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                                            constraint=lambda z: tf.clip_by_value(z, 0.01, 1))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})

        # Prediction variables
        self.final_predictions = None
        self.targets = []
        self.predictive_uncertainty = None
        self.all_predictions = []

        # Evaluation metrics
        self.uq_metrics = None
        self.uq_metric_categories = None

    def train(self, X_train, y_train, plot=False):
        if self.logger is not None:
            self.logger.log_initial_training_phase(
                self.phase1, self.phase2, self.phase4, self.phase8,
                self.phase9, self.phase10, self.phase11, self.phase12,
                self.memristor_weight
            )
        res_mem = {}
        encoded_phases = tf.constant(2 * np.arccos(X_train), dtype=tf.float64)
        num_samples = len(encoded_phases)

        # Initialize memory variables
        memory_p3 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
        memory_p5 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
        memory_p6 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
        memory_p7 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)

        # Training loop
        pbar = trange(self.training_steps, desc='Training', unit='step')
        for step in pbar:
            if self.eng.run_progs:
                self.eng.reset()

            with tf.GradientTape() as tape:
                loss = 0.0

                for i in tqdm(range(num_samples),
                              desc=f'Step {step+1}/{self.training_steps}',
                              leave=False,
                              unit='sample'):
                    time_step = i % self.memory_depth

                    if i == 0:
                        memristor_phase3 = tf.acos(tf.sqrt(0.5))
                        memristor_phase5 = tf.acos(tf.sqrt(0.5))
                        memristor_phase6 = tf.acos(tf.sqrt(0.5))
                        memristor_phase7 = tf.acos(tf.sqrt(0.5))
                    else:
                        avg_p3 = tf.reduce_mean(memory_p3)
                        avg_p5 = tf.reduce_mean(memory_p5)
                        avg_p6 = tf.reduce_mean(memory_p6)
                        avg_p7 = tf.reduce_mean(memory_p7)

                        memristor_phase3 = tf.acos(tf.sqrt(
                            avg_p3 + self.memristor_weight * avg_p7
                        ))
                        memristor_phase5 = tf.acos(tf.sqrt(
                            avg_p5 + self.memristor_weight * avg_p3
                        ))
                        memristor_phase6 = tf.acos(tf.sqrt(
                            avg_p6 + self.memristor_weight * avg_p5
                        ))
                        memristor_phase7 = tf.acos(tf.sqrt(
                            avg_p7 + self.memristor_weight * avg_p6
                        ))

                    # Build the circuit
                    memristor_circuit = MemristorMegaBigCircuit(
                        phase1=self.phase1, phase2=self.phase2,
                        phase3=memristor_phase3, phase4=self.phase4,
                        phase5=memristor_phase5, phase6=memristor_phase6,
                        phase7=memristor_phase7, phase8=self.phase8,
                        phase9=self.phase9, phase10=self.phase10,
                        phase11=self.phase11, phase12=self.phase12,
                        encoded_phases=encoded_phases[i]
                    )

                    results = self.eng.run(memristor_circuit.build_circuit())
                    prob = results.state.all_fock_probs()

                    # Assume we're interested in the probability of state [0, 0, 1, 0, 0, 0]
                    prob_state_001000 = tf.cast(tf.math.real(prob[0, 0, 1, 0, 0, 0]), dtype=tf.float64)

                    # Update memory variables
                    memory_p3 = tf.tensor_scatter_nd_update(memory_p3, [[time_step]], [prob_state_001000])
                    memory_p5 = tf.tensor_scatter_nd_update(memory_p5, [[time_step]], [prob_state_001000])
                    memory_p6 = tf.tensor_scatter_nd_update(memory_p6, [[time_step]], [prob_state_001000])
                    memory_p7 = tf.tensor_scatter_nd_update(memory_p7, [[time_step]], [prob_state_001000])

                    # Compute loss
                    loss += tf.square(tf.abs(y_train[i] - prob_state_001000))

                # Compute gradients and update parameters
                gradients = tape.gradient(loss, [
                    self.phase1, self.phase2, self.phase4, self.phase8,
                    self.phase9, self.phase10, self.phase11, self.phase12,
                    self.memristor_weight
                ])

                self.optimizer.apply_gradients(zip(gradients, [
                    self.phase1, self.phase2, self.phase4, self.phase8,
                    self.phase9, self.phase10, self.phase11, self.phase12,
                    self.memristor_weight
                ]))

                pbar.set_postfix({'loss': f'{float(loss):.4f}'})
                if self.logger is not None:
                    self.logger.log_training_step(
                        step, loss, self.phase1, self.phase2, self.phase4,
                        self.phase8, self.phase9, self.phase10,
                        self.phase11, self.phase12, self.memristor_weight
                    )
                res_mem[('loss', 'tr', step)] = [
                    loss.numpy(), self.phase1.numpy(), self.phase2.numpy(),
                    self.phase4.numpy(), self.phase8.numpy(), self.phase9.numpy(),
                    self.phase10.numpy(), self.phase11.numpy(), self.phase12.numpy(),
                    self.memristor_weight.numpy()
                ]

        # Final metrics and logging
        final_metrics = {
            'final_loss': float(loss),
            'final_memristor_weight': float(self.memristor_weight),
            'memory_depth': self.memory_depth,
            'training_steps': self.training_steps,
            'learning_rate': self.learning_rate,
            'cutoff_dim': self.cutoff_dim
        }
        if self.logger is not None:
            self.logger.log_final_results(final_metrics)

        # Save trained model parameters
        trained_params = {
            'phase1': self.phase1.numpy(),
            'phase2': self.phase2.numpy(),
            'phase4': self.phase4.numpy(),
            'phase8': self.phase8.numpy(),
            'phase9': self.phase9.numpy(),
            'phase10': self.phase10.numpy(),
            'phase11': self.phase11.numpy(),
            'phase12': self.phase12.numpy(),
            'memristor_weight': self.memristor_weight.numpy(),
            'final_loss': loss.numpy(),
            'memory_depth': self.memory_depth,
            'training_steps': self.training_steps,
            'res_mem': res_mem
        }
        if self.logger is not None:
            self.logger.save_model_artifact(trained_params, 'trained_parameters.pkl')

        # Plot training results
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
            memory_p3 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
            memory_p5 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
            memory_p6 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)
            memory_p7 = tf.Variable(np.zeros(self.memory_depth), dtype=tf.float32)

            if self.stochastic:
                phase1_sample = np.random.normal(self.phase1.numpy(), self.var)
                phase2_sample = np.random.normal(self.phase2.numpy(), self.var)
                phase4_sample = np.random.normal(self.phase4.numpy(), self.var)
                phase8_sample = np.random.normal(self.phase8.numpy(), self.var)
                phase9_sample = np.random.normal(self.phase9.numpy(), self.var)
                phase10_sample = np.random.normal(self.phase10.numpy(), self.var)
                phase11_sample = np.random.normal(self.phase11.numpy(), self.var)
                phase12_sample = np.random.normal(self.phase12.numpy(), self.var)
                memristor_weight_sample = np.random.normal(self.memristor_weight.numpy(), self.var)
            else:
                phase1_sample = self.phase1.numpy()
                phase2_sample = self.phase2.numpy()
                phase4_sample = self.phase4.numpy()
                phase8_sample = self.phase8.numpy()
                phase9_sample = self.phase9.numpy()
                phase10_sample = self.phase10.numpy()
                phase11_sample = self.phase11.numpy()
                phase12_sample = self.phase12.numpy()
                memristor_weight_sample = self.memristor_weight.numpy()

            phase_pbar = tqdm(range(len(encoded_phases)),
                              desc=f'Sample {sample+1}/{self.samples}',
                              leave=False,
                              unit='phase')

            for i in phase_pbar:
                time_step = i % self.memory_depth

                if i == 0:
                    memristor_phase3 = tf.acos(tf.sqrt(0.5))
                    memristor_phase5 = tf.acos(tf.sqrt(0.5))
                    memristor_phase6 = tf.acos(tf.sqrt(0.5))
                    memristor_phase7 = tf.acos(tf.sqrt(0.5))
                else:
                    avg_p3 = tf.reduce_mean(memory_p3)
                    avg_p5 = tf.reduce_mean(memory_p5)
                    avg_p6 = tf.reduce_mean(memory_p6)
                    avg_p7 = tf.reduce_mean(memory_p7)

                    memristor_phase3 = tf.acos(tf.sqrt(
                        avg_p3 + memristor_weight_sample * avg_p7
                    ))
                    memristor_phase5 = tf.acos(tf.sqrt(
                        avg_p5 + memristor_weight_sample * avg_p3
                    ))
                    memristor_phase6 = tf.acos(tf.sqrt(
                        avg_p6 + memristor_weight_sample * avg_p5
                    ))
                    memristor_phase7 = tf.acos(tf.sqrt(
                        avg_p7 + memristor_weight_sample * avg_p6
                    ))

                memristor_circuit = MemristorMegaBigCircuit(
                    phase1=phase1_sample, phase2=phase2_sample,
                    phase3=memristor_phase3, phase4=phase4_sample,
                    phase5=memristor_phase5, phase6=memristor_phase6,
                    phase7=memristor_phase7, phase8=phase8_sample,
                    phase9=phase9_sample, phase10=phase10_sample,
                    phase11=phase11_sample, phase12=phase12_sample,
                    encoded_phases=encoded_phases[i]
                )

                results = eng.run(memristor_circuit.build_circuit())
                prob = results.state.all_fock_probs()

                prob_state_001000 = tf.cast(tf.math.real(prob[0, 0, 1, 0, 0, 0]), dtype=tf.float32)

                # Update memory variables
                memory_p3 = tf.tensor_scatter_nd_update(memory_p3, [[time_step]], [prob_state_001000])
                memory_p5 = tf.tensor_scatter_nd_update(memory_p5, [[time_step]], [prob_state_001000])
                memory_p6 = tf.tensor_scatter_nd_update(memory_p6, [[time_step]], [prob_state_001000])
                memory_p7 = tf.tensor_scatter_nd_update(memory_p7, [[time_step]], [prob_state_001000])

                sample_predictions.append(prob_state_001000.numpy())
                if sample == 0:
                    self.targets.append(float(y_test[i].numpy()))

            self.all_predictions.append(sample_predictions)

        self.all_predictions = np.array(self.all_predictions)

        if self.stochastic:
            self.final_predictions = np.mean(self.all_predictions, axis=0)
            self.predictive_uncertainty = np.std(self.all_predictions, axis=0)
            if self.logger is not None:
                self.logger.log_prediction(
                    self.final_predictions, self.predictive_uncertainty, self.samples
                )
            if plot and self.logger is not None:
                plot_predictions_new(
                    X_test, y_test, self.final_predictions, self.predictive_uncertainty,
                    f"{self.logger.base_dir}/plots/prediction_results_sample{self.samples}_{self.param_id}.png"
                )
            elif plot:
                plot_predictions_new(
                    X_test, y_test, self.final_predictions, self.predictive_uncertainty
                )
        else:
            self.final_predictions = self.all_predictions[0]
            self.predictive_uncertainty = np.array([])
            self.targets = np.array(self.targets)
            if self.logger is not None:
                self.logger.log_prediction(self.final_predictions)
            if plot and self.logger is not None:
                plot_predictions_new(
                    X_test, y_test, self.final_predictions, None,
                    f"{self.logger.base_dir}/plots/prediction_results_deterministic_{self.param_id}.png"
                )
            elif plot:
                plot_predictions_new(X_test, y_test, self.final_predictions, None)

    def evaluate(self):
        """Computes UQ metrics

        Interpretation:
        - final_predictions: predictions from circuit, np.array
        - targets: data points, np.array
        - predictive_uncertainty: predictive uncertainty from circuit, np.array 

        Returns:
            Dictionary containing all metrics. Accuracy metrics: Mean average error ('mae'), Root mean squared
            error ('rmse'), Median absolute error ('mdae'), Mean absolute
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
                self.uq_metrics = []  # TODO: define empty result
            # Categories when predictive uncertainty is present
            self.uq_metric_categories = [
                "scoring_rule",
                "avg_calibration",
                "sharpness",
                "accuracy",
            ]
        else:
            # Categories when no predictive uncertainty is present
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