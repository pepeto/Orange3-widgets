from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.settings import Setting
from Orange.widgets import gui
from AnyQt.QtWidgets import QMessageBox, QLabel
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from hmmlearn.hmm import GaussianHMM
import numpy as np

class OWHMM(OWWidget):
    name = "HMM Clustering"
    description = "Apply Hidden Markov Model clustering to data."
    icon = "icons/HMM.svg"
    priority = 21
    keywords = []
    help = "https://hmmlearn.readthedocs.io/en/latest/"

    # Use a separate executor for threading
    from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
    from orangewidget.utils.signals import Output as SignalOutput

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        annotated_data = Output("State Sequence", Table)
        model = Output("Trained Model", object)

    want_main_area = False
    resizing_enabled = True

    n_components = Setting(2)
    covariance_type = Setting("diag")
    _covariance_index = Setting(2)
    n_iter = Setting(100)
    random_state = Setting(42)
    scaler_option = Setting(1)  # 0: No Scaler, 1: StandardScaler, 2: RobustScaler

    def __init__(self):
        super().__init__()
        self.data = None
        self._covariance_options = ["full", "tied", "diag", "spherical"]
        self._scaler_options = ["No Scaler", "StandardScaler", "RobustScaler"]

        box = gui.widgetBox(self.controlArea, "HMM Settings")

        scaler_box = gui.widgetBox(box, "Scaler")
        gui.radioButtons(scaler_box, self, "scaler_option", 
		btnLabels=self._scaler_options, callback=self.apply)

        gui.spin(box, self, "n_components", 1, 10, label="Number of States:")
        gui.comboBox(box, self, "_covariance_index", label="Covariance type:",
                     items=self._covariance_options, callback=self._update_covariance_type)
        gui.spin(box, self, "n_iter", 10, 500, label="Max iterations:")
        gui.spin(box, self, "random_state", 0, 9999, label="Random state:")

        # Info label
        self.info_label = QLabel("No data")
        box.layout().addWidget(self.info_label)

        gui.button(box, self, "Run", callback=self.apply)


        self._executor = ThreadExecutor()
        self._task = None

    @Inputs.data
    def set_data(self, data):
        self.cancel()
        self.data = data
        if data is not None:
            self.info_label.setText(f"{len(data)} rows, {len(data.domain.attributes)} features")
        else:
            self.info_label.setText("No data")
        self._update_covariance_type()
        self.apply()

    def cancel(self):
        if self._task is not None:
             self._task.cancel()
             self._task = None

    def _update_covariance_type(self):
        self.covariance_type = self._covariance_options[self._covariance_index]

    def _scale_data(self, X):
        if self.scaler_option == 1:
            from sklearn.preprocessing import StandardScaler
            return StandardScaler().fit_transform(X)
        elif self.scaler_option == 2:
            from sklearn.preprocessing import RobustScaler
            return RobustScaler().fit_transform(X)
        return X

    def apply(self):
        if self.data is None:
            self.Outputs.annotated_data.send(None)
            self.Outputs.model.send(None)
            return

        self.cancel() # Cancel previous run
        
        # Prepare data on main thread
        X = self._scale_data(self.data.X)
        
        if self.n_components * X.shape[1] > X.shape[0]:
             self.info_label.setText("Warning: check console (Too many states)")
             print("Warning: Too many states for number of samples.")

        # Launch background task
        self.info_label.setText("Training HMM...")
        self._task = task = HMMTask(
            X, 
            self.n_components, 
            self.covariance_type, 
            self.n_iter, 
            self.random_state
        )
        self._task.watcher.finished.connect(self._on_finished)
        self._task.start()

    @gui.deferred
    def _on_finished(self):
        if self._task is None:
             return
        
        try:
            hmm, states, state_probs = self._task.future.result()
        except Exception as e:
            self.info_label.setText("Error during training")
            QMessageBox.critical(self, "HMM Error", str(e))
            self.Outputs.annotated_data.send(None)
            self.Outputs.model.send(None)
            return
        finally:
            self._task = None

        # Output table construction (Robust)
        state_var = DiscreteVariable("HMM_state", values=[str(i) for i in range(self.n_components)])
        prob_var = ContinuousVariable("HMM_confidence")
        
        # Create robust new domain
        new_domain = Domain(
            self.data.domain.attributes,
            self.data.domain.class_vars,
            self.data.domain.metas + (state_var, prob_var)
        )
        
        # Prepare metas robustly
        M_curr = self.data.metas if self.data.metas is not None else np.zeros((len(self.data), 0))
        new_metas = np.hstack([
            M_curr,
            states.reshape(-1, 1), 
            state_probs
        ])
        
        annotated = Table(new_domain, self.data.X, self.data.Y, new_metas)

        self.Outputs.annotated_data.send(annotated)
        self.Outputs.model.send(hmm)
        self.info_label.setText(f"Done. Found {self.n_components} states.")

# Thread Task Class
class HMMTask:
    def __init__(self, X, n_components, covariance_type, n_iter, random_state):
        self.future = None
        self.watcher = FutureWatcher()
        self._executor = ThreadExecutor() # temporary executor for the task instance? No, we should use widget's
        self.X = X
        self.params = (n_components, covariance_type, n_iter, random_state)
    
    def start(self):
        self.future = self._executor.submit(run_hmm, self.X, *self.params)
        self.watcher.bind(self.future)
    
    def cancel(self):
        if self.future:
            self.future.cancel()

def run_hmm(X, n_components, covariance_type, n_iter, random_state):
    # This runs in background thread
    hmm = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    hmm.fit(X)
    states = hmm.predict(X)
    state_probs = hmm.predict_proba(X).max(axis=1).reshape(-1, 1)
    return hmm, states, state_probs

    def send_report(self):
        if self.data is not None:
             self.report_items((
                ("Model", "HMM (Hidden Markov Model)"),
                ("States", self.n_components),
                ("Covariance type", self.covariance_type),
                ("Max iterations", self.n_iter)
            ))
