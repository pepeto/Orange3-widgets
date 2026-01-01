"""
Orange3 widget: GMM Clustering / Classifier
───────────────────────────────────────────
• GaussianMixture / BayesianGaussianMixture learner.
• Optional Hungarian mapping of components → true classes.
• Optional auto-selection of n_components by BIC / AIC sweep.
"""

# ── Imports ────────────────────────────────────────────────────────────
from AnyQt.QtWidgets import QMessageBox, QLabel
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.settings import Setting
from Orange.widgets import gui
from Orange.data import Table, Domain, ContinuousVariable
from Orange.classification import Learner, Model

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np


# ── Helper: optimal mapping via Hungarian ──────────────────────────────
def _optimal_mapping(y_true, raw_clusters, n_clusters):
    """
    Return dict {cluster_index: class_index} that maximises accuracy.
    Assumes y_true are integer-encoded (0…C-1).
    """
    cont = contingency_matrix(y_true, raw_clusters)
    if cont.shape[0] != cont.shape[1]:
        # Handle case where n_clusters != n_classes slightly more gracefully if needed
        # but linear_sum_assignment handles rectangular matrices fine.
        pass
    
    # Transpose because contingency_matrix is (true, pred), assignment needs cost matrix
    # We want to match row (true class) to col (cluster) to MAXIMIZE intersection.
    # linear_sum_assignment minimizes cost, so we pass negative counts.
    row_ind, col_ind = linear_sum_assignment(-cont)
    mapping = {c: r for r, c in zip(row_ind, col_ind)} # Cluster -> Class mapping

    # leftover clusters → majority vote
    for clu in set(range(n_clusters)) - set(row_ind):
        mapping[clu] = int(cont[clu].argmax())
    return mapping


# ── Learner / Model ────────────────────────────────────────────────────
class GMMBaseClassifier(Learner):
    """(B)GMM learner with optional Hungarian mapping."""
    def __init__(self, *,                       # keyword-only for clarity
                 model_type="gmm",
                 n_components=3,
                 covariance_type="diag",
                 max_iter=300,
                 random_state=42,
                 scaler_option=1,
                 weight_prior_type="dirichlet_process",
                 weight_prior_value=1.0,
                 use_hungarian=True):
        super().__init__()
        self.model_type = model_type
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler_option = scaler_option
        self.weight_prior_type = weight_prior_type
        self.weight_prior_value = weight_prior_value
        self.use_hungarian = use_hungarian

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_scaler(X, option):
        if option == 1:
            return StandardScaler().fit(X)
        if option == 2:
            return RobustScaler().fit(X)
        return None

    # ------------------------------------------------------------------ #
    def fit(self, Xtbl, Y=None, W=None):
        """
        Fit the mixture and, if possible, compute the Hungarian mapping.
        Works whether the class is in `Xtbl.domain` or passed separately as Y.
        """
        X = Xtbl.X if hasattr(Xtbl, "X") else Xtbl
        self.domain = getattr(Xtbl, "domain", None)

        # ---- retrieve true labels ------------------------------------
        y_true = None
        if self.domain and self.domain.class_var and self.domain.class_var.is_discrete:
            y_true = Xtbl.Y.ravel().astype(int)
        if Y is not None and len(Y):                       # override if provided
            y_true = np.asarray(Y).ravel().astype(int)

        n_classes = len(np.unique(y_true)) if y_true is not None else None

        # ---- scaling --------------------------------------------------
        self.scaler_ = self._build_scaler(X, self.scaler_option)
        X_scaled = self.scaler_.transform(X) if self.scaler_ else X

        # ---- mixture --------------------------------------------------
        mix_cls = GaussianMixture if self.model_type == "gmm" else BayesianGaussianMixture
        mix_args = dict(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        if self.model_type == "bgmm":
            mix_args.update(weight_concentration_prior_type=self.weight_prior_type,
                            weight_concentration_prior=self.weight_prior_value)

        self.gmm_ = mix_cls(**mix_args).fit(X_scaled)

        # ---- Hungarian mapping ---------------------------------------
        self.mapping_ = None
        if self.use_hungarian and y_true is not None:
            if self.n_components < n_classes:
                QMessageBox.warning(
                    None, "GMM Mapping",
                    "Components < classes: Hungarian mapping skipped."
                )
            else:
                raw_clusters = self.gmm_.predict(X_scaled)
                self.mapping_ = _optimal_mapping(
                    y_true, raw_clusters, n_clusters=self.n_components
                )

        self.n_classes_ = n_classes
        return GMMBaseModel(self)


class GMMBaseModel(Model):
    """Prediction wrapper compatible with Orange's ret flags."""
    def __init__(self, learner: GMMBaseClassifier):
        super().__init__(learner.domain)
        self.gmm_ = learner.gmm_
        self.scaler_ = learner.scaler_
        self.mapping_ = learner.mapping_
        self.n_classes_ = learner.n_classes_

    # ------------------------------------------------------------------ #
    def _probabilities(self, X):
        Xn = X.X if hasattr(X, "X") else X
        if self.scaler_ is not None:
            Xn = self.scaler_.transform(Xn)

        comp_probs = self.gmm_.predict_proba(Xn)

        # unsupervised → return component probs
        if self.mapping_ is None or self.n_classes_ is None:
            return comp_probs

        probs = np.zeros((comp_probs.shape[0], self.n_classes_))
        for k in range(comp_probs.shape[1]):
            # If cluster k maps to class cls, add its prob to that class
            cls = self.mapping_.get(k, None)
            if cls is not None and 0 <= cls < self.n_classes_:
                probs[:, cls] += comp_probs[:, k]

        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        return probs / row_sum

    # ------------------------------------------------------------------ #
    def __call__(self, data, ret=Model.Value):
        probs = self._probabilities(data)
        preds = probs.argmax(axis=1)

        if ret == Model.Probs:
            return probs
        if ret == Model.ValueProbs:
            return preds, probs
        return preds


# ── Orange3 Widget ─────────────────────────────────────────────────────
class OWGMM(OWWidget):
    name = "GMM Clustering"
    description = ("Gaussian/Bayesian Mixture clustering with learner "
                   "output and optional Hungarian mapping.")
    icon = "icons/GMM.svg"
    priority = 20
    category = "Custom"

    # Inputs / Outputs ---------------------------------------------------
    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        annotated_data = Output("Clustered Data", Table)
        model = Output("GMM Model", object, auto_summary=False)
        learner = Output("Learner", Learner)
        classifier = Output("Classifier", Model)

    # Settings -----------------------------------------------------------
    mixture_type = Setting(0)            # 0 = GMM, 1 = BGMM
    n_components = Setting(3)
    auto_tune = Setting(False)
    k_max = Setting(10)
    criterion_idx = Setting(0)           # 0=BIC, 1=AIC
    covariance_type = Setting("diag")
    _cov_index = Setting(2)
    max_iter = Setting(300)
    random_state = Setting(42)
    scaler_option = Setting(1)           # 0=None, 1=Standard, 2=Robust
    weight_prior_type = Setting("dirichlet_process")
    weight_prior_value = Setting(1.0)
    use_hungarian = Setting(True)

    want_main_area = False
    resizing_enabled = True

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def __init__(self):
        super().__init__()
        self.data = None
        self._cov_opts = ["full", "tied", "diag", "spherical"]
        self._scale_opts = ["No Scaler", "StandardScaler", "RobustScaler"]
        self._mix_opts = ["GaussianMixture", "BayesianGaussianMixture"]

        # GUI -----------------------------------------------------------
        box = gui.widgetBox(self.controlArea, "Settings")

        gui.comboBox(box, self, "mixture_type",
                     label="Mixture:", items=self._mix_opts,
                     callback=self.apply)

        self.comp_spin = gui.spin(box, self, "n_components", 1, 100,
                                  label="Components:", callback=self.apply)

        # Auto-tune block
        at_box = gui.widgetBox(box, "Auto-tune components")
        gui.checkBox(at_box, self, "auto_tune",
                     "Auto-tune by", callback=self._toggle_auto)
        gui.comboBox(at_box, self, "criterion_idx",
                     items=["BIC", "AIC"], label="Criterion:",
                     callback=self.apply)
        gui.spin(at_box, self, "k_max", 2, 50,
                 label="k-max:", callback=self.apply)

        # covariance combo
        gui.comboBox(box, self, "_cov_index",
                     label="Covariance:", items=self._cov_opts,
                     callback=lambda: (self._sync_cov(), self.apply()))

        gui.spin(box, self, "max_iter", 1, 2000,
                 label="Max iter:", callback=self.apply)
        gui.spin(box, self, "random_state", 0, 9999,
                 label="Random state:", callback=self.apply)

        gui.radioButtons(gui.widgetBox(box, "Scaler"), self, "scaler_option",
                         btnLabels=self._scale_opts, callback=self.apply)

        gui.checkBox(box, self, "use_hungarian",
                     "Optimal cluster-to-class mapping (Hungarian)",
                     callback=self.apply)

        # BGMM params
        bg_box = gui.widgetBox(box, "BGMM parameters")
        gui.comboBox(bg_box, self, "weight_prior_type",
                     label="Prior type:",
                     items=["dirichlet_process", "dirichlet_distribution"],
                     sendSelectedValue=True, valueType=str,
                     callback=self.apply)
        gui.doubleSpin(bg_box, self, "weight_prior_value",
                       0.001, 100.0, step=0.1,
                       label="Prior value:", callback=self.apply)
        
        self.bg_info = QLabel("")
        bg_box.layout().addWidget(self.bg_info)
        
        # Info label implementation (replacing self.info)
        self.info_label = QLabel("No data")
        box.layout().addWidget(self.info_label)

        gui.button(self.controlArea, self, "Run", callback=self.apply, default=True)
        self._toggle_auto()  # initialise spin enabled/disabled

    # ------------------------------------------------------------------ #
    # GUI helpers
    def _sync_cov(self):
        self.covariance_type = self._cov_opts[self._cov_index]

    def _toggle_auto(self):
        self.comp_spin.setDisabled(self.auto_tune)
        self.apply()

    # ------------------------------------------------------------------ #
    # Input slot
    @Inputs.data
    def set_data(self, data):
        self.data = data
        if data is None:
            self.info_label.setText("No data")
        else:
            self.info_label.setText(f"{len(data)} rows, {len(data.domain.attributes)} attrs")
        self.apply()

    # ------------------------------------------------------------------ #
    # Auto-tune n_components
    def _select_components(self, X_scaled):
        crit = "bic" if self.criterion_idx == 0 else "aic"
        best_k, best_score = None, np.inf
        for k in range(1, self.k_max + 1):
            gm = GaussianMixture(n_components=k,
                                 covariance_type=self.covariance_type,
                                 max_iter=self.max_iter,
                                 random_state=self.random_state).fit(X_scaled)
            score = gm.bic(X_scaled) if crit == "bic" else gm.aic(X_scaled)
            if score < best_score:
                best_k, best_score = k, score
        return best_k

    # ------------------------------------------------------------------ #
    # Main execution
    def apply(self):
        self.clear_messages()
        if self.data is None:
            self._send_none()
            return

        self._sync_cov()  # keep string in sync

        # Quick scaling for preview / auto-tune
        try:
            if self.scaler_option == 1:
                X_scaled = StandardScaler().fit_transform(self.data.X)
            elif self.scaler_option == 2:
                X_scaled = RobustScaler().fit_transform(self.data.X)
            else:
                X_scaled = self.data.X
        except Exception as err:
            self.error(f"Scaling failed: {err}")
            self._send_none()
            return

        if self.auto_tune:
            try:
                n_comps = self._select_components(X_scaled)
                self.n_components = n_comps
            except Exception as err:
                self.error(f"Auto-tune failed: {err}")
                self._send_none()
                return

        # Prepare True Labels for Hungarian safely
        Y = None
        if self.data.domain.class_var and self.data.domain.class_var.is_discrete:
            # We must pass indices (0..K-1) not raw float values 
            # Orange's Y for discrete is usually float indices, but we cast to int
            # and check if they are valid indices.
            Y = self.data.Y.ravel().astype(int)

        mix_type = "gmm" if self.mixture_type == 0 else "bgmm"

        learner = GMMBaseClassifier(
            model_type=mix_type,
            n_components=n_comps,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
            scaler_option=self.scaler_option,
            weight_prior_type=self.weight_prior_type,
            weight_prior_value=self.weight_prior_value,
            use_hungarian=self.use_hungarian,
        )

        # Train
        try:
            classifier = learner.fit(self.data, Y=Y)
            mix = classifier.gmm_
            labels = mix.predict(X_scaled)
            conf = mix.predict_proba(X_scaled).max(axis=1).reshape(-1, 1)
        except Exception as err:
            self.error(f"GMM fit failed: {err}")
            self._send_none()
            return

        # BGMM info
        if mix_type == "bgmm":
            eff = np.sum(mix.weights_ > 1e-2)
            self.bg_info.setText(f"Effective clusters: {eff}")
        else:
            self.bg_info.setText("")

        # Annotated output
        # Create result variables
        lab_var = DiscreteVariable("GMM_label", values=[str(i) for i in range(n_comps)])
        conf_var = ContinuousVariable("GMM_conf")
        
        # New Domain: Attributes + Class + Metas + [GMM_Label, GMM_Conf]
        extra_metas = [lab_var, conf_var]
        new_domain = Domain(self.data.domain.attributes,
                          self.data.domain.class_vars,
                          self.data.domain.metas + tuple(extra_metas))
        
        # Prepare data parts
        X_data = self.data.X
        Y_data = self.data.Y
        
        # Prepare new metas
        # self.data.metas might be (N, M) or (N, 0) or None
        M_current = self.data.metas if self.data.metas is not None else np.zeros((len(self.data), 0))
        
        # Stack new metas: reshape to ensuring (N, 1)
        new_meta_data = np.hstack([
            M_current,
            labels.reshape(-1, 1),
            conf.reshape(-1, 1)
        ])
        
        annotated = Table(new_domain, X_data, Y_data, new_meta_data)

        # Send outputs
        self.Outputs.annotated_data.send(annotated)
        self.Outputs.model.send(mix)
        self.Outputs.learner.send(learner)
        self.Outputs.classifier.send(classifier)
        
        # Update info label
        self.info_label.setText(f"{len(self.data)} rows, {n_comps} clusters found")

    # ------------------------------------------------------------------ #
    def _send_none(self):
        self.Outputs.annotated_data.send(None)
        self.Outputs.model.send(None)
        self.Outputs.learner.send(None)
        self.Outputs.classifier.send(None)
