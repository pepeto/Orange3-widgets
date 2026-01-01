import numpy as np
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, Domain, DiscreteVariable

# ZigZag pivot detection helpers (self-contained)
PEAK = 1
VALLEY = -1

def identify_initial_pivot(X, up_thresh, down_thresh):
    x0 = X[0]
    max_x = min_x = x0
    max_t = min_t = 0
    up_thresh += 1
    down_thresh += 1
    for t in range(1, len(X)):
        xt = X[t]
        if xt / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK
        if xt / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY
        if xt > max_x:
            max_x, max_t = xt, t
        if xt < min_x:
            min_x, min_t = xt, t
    return VALLEY if x0 < X[-1] else PEAK


def _to_ndarray(X):
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, (list, tuple)):
        return np.array(X)
    raise ValueError("Input must be a numpy array, list, or tuple")


def peak_valley_pivots_detailed(X, up_thresh, down_thresh,
                                 limit_to_finalized_segments=True,
                                 use_eager_switching_for_non_final=False):
    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')
    initial_pivot = identify_initial_pivot(X, up_thresh, down_thresh)
    n = len(X)
    pivots = np.zeros(n, dtype=int)
    trend = -initial_pivot
    last_t, last_x = 0, X[0]
    pivots[0] = initial_pivot
    up_thresh += 1
    down_thresh += 1
    for t in range(1, n):
        x = X[t]
        r = x / last_x
        if trend == VALLEY:
            if r >= up_thresh:
                pivots[last_t] = trend
                trend = PEAK
                last_x, last_t = x, t
            elif x < last_x:
                last_x, last_t = x, t
        else:
            if r <= down_thresh:
                pivots[last_t] = trend
                trend = VALLEY
                last_x, last_t = x, t
            elif x > last_x:
                last_x, last_t = x, t
    if limit_to_finalized_segments:
        if use_eager_switching_for_non_final and 0 < last_t < n - 1:
            pivots[last_t] = trend
            pivots[-1] = -trend
        else:
            if last_t == n - 1:
                pivots[last_t] = trend
            elif pivots[-1] == 0:
                pivots[-1] = -trend
    return pivots


def peak_valley_pivots(X, up_thresh, down_thresh):
    X = _to_ndarray(X)
    if not str(X.dtype).startswith('float'):
        X = X.astype(np.float64)
    return peak_valley_pivots_detailed(X, up_thresh, down_thresh)


def pivots_to_modes(pivots):
    modes = np.zeros(len(pivots), dtype=int)
    mode = -pivots[0]
    modes[0] = pivots[0]
    for t in range(1, len(pivots)):
        p = pivots[t]
        modes[t] = mode
        if p != 0:
            mode = -p
    return modes


class OWZigZag(OWWidget):
    name = "ZigZag"
    description = "Generate ZigZag and Trend columns based on swing percentage"
    icon = "icons/zigzag.svg"
    category = "Custom"
    keywords = ["zigzag", "trend", "pivots"]

    # Place parameters in mainArea and remove unused toggle handle
    want_control_area = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    swing = settings.Setting(10.0)
    feature_index = settings.Setting(0)

    def __init__(self):
        super().__init__()
        self.data = None
        box = gui.widgetBox(self.mainArea, "Parameters")
        gui.doubleSpin(
            box, self, "swing", 0.0, 100.0, 0.1,
            label="Swing Percentage (%):", callback=self.commit
        )
        self.feature_cb = gui.comboBox(
            box, self, "feature_index", items=[],
            label="Feature:", callback=self.commit
        )
        # Ensure the combo is wide enough to show values
        self.feature_cb.setMinimumWidth(120)
        gui.button(box, self, "Apply", callback=self.commit)
        self.resize(200, 100)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.feature_cb.clear()
        if data is not None:
            features = [var.name for var in data.domain.attributes if var.is_continuous]
            self.feature_index = 0
            self.feature_cb.addItems(features)
        self.commit()

    def commit(self):
        if self.data is None:
            self.Outputs.data.send(None)
            return
        features = [var.name for var in self.data.domain.attributes if var.is_continuous]
        if not features:
            self.Outputs.data.send(None)
            return
        idx = min(self.feature_index, len(features) - 1)
        feature = features[idx]
        col_idx = [var.name for var in self.data.domain.attributes].index(feature)
        X = self.data.X[:, col_idx]
        up = self.swing / 100.0
        down = -up
        pivots = peak_valley_pivots(X, up, down)
        modes = pivots_to_modes(pivots)
        pivot_codes = pivots + 1  # -1->0, 0->1, 1->2
        mode_codes = (modes == PEAK).astype(int)  # -1->0, 1->1

        suffix = str(int(self.swing * 100))
        zz_name = f"ZigZag{suffix}"
        tr_name = f"Trend{suffix}"
        zz_var = DiscreteVariable(zz_name, values=["-1", "0", "1"])
        tr_var = DiscreteVariable(tr_name, values=["-1", "1"])

        orig_attrs = list(self.data.domain.attributes)
        new_domain = Domain(orig_attrs + [zz_var, tr_var],
                            self.data.domain.class_vars,
                            self.data.domain.metas)
                            
        # Ensure correct shapes for hstack
        pigs = pivot_codes.reshape(-1, 1)
        mogs = mode_codes.reshape(-1, 1)
        
        # Safe concatenation
        new_X = np.hstack((self.data.X, pigs, mogs))
        
        # Handle Y and Metas safely (pass None if they don't exist effectively)
        Y_data = self.data.Y if self.data.domain.class_vars else None
        M_data = self.data.metas if self.data.domain.metas else None
        
        new_table = Table.from_numpy(new_domain, new_X, Y=Y_data, metas=M_data)
        self.Outputs.data.send(new_table)
