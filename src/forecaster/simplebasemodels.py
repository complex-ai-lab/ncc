import numpy as np
import torch
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA


def thetamodel_forecast_helper(data, period, steps_ahead):
    """Forecast using a thetamodel. """
    tm = ThetaModel(endog=data, period=period)
    res = tm.fit()
    predictions = res.forecast(steps_ahead).values
    return predictions


def thetamodel_forecast(input_data, target_idx, steps_ahead, period=52):
    """Input: torch.Tensor in the shape of (batch_size x features x seq length)."""
    input_data = input_data.numpy()
    predictions = []
    for i in range(len(input_data)):
        current_data = input_data[i, :, target_idx]
        current_pred = thetamodel_forecast_helper(current_data, period, steps_ahead)
        predictions.append(current_pred)
    predictions = np.array(predictions)
    return predictions


class ThetaModelWrapper:
    def __init__(self, aheads, period, target_idx, ) -> None:
        self.aheads = aheads
        self.period = period
        self.idx = target_idx
    
    def forward(self, input_data):
        return thetamodel_forecast(input_data, self.idx, self.aheads, self.period)
    
    def eval(self):
        pass


class RandomForestWrapper:
    def __init__(self, aheads, target_idx) -> None:
        self.aheads = aheads
        self.target_idx = target_idx
    
    def train(self, dataloader):
        def get_dataset(target_idx, dataloader):
            Xs = []
            ys = []
            for batch in dataloader:
                # get data, shape of x is (batch_size x features x seq length)
                region, meta, x, x_mask, y, y_mask, weekid = batch
                current_X = x[:, :, target_idx]
                current_y = y[:, :]
                Xs.append(current_X)
                ys.append(current_y[:, :, 0])
            Xs = torch.concat(Xs, dim=0)
            ys = torch.concat(ys, dim=0)
            return Xs.numpy(), ys.numpy()
        Xs, ys = get_dataset(self.target_idx, dataloader=dataloader)
        regr = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=1000)
        regr.fit(Xs, ys)
        self.regr = regr

    def forward(self, input_data):
        """Input: torch.Tensor in the shape of (batch_size x features x seq length)."""
        input_data = input_data.numpy()[:, :, self.target_idx]
        predictions = self.regr.predict(input_data)
        return predictions
    
    def eval(self):
        pass


class ArimaWrapper:
    def __init__(self, aheads, target_idx) -> None:
        self.aheads = aheads
        self.target_idx = target_idx
    
    def train(self, dataloader):
        pass

    def forward(self, input_data):
        """Input: torch.Tensor in the shape of (batch_size x features x seq length)."""
        input_data = input_data.numpy()[:, :, self.target_idx]
        batch_size = input_data.shape[0]
        predictions = []
        for i in range(batch_size):
            current_input_data = input_data[i, :]
            # print(input_data.shape)
            regr = ARIMA(current_input_data, order=(3,1,0))
            regr = regr.fit()
            current_pred = regr.forecast()
            predictions.append(np.array([current_pred[0] for _ in range(self.aheads)]))
        return np.array(predictions)
    
    def eval(self):
        pass


# model = ARIMA(series, order=(5,1,0))
    
    