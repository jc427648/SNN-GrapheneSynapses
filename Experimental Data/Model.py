import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
import optuna


class VTEAMMod:
    def __init__(
        self,
        wInit,
        alphaSetp,
        alphaSetn,
        alphaResetp,
        alphaResetn,
        kSetp,
        kSetn,
        kResetp,
        kResetn,
        ROn,
        ROff,
        VSetp,
        VSetn,
        VResetp,
        VResetn,
    ):
        self.w = wInit
        self.dw_dt = 0
        self.wMax = 1
        self.wMin = 0
        self.alphaSetp = alphaSetp
        self.alphaSetn = alphaSetn
        self.alphaResetp = alphaResetp
        self.alphaResetn = alphaResetn

        self.kSetp = kSetp
        self.kSetn = kSetn
        self.kResetp = kResetp
        self.kResetn = kResetn

        self.ROn = ROn
        self.R = ROn + (ROff - ROn) / (self.wMax - self.wMin) * (wInit - self.wMin)
        self.ROff = ROff
        self.VSetp = VSetp
        self.VSetn = VSetn
        self.VResetp = VResetp
        self.VResetn = VResetn

        self.SetPState = 0
        self.SetNState = 0
        self.ResetPState = 0
        self.ResetNState = 0

    # Calculate device resistance
    def UpdateR(self):
        self.R = self.ROn + (self.ROff - self.ROn) / (self.wMax - self.wMin) * (
            self.w - self.wMin
        )

    # Calculate the current for a given reference voltage.
    def GetCurrentOutput(self, V_ref, dt):
        I_out = np.zeros(np.size(V_ref))
        for i in range(np.size(V_ref)):
            # If above positive setting voltage
            if V_ref[i] > self.VSetp:
                self.dw_dt = self.kSetp * (V_ref[i] / self.VSetp - 1) ** self.alphaSetp
                self.w += dt * self.dw_dt
                if self.w > 1:
                    self.w = 1
                elif self.w < 0:
                    self.w = 0

                self.SetPState = 1
                self.SetNState = 0
                self.ResetPState = 0
                self.ResetNState = 0
            # If below negative setting voltage
            elif V_ref[i] < self.VSetn:
                self.dw_dt = self.kSetn * (V_ref[i] / self.VSetn - 1) ** self.alphaSetn
                self.w += dt * self.dw_dt
                if self.w > 1:
                    self.w = 1
                elif self.w < 0:
                    self.w = 0

                self.SetPState = 0
                self.SetNState = 1
                self.ResetPState = 0
                self.ResetNState = 0
            # If  being reset once in positive voltage region
            elif (0 < V_ref[i] < self.VResetp) & (
                (self.ResetPState == 1) | (self.SetPState == 1)
            ):
                self.dw_dt = (
                    self.kResetp * (V_ref[i] / self.VResetp) ** self.alphaResetp
                )  # Resetting mechanic is a little different.
                self.w += dt * self.dw_dt
                if self.w > 1:
                    self.w = 1
                elif self.w < 0:
                    self.w = 0

                self.SetPState = 0
                self.SetNState = 0
                self.ResetPState = 1
                self.ResetNState = 0
            # If being reset once in negative voltage region.
            elif (0 > V_ref[i] > self.VResetn) & (
                (self.ResetNState == 1 | self.SetNState == 1)
            ):
                self.dw_dt = (
                    self.kResetn * (V_ref[i] / self.VResetn) ** self.alphaResetn
                )
                self.w += dt * self.dw_dt
                if self.w > 1:
                    self.w = 1
                elif self.w < 0:
                    self.w = 0

                self.SetPState = 0
                self.SetNState = 0
                self.ResetPState = 0
                self.ResetNState = 1

            self.UpdateR()
            # Calculate current vector
            I_out[i] = V_ref[i] / self.R

        return I_out


# This function will calculate the RMS error from reference data to model data.
def RMSErr(trial, I_ref, V_ref, r_sq_prune_threshold):
    I_ref = I_ref.reshape(-1, 1)
    I_bar_sq = np.sum(np.square(I_ref))
    params = {}
    params["alphaSetp"] = trial.suggest_float("alphaSetp", -10, 10)
    params["alphaSetn"] = trial.suggest_float("alphaSetn", -10, 10)
    params["alphaResetp"] = trial.suggest_float("alphaResetp", 0, 10)
    params["alphaResetn"] = trial.suggest_float("alphaResetn", -10, 0)
    params["kSetp"] = trial.suggest_float("kSetp", 0, 1)
    params["kSetn"] = trial.suggest_float("kSetn", -1, 0)
    params["kResetp"] = trial.suggest_float("kResetp", 0, 10)
    params["kResetn"] = trial.suggest_float("kResetn", -10, 0)
    params["ROff"] = trial.suggest_int("ROff", 10000, 30000)
    params["ROn"] = trial.suggest_int("ROn", 1000, 10000)
    params["VSetp"] = trial.suggest_float("VSetp", 0, 10)
    params["VSetn"] = trial.suggest_float("VSetn", -10, 0)
    params["VResetp"] = trial.suggest_float("VResetp", 0, 100)
    params["VResetn"] = trial.suggest_float("VResetn", -100, 0)

    Model = VTEAMMod(0, **params)
    IVTEAM = Model.GetCurrentOutput(V_ref, 0.2).reshape(-1, 1)
    V_ref = V_ref.reshape(-1, 1)
    # Prune trails with a (near) linear I/V relationship
    linear_model = LinearRegression()
    linear_model.fit(V_ref, IVTEAM)
    r_sq = linear_model.score(V_ref, IVTEAM)
    if r_sq >= r_sq_prune_threshold:
        raise optuna.TrialPruned()

    return np.sqrt(np.sum(np.square(IVTEAM - I_ref)) / I_bar_sq)
