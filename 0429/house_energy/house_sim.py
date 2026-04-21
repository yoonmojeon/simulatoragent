# -*- coding: utf-8 -*-
"""
House Energy Management Co-Simulation Engine
=============================================
7-FMU co-simulation (OSP demo-cases/house):
  Clock → TempController → Room1, Room2
  Room1/2 ↔ InnerWall, OuterWall1/2

단위: FMU는 모두 Celsius [°C] 사용
  T_outside = 5.0°C (외기)
  COMFORT = 18~23°C
  step_size = 60s (communication), stop = 3600s (1 hour, 속도/가능성 고려)

Risk = energy_ratio + comfort_violation + (1−sr)×0.5
  energy_ratio      = mean_heater_power / (2 × MAX_W)
  comfort_violation = 두 방 중 쾌적 범위 이탈 스텝 비율
"""
from __future__ import annotations

import io
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from fmpy import read_model_description, extract
    from fmpy.fmi2 import FMU2Slave
    from fmpy.fmi1 import FMU1Slave
except ImportError:
    raise ImportError("fmpy 가 설치되어 있지 않습니다: pip install fmpy")

_DEMO = Path(__file__).resolve().parents[2] / "0422" / "simulatoragent" / "demo-cases-master" / "house"

OUTSIDE_TEMP    = 5.0       # °C
COMFORT_LOW     = 18.0      # °C
COMFORT_HIGH    = 24.0      # °C
MAX_HEATER_W    = 20.0      # W per room (conservative; OvenHeatTransfer range 1-20W)
STOP_TIME       = 1800.0    # 30 min simulation [s]
STEP_SIZE       = 1.0       # communication step [s] — stable with small conductances


def _fmu_path(name: str) -> str:
    p = _DEMO / f"{name}.fmu"
    if not p.exists():
        raise FileNotFoundError(f"FMU not found: {p}")
    return str(p)


# ── FMU 래퍼 ────────────────────────────────────────────────────────────────
class _FMU:
    def __init__(self, name: str, path: str, step: float = STEP_SIZE):
        self.name = name
        self._path = path
        self._step = step
        self._md = read_model_description(path, validate=False)
        self._vr = {v.name: v.valueReference for v in self._md.modelVariables}
        self._vi = {v.name: v for v in self._md.modelVariables}
        self._unzip: str | None = None
        self._fmu = None
        self._t = 0.0

    def setup(self, stop: float, init_params: dict | None = None):
        self._unzip = extract(self._path)
        ident = self._md.coSimulation.modelIdentifier
        ver = self._md.fmiVersion
        if ver == "1.0":
            self._fmu = FMU1Slave(
                guid=self._md.guid, unzipDirectory=self._unzip,
                modelIdentifier=ident, instanceName=self.name,
            )
            self._fmu.instantiate(loggingOn=False)
            if init_params:
                self._set(init_params)
            self._fmu.initialize(tStart=0.0, stopTime=stop)
        else:
            self._fmu = FMU2Slave(
                guid=self._md.guid, unzipDirectory=self._unzip,
                modelIdentifier=ident, instanceName=self.name,
            )
            self._fmu.instantiate()
            self._fmu.setupExperiment(startTime=0.0, stopTime=stop)
            self._fmu.enterInitializationMode()
            if init_params:
                self._set(init_params)
            self._fmu.exitInitializationMode()
        self._t = 0.0

    def teardown(self):
        if self._fmu:
            try:
                self._fmu.terminate(); self._fmu.freeInstance()
            except Exception:
                pass
            self._fmu = None
        if self._unzip and os.path.exists(self._unzip):
            shutil.rmtree(self._unzip, ignore_errors=True)

    def _set(self, vals: dict):
        for name, val in vals.items():
            if name not in self._vr:
                continue
            vr = self._vr[name]
            vi = self._vi[name]
            try:
                if hasattr(vi, "Real") and vi.Real is not None:
                    self._fmu.setReal([vr], [float(val)])
                elif hasattr(vi, "Integer") and vi.Integer is not None:
                    self._fmu.setInteger([vr], [int(val)])
                elif hasattr(vi, "Boolean") and vi.Boolean is not None:
                    self._fmu.setBoolean([vr], [bool(val)])
                else:
                    self._fmu.setReal([vr], [float(val)])
            except Exception:
                pass

    def set(self, vals: dict):
        self._set(vals)

    def get(self, name: str) -> float:
        if name not in self._vr:
            return 0.0
        try:
            return self._fmu.getReal([self._vr[name]])[0]
        except Exception:
            return 0.0

    def step(self) -> None:
        try:
            self._fmu.doStep(self._t, self._step, True)
        except Exception:
            pass
        self._t += self._step


# ══════════════════════════════════════════════════════════════════════════════
class HouseCoSimRunner:
    """
    7-FMU House Energy Management Co-simulation.

    override_params (모두 Celsius 또는 물리 단위):
      {
        "controller": {
            "OvenHeatTransfer": 1500.0,   # W/K
            "T_heatStart":      18.0,     # °C  (heating ON below this)
            "T_heatStop":       22.0,     # °C  (heating OFF above this)
            "transferTime":     900.0,    # s
        },
        "room1":      {"Tinit_Room1": 15.0},   # °C initial temp
        "room2":      {"Tinit_Room2": 15.0},
        "inner_wall": {"k": 100.0},
        "outer_wall1":{"k_outsidewall": 40.0},
        "outer_wall2":{"k_outsidewall": 40.0},
      }
    """

    def run_trial(self, override: dict | None = None) -> dict:
        override = override or {}
        orig = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return self._run(override)
        finally:
            sys.stdout = orig

    def _run(self, override: dict) -> dict:
        stop = STOP_TIME
        step = STEP_SIZE
        start_wall = time.time()

        fmus = {
            "clock":       _FMU("clock",       _fmu_path("Clock"),        step),
            "controller":  _FMU("controller",  _fmu_path("TempController"), step),
            "room1":       _FMU("room1",       _fmu_path("Room1"),        step),
            "room2":       _FMU("room2",       _fmu_path("Room2"),        step),
            "inner_wall":  _FMU("inner_wall",  _fmu_path("InnerWall"),    step),
            "outer_wall1": _FMU("outer_wall1", _fmu_path("OuterWall1"),   step),
            "outer_wall2": _FMU("outer_wall2", _fmu_path("OuterWall2"),   step),
        }

        # 단위 스케일: OvenHeatTransfer는 [W] 단위 on/off 출력
        # k_outsidewall/k 는 [W/K] (작게 설정해 step=1s 안정성 보장, ~0.001~0.1 W/K)
        defaults = {
            "controller":  {"OvenHeatTransfer": 8.0,  "T_heatStart": 17.0,
                            "T_heatStop": 22.0, "transferTime": 20.0},
            "room1":       {"Tinit_Room1": 12.0},
            "room2":       {"Tinit_Room2": 12.0},
            "inner_wall":  {"k": 0.005},
            "outer_wall1": {"k_outsidewall": 0.003, "T_outside": OUTSIDE_TEMP},
            "outer_wall2": {"k_outsidewall": 0.003, "T_outside": OUTSIDE_TEMP},
        }

        init_params: dict[str, dict] = {}
        for key, defs in defaults.items():
            merged = dict(defs)
            merged.update(override.get(key, {}))
            init_params[key] = merged

        try:
            for key, f in fmus.items():
                f.setup(stop=stop + step, init_params=init_params.get(key, {}))

            comfort_ok  = 0
            total_steps = 0
            heater_hist: list[float] = []
            t1_hist: list[float] = []
            t2_hist: list[float] = []

            # 초기 출력 읽기 (step 전 모든 FMU의 현재 상태)
            h_r1 = h_r2 = 0.0
            h_inner = h_out1 = h_out2 = 0.0

            n_steps = int(stop / step)
            for _ in range(n_steps):
                # ── 1. 현재 출력 읽기 (Jacobi 병렬 커플링) ───────────────
                t_clock = fmus["clock"].get("Clock")
                t_room1 = fmus["room1"].get("T_room")
                t_room2 = fmus["room2"].get("T_room")
                # NaN/inf 방어
                if not (-200 < t_room1 < 200): t_room1 = 15.0
                if not (-200 < t_room2 < 200): t_room2 = 15.0

                h_inner = fmus["inner_wall"].get("h_wall")
                h_out1  = fmus["outer_wall1"].get("h_wall")
                h_out2  = fmus["outer_wall2"].get("h_wall")
                h_r1    = fmus["controller"].get("h_room1")
                h_r2    = fmus["controller"].get("h_room2")

                # ── 2. 모든 FMU 입력 설정 (step 전에 한꺼번에) ───────────
                fmus["controller"].set({
                    "T_room1": t_room1, "T_room2": t_room2, "T_clock": t_clock,
                })
                fmus["inner_wall"].set({"T_room1": t_room1, "T_room2": t_room2})
                fmus["outer_wall1"].set({"T_room1": t_room1})
                fmus["outer_wall2"].set({"T_room2": t_room2})
                fmus["room1"].set({
                    "h_powerHeater": h_r1,
                    "h_InnerWall":   h_inner,
                    "h_OuterWall":   h_out1,
                })
                fmus["room2"].set({
                    "h_powerHeater": h_r2,
                    "h_InnerWall":   h_inner,
                    "h_OuterWall":   h_out2,
                })

                # ── 3. 모든 FMU 스텝 ──────────────────────────────────────
                for f in fmus.values():
                    f.step()

                # ── 4. 지표 기록 ──────────────────────────────────────────
                heater_hist.append(abs(h_r1) + abs(h_r2))
                t1_hist.append(t_room1)
                t2_hist.append(t_room2)
                total_steps += 1

                if (COMFORT_LOW <= t_room1 <= COMFORT_HIGH and
                        COMFORT_LOW <= t_room2 <= COMFORT_HIGH):
                    comfort_ok += 1

        finally:
            for f in fmus.values():
                f.teardown()

        if not heater_hist:
            return {"risk_score": 9999.0, "error": "no_steps"}

        mean_power     = sum(heater_hist) / len(heater_hist)
        energy_ratio   = min(mean_power / (2 * MAX_HEATER_W), 1.0)
        comfort_rate   = comfort_ok / max(total_steps, 1)
        comfort_viol   = 1.0 - comfort_rate
        # success: comfort_rate >= 0.5 (두 방 모두 18-24°C 절반 이상 유지)
        success        = 1.0 if comfort_rate >= 0.50 else 0.0
        fp             = 0.3
        risk           = round(energy_ratio + comfort_viol + (1.0 - success) * fp, 4)

        mean_t1 = sum(t1_hist) / len(t1_hist) if t1_hist else float("nan")
        mean_t2 = sum(t2_hist) / len(t2_hist) if t2_hist else float("nan")
        # NaN 방어
        if math.isnan(mean_t1): mean_t1 = 0.0
        if math.isnan(mean_t2): mean_t2 = 0.0

        return {
            "risk_score":         risk,
            "energy_ratio":       round(energy_ratio, 4),
            "comfort_rate":       round(comfort_rate, 4),
            "comfort_violation":  round(comfort_viol, 4),
            "success_rate":       success,
            "mean_power_w":       round(mean_power, 1),
            "mean_temp_room1_c":  round(mean_t1, 2),
            "mean_temp_room2_c":  round(mean_t2, 2),
            "elapsed_s":          round(time.time() - start_wall, 2),
            "n_steps":            total_steps,
        }

    def run_n_trials(self, override_params: dict | None = None, n: int = 3) -> dict:
        trials = []
        for _ in range(n):
            trials.append(self.run_trial(override_params))

        risks  = [t["risk_score"]  for t in trials]
        mean_r = sum(risks) / n
        std_r  = math.sqrt(sum((x - mean_r) ** 2 for x in risks) / n)
        sr     = sum(1 for t in trials if t.get("success_rate", 0) >= 0.5) / n
        risk   = round(mean_r + std_r + (1.0 - sr) * 0.3, 4)

        pwr = [t.get("mean_power_w", 0) for t in trials]
        er  = [t.get("energy_ratio",  0) for t in trials]
        cr  = [t.get("comfort_rate",  0) for t in trials]

        return {
            "n_trials":          n,
            "params":            override_params or {},
            "risk_score":        risk,
            "mean_vps":          round(mean_r, 4),
            "std_vps":           round(std_r, 4),
            "success_rate":      sr,
            "mean_power_kw":     round(sum(pwr) / n / 1000, 3),
            "energy_j":          round(sum(pwr) / n * STOP_TIME, 0),
            "mean_energy_ratio": round(sum(er) / n, 4),
            "mean_comfort_rate": round(sum(cr) / n, 4),
            "failure_penalty":   0.5,
            "trials":            trials,
        }
