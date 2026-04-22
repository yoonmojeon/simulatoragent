# -*- coding: utf-8 -*-
"""
FMU Co-Simulation Engine
========================
OSP 없이 fmpy를 사용해 FMU를 직접 Python에서 실행하는 경량 코시뮬레이터.

핵심 설계:
  - FMI 2.0 CoSimulation FMU를 개별 로드/스텝
  - LLM이 FMU 파라미터를 직접 설정하고 코시뮬 실행
  - N회 반복 실험으로 통계적 신뢰도 확보
"""

import sys, os, shutil, time, json, math, contextlib, io
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# FMU가 출력하는 [OK] / 디버그 메시지를 억제하는 컨텍스트 매니저
class _SuppressFMULog(contextlib.AbstractContextManager):
    """FMU fmuLogger의 Python print() 출력을 /dev/null 로 리다이렉트."""
    def __enter__(self):
        self._real = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *_):
        sys.stdout = self._real
        return False

try:
    from fmpy import read_model_description, extract
    from fmpy.fmi2 import FMU2Slave
    from fmpy.fmi1 import FMU1Slave
except ImportError:
    raise ImportError("fmpy 가 설치되어 있지 않습니다: pip install fmpy")


# ── 전역 상수 ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DEMO_CASES = [
    # 0429/demo-cases (현재 프로젝트 기준 경로)
    os.path.join(os.path.dirname(_HERE), "demo-cases"),
    # 0429/energy_llm_agent/demo-cases (로컬 복사본이 있는 경우)
    os.path.join(_HERE, "demo-cases"),
    # 이전 구조 호환
    os.path.join(_HERE, "demo-cases-master"),
]

DEMO_CASES = next((p for p in _CANDIDATE_DEMO_CASES if os.path.exists(p)), _CANDIDATE_DEMO_CASES[0])


def _fmu_path(case: str, name: str) -> str:
    """demo-cases 안의 FMU 경로 반환."""
    candidates = [
        os.path.join(DEMO_CASES, case, "fmus", name + ".fmu"),
        os.path.join(DEMO_CASES, case, name + ".fmu"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"FMU not found: {case}/{name}")


# ══════════════════════════════════════════════════════════════════════════════
# FMUInstance  — 단일 FMU 래퍼
# ══════════════════════════════════════════════════════════════════════════════

class FMUInstance:
    """fmpy FMU2Slave 래퍼 — 로드/초기화/스텝/값조회를 편하게."""

    def __init__(self, name: str, fmu_path: str, step_size: float = 0.1):
        self.name       = name
        self.fmu_path   = fmu_path
        self.step_size  = step_size
        # validate=False: power_system 같은 non-standard Boolean variability 허용
        self._md        = read_model_description(fmu_path, validate=False)
        self._fmi_ver   = self._md.fmiVersion  # "1.0" or "2.0"
        self._vr        = {v.name: v.valueReference for v in self._md.modelVariables}
        self._var_info  = {v.name: v for v in self._md.modelVariables}
        self._unzip_dir = None
        self._fmu       = None
        self._t         = 0.0

    # ── 파라미터 / 변수 정보 ────────────────────────────────────────────────

    def list_variables(self, causality: str = None) -> list:
        result = []
        for v in self._md.modelVariables:
            if causality and v.causality != causality:
                continue
            el = (v.Real or v.Integer or v.Boolean or v.String or v.Enumeration
                  if hasattr(v, 'Real') else None)
            start = getattr(el, 'start', None) if el else None
            result.append({"name": v.name, "causality": v.causality,
                            "variability": v.variability, "start": start})
        return result

    def get_params(self) -> dict:
        return {v["name"]: v["start"] for v in self.list_variables("parameter")}

    # ── 수명 관리 ────────────────────────────────────────────────────────────

    def setup(self, start_time: float = 0.0, stop_time: float = 600.0,
              init_params: dict = None):
        """FMU 압축 해제 → 인스턴스화 → 초기화. FMI 1.0 / 2.0 모두 지원."""
        self._unzip_dir = extract(self.fmu_path)
        identifier = self._md.coSimulation.modelIdentifier

        if self._fmi_ver == "1.0":
            self._fmu = FMU1Slave(
                guid            = self._md.guid,
                unzipDirectory  = self._unzip_dir,
                modelIdentifier = identifier,
                instanceName    = self.name,
            )
            self._fmu.instantiate(loggingOn=False)
            if init_params:
                self._set_values(init_params)
            self._fmu.initialize(tStart=start_time, stopTime=stop_time)
        else:
            self._fmu = FMU2Slave(
                guid            = self._md.guid,
                unzipDirectory  = self._unzip_dir,
                modelIdentifier = identifier,
                instanceName    = self.name,
            )
            self._fmu.instantiate()
            self._fmu.setupExperiment(startTime=start_time, stopTime=stop_time)
            self._fmu.enterInitializationMode()
            if init_params:
                self._set_values(init_params)
            self._fmu.exitInitializationMode()

        self._t = start_time

    def teardown(self):
        if self._fmu:
            try:
                self._fmu.terminate()
                self._fmu.freeInstance()
            except Exception:
                pass
            self._fmu = None
        if self._unzip_dir and os.path.exists(self._unzip_dir):
            shutil.rmtree(self._unzip_dir, ignore_errors=True)
            self._unzip_dir = None

    # ── 값 읽기 / 쓰기 ───────────────────────────────────────────────────────

    def _set_values(self, values: dict):
        for name, val in values.items():
            if name not in self._vr:
                continue
            vr = self._vr[name]
            v_info = self._var_info[name]
            try:
                # type 판별: Real / Integer / Boolean
                if hasattr(v_info, 'Real') and v_info.Real is not None:
                    self._fmu.setReal([vr], [float(val)])
                elif hasattr(v_info, 'Integer') and v_info.Integer is not None:
                    self._fmu.setInteger([vr], [int(val)])
                elif hasattr(v_info, 'Boolean') and v_info.Boolean is not None:
                    self._fmu.setBoolean([vr], [bool(val)])
                else:
                    self._fmu.setReal([vr], [float(val)])
            except Exception:
                pass

    def set_inputs(self, values: dict):
        self._set_values(values)

    def get_real(self, name: str) -> float:
        return self._fmu.getReal([self._vr[name]])[0]

    def get_outputs(self, names: list = None) -> dict:
        result = {}
        targets = names or [v["name"] for v in self.list_variables("output")]
        for n in targets:
            if n in self._vr:
                result[n] = self._fmu.getReal([self._vr[n]])[0]
        return result

    # ── 시뮬레이션 스텝 ──────────────────────────────────────────────────────

    def step(self):
        if self._fmi_ver == "1.0":
            self._fmu.doStep(
                currentCommunicationPoint = self._t,
                communicationStepSize     = self.step_size,
                newStep                   = True,
            )
        else:
            self._fmu.doStep(
                currentCommunicationPoint = self._t,
                communicationStepSize     = self.step_size,
            )
        self._t += self.step_size

    @property
    def time(self):
        return self._t


# ══════════════════════════════════════════════════════════════════════════════
# CoSimRunner — 여러 FMU를 묶어 코시뮬 실행
# ══════════════════════════════════════════════════════════════════════════════

class CoSimRunner:
    """
    여러 FMUInstance를 연결해 코시뮬레이션을 실행.

    scenario_def 형식:
    {
        "name": "crane_lowering",
        "fmus": [
            {"name": "wave_model", "fmu": "linearized_wave_model", "case": "construction-vessel",
             "step_size": 0.1, "default_params": {"wave_height": 4.0, "peak_frequency": 0.8}},
            {"name": "winch",      "fmu": "winch",    "case": "construction-vessel",
             "step_size": 0.1, "default_params": {"K_p": 200.0, "K_d": 1000.0}},
        ],
        "connections": [
            {"from_fmu": "wave_model", "from_var": "first_order_disturbances.down",
             "to_fmu":   "winch",      "to_var":   "vessel_position.down"},
            ...
        ],
        "control": {
            "winch": {"load_depth_setpoint": 30.0}
        },
        "metric": {
            "depth_fmu": "winch", "depth_var": "load_depth", "target_depth": 30.0,
            "power_fmu": "winch", "power_var": "power_consumption", "power_limit_kw": 11900.0
        },
        "step_size": 0.1,
        "max_duration": 300.0,
        "depth_tolerance": 1.0
    }
    """

    def __init__(self, scenario_def: dict):
        self.sdef     = scenario_def
        self._fmus: dict[str, FMUInstance] = {}

    def run_trial(self, override_params: dict = None,
                  max_duration: float = None) -> dict:
        """
        한 번 코시뮬 실행 → metric dict 반환.
        override_params = {"winch": {"K_p": 400}, "wave_model": {"wave_height": 4.0}}
        """
        # FMU 내부 [OK] 로그를 억제 (fmpy fmuLogger → Python print)
        _orig_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return self._run_trial_inner(override_params=override_params,
                                         max_duration=max_duration)
        finally:
            sys.stdout = _orig_stdout

    def _run_trial_inner(self, override_params: dict = None,
                         max_duration: float = None) -> dict:
        override_params = override_params or {}
        max_dur = max_duration or self.sdef.get("max_duration", 300.0)
        step    = self.sdef.get("step_size", 0.1)

        # ── FMU 초기화 ────────────────────────────────────────────────────
        fmu_instances = {}
        try:
            for fspec in self.sdef["fmus"]:
                fname   = fspec["name"]
                fpath   = _fmu_path(fspec["case"], fspec["fmu"])
                ss      = fspec.get("step_size", step)
                inst    = FMUInstance(fname, fpath, ss)

                # 기본 파라미터 + override 병합
                init_p  = dict(fspec.get("default_params", {}))
                init_p.update(override_params.get(fname, {}))

                inst.setup(start_time=0.0, stop_time=max_dur + 1.0,
                           init_params=init_p)

                # 제어 입력 (setpoint 등)
                ctrl = self.sdef.get("control", {}).get(fname, {})
                if ctrl:
                    inst.set_inputs(ctrl)

                fmu_instances[fname] = inst

            # ── 코시뮬 루프 ───────────────────────────────────────────────
            metric   = self.sdef["metric"]
            d_fmu    = metric["depth_fmu"]
            d_var    = metric["depth_var"]
            p_fmu    = metric["power_fmu"]
            p_var    = metric["power_var"]
            # vessel 등 전력[W]: power_limit_kw × 1000
            # LARS 등 비전력 신호(rad/s 등): power_limit_raw 로 직접 한계값 지정
            if metric.get("power_limit_raw") is not None:
                p_limit = float(metric["power_limit_raw"])
            else:
                p_limit = metric.get("power_limit_kw", 11900.0) * 1000.0  # kW → W
            tgt      = metric.get("target_depth", 30.0)
            tol      = self.sdef.get("depth_tolerance", 2.0)

            violations = 0
            steps      = 0
            t          = 0.0
            power_hist = []
            depth_hist = []
            reached    = False
            time_to_target_s = None
            # target_depth=None → 전체 시간 실행 (truck/진동 시나리오용)
            check_depth = (tgt is not None)
            equil       = metric.get("equilibrium", tgt or 0.0)

            while t < max_dur:
                # 1) 각 FMU 스텝
                for inst in fmu_instances.values():
                    inst.step()
                t += step
                steps += 1

                # 2) FMU 간 신호 전달
                # 핵심: 먼저 모든 source 값을 읽어 버퍼에 저장(read-phase),
                #       그 다음 버퍼에서 target에 쓰기(write-phase).
                # → "inputs changed, step not called before get" 오류 방지.
                read_buf: dict[tuple, float] = {}

                # 2a) scalar connections - read phase
                for conn in self.sdef.get("connections", []):
                    key = (conn["from_fmu"], conn["from_var"])
                    if key not in read_buf:
                        try:
                            read_buf[key] = fmu_instances[conn["from_fmu"]].get_real(conn["from_var"])
                        except Exception:
                            read_buf[key] = 0.0

                # sum_connections - read phase
                for sc in self.sdef.get("sum_connections", []):
                    for side in ("from_a", "from_b"):
                        key = (sc[side]["fmu"], sc[side]["var"])
                        if key not in read_buf:
                            try:
                                read_buf[key] = fmu_instances[sc[side]["fmu"]].get_real(sc[side]["var"])
                            except Exception:
                                read_buf[key] = 0.0

                # metric 변수도 read_buf에 미리 추가 (write phase 이전에 읽어야 함)
                for _key in [(d_fmu, d_var), (p_fmu, p_var)]:
                    if _key not in read_buf:
                        try:
                            read_buf[_key] = fmu_instances[_key[0]].get_real(_key[1])
                        except Exception:
                            read_buf[_key] = 0.0

                # 2b) write phase - scalar connections
                for conn in self.sdef.get("connections", []):
                    key = (conn["from_fmu"], conn["from_var"])
                    val = read_buf.get(key, 0.0)
                    try:
                        fmu_instances[conn["to_fmu"]].set_inputs({conn["to_var"]: val})
                    except Exception:
                        pass

                # write phase - VectorSum (A + B → C)
                for sc in self.sdef.get("sum_connections", []):
                    val_a = read_buf.get((sc["from_a"]["fmu"], sc["from_a"]["var"]), 0.0)
                    val_b = read_buf.get((sc["from_b"]["fmu"], sc["from_b"]["var"]), 0.0)
                    try:
                        fmu_instances[sc["to_fmu"]].set_inputs({sc["to_var"]: val_a + val_b})
                    except Exception:
                        pass

                # 3) 메트릭 수집 (이미 read_buf에 있음)
                depth = read_buf.get((d_fmu, d_var), 0.0)
                power = read_buf.get((p_fmu, p_var), 0.0)

                power_hist.append(abs(power))
                depth_hist.append(depth)

                # 평형점이 설정된 경우(truck/DP): 편차 기준으로 위반 판단
                violation_val = abs(power - equil) if "equilibrium" in metric else abs(power)
                if violation_val > p_limit:
                    violations += 1

                # 4) 종료 조건: 목표 도달 (check_depth=False면 전 구간 실행)
                if check_depth and t > 1.0 and abs(depth - tgt) <= tol:
                    reached = True
                    time_to_target_s = t
                    break

            if not check_depth:
                # 마지막 20% 구간 안정화 판단.
                # success_mode:
                #   "range"    (기본) — peak-to-peak 진동 범위 < tol (트럭 충격 흡수용)
                #   "abs_mean" — 평균 절대 편차 < tol (DP 위치 유지용)
                last_n = max(1, steps // 5)
                seg = depth_hist[-last_n:]
                success_mode = metric.get("success_mode", "range")
                if seg:
                    if success_mode == "abs_mean":
                        deviation = sum(abs(d - equil) for d in seg) / len(seg)
                        reached = deviation < tol
                    else:   # "range" default
                        osc_range = max(seg) - min(seg)
                        reached = osc_range < tol
                else:
                    reached = True

            elapsed = t
            mean_power_kw = (sum(power_hist) / len(power_hist) / 1000) if power_hist else 0
            max_power_kw  = (max(power_hist) / 1000) if power_hist else 0
            vps           = violations / steps if steps > 0 else 0
            if depth_hist and check_depth:
                overshoot = max(abs(d - tgt) for d in depth_hist)
            elif depth_hist:
                overshoot = max(abs(d - equil) for d in depth_hist)
            else:
                overshoot = 0.0
            energy_j = sum(power_hist) * step if power_hist else 0.0

            return {
                "success":       reached,
                "elapsed_s":     round(elapsed, 2),
                "time_to_target_s": round(time_to_target_s, 2) if time_to_target_s is not None else None,
                "steps":         steps,
                "violations":    violations,
                "violation_count": violations,
                "viol_per_step": round(vps, 4),
                "mean_power_kw": round(mean_power_kw, 1),
                "max_power_kw":  round(max_power_kw, 1),
                "max_signal_raw": round(max(power_hist), 4) if power_hist else 0.0,
                "energy_j":      round(energy_j, 3),
                "overshoot_abs": round(overshoot, 4),
                "final_depth":   round(depth_hist[-1], 3) if depth_hist else 0,
                "target_depth":  tgt,
                "params_used":   override_params,
            }

        finally:
            for inst in fmu_instances.values():
                inst.teardown()

    def run_n_trials(self, override_params: dict = None,
                     n: int = 3, max_duration: float = None) -> dict:
        """N회 반복 실험 → 평균/분산 포함 결과."""
        trials = []
        for i in range(n):
            r = self.run_trial(override_params, max_duration)
            trials.append(r)

        vps_list     = [t["viol_per_step"] for t in trials]
        time_list    = [t["elapsed_s"]     for t in trials]
        mean_vps     = sum(vps_list) / n
        std_vps      = math.sqrt(sum((x - mean_vps)**2 for x in vps_list) / n)
        success_rate = sum(1 for t in trials if t["success"]) / n

        # ── Risk Score = mean_vps + std_vps + (1 - success_rate) * failure_penalty
        # failure_penalty: 목표 미달성에 대한 패널티 (시나리오별 설정).
        #   크레인: 0.5 — 목표 수심 미도달은 작업 실패이므로 강한 패널티
        #   트럭/DP: 0.2 — 안정화 미달성에 대한 중간 패널티
        # 이 항은 Safety Paralysis(위반=0이지만 목표 미도달) 를 올바르게 페널티화함.
        failure_penalty = self.sdef.get("metric", {}).get("failure_penalty", 0.0)
        risk = round(mean_vps + std_vps + (1.0 - success_rate) * failure_penalty, 4)

        # 에너지 지표 집계 (top-level 노출)
        pwr_list    = [t.get("mean_power_kw", 0) for t in trials]
        energy_list = [t.get("energy_j", 0)      for t in trials]
        mean_power_kw = round(sum(pwr_list) / n, 2)
        mean_energy_j = round(sum(energy_list) / n, 1)

        return {
            "n_trials":        n,
            "params":          override_params or {},
            "mean_vps":        round(mean_vps, 4),
            "std_vps":         round(std_vps, 4),
            "risk_score":      risk,
            "mean_time_s":     round(sum(time_list) / n, 1),
            "success_rate":    success_rate,
            "failure_penalty": failure_penalty,
            "mean_power_kw":   mean_power_kw,
            "energy_j":        mean_energy_j,
            "trials":          trials,
        }
