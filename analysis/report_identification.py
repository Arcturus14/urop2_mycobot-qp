"""Analyze real step experiments and write an evidence-linked Korean report."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .step_models import errors, fit_model, predict

MODELS = ["second_order", "first_order", "velocity_limited", "deadzone_velocity"]
LABELS = {"second_order": "2차 + 지연", "first_order": "1차 + 지연",
          "velocity_limited": "속도 제한 + 지연", "deadzone_velocity": "데드밴드·속도 제한 + 지연"}


def text_number(value, digits=3):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def metrics(event):
    t, y, y0 = event["t"], event["y"], event["y0"]
    post = t >= 0
    tp, yp = t[post], y[post]
    final = float(np.median(yp[tp >= tp[-1] - 0.35]))
    amplitude = final - y0
    result = {"file": event["file"], "joint": event["joint"], "repeat": event["repeat"],
              "step": event["step"], "split": event["split"], "command_delta_deg": event["delta"],
              "initial_error_deg": event["error"], "output_delta_deg": amplitude,
              "final_angle_deg": final, "steady_error_deg": event["to_deg"] - final,
              "observed_moving": abs(amplitude) >= 0.15,
              "onset_s": None, "rise_10_90_s": None, "settle_observed_final_s": None,
              "overshoot_pct": None}
    if abs(amplitude) >= 0.15:
        progress = np.sign(amplitude) * (yp - y0)
        indices = np.flatnonzero(progress >= max(0.09, 3 * event["pre_std"]))
        if len(indices):
            result["onset_s"] = float(tp[indices[0]])
        crossing = []
        for fraction in (0.1, 0.9):
            indices = np.flatnonzero(progress >= fraction * abs(amplitude))
            crossing.append(float(tp[indices[0]]) if len(indices) else None)
        if None not in crossing:
            result["rise_10_90_s"] = crossing[1] - crossing[0]
        band = max(0.1, 0.02 * abs(amplitude))
        outside = np.flatnonzero(np.abs(yp - final) > band)
        settle_index = int(outside[-1] + 1) if len(outside) else 0
        if settle_index < len(tp) and tp[-1] - tp[settle_index] >= 0.3:
            result["settle_observed_final_s"] = float(tp[settle_index])
        result["overshoot_pct"] = float(max(0.0, np.max(progress) / abs(amplitude) - 1.0) * 100)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download", nargs="+", help="One or more directories returned by fetch")
    parser.add_argument("--baseline-download", help="Additional stationary/timing download")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources = [Path(path).resolve() for path in args.download]
    source, output = sources[0], Path(args.output).resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs, transfers = [], []
    for directory in sources:
        candidates = sorted(directory.rglob("*_identification.json"))
        if len(candidates) != 1:
            parser.error("Exactly one identification metadata file expected per download")
        path = candidates[0]
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["backend"] != "real":
            parser.error("This report requires actual robot data, not mock data")
        transfer = json.loads((directory / "transfer.json").read_text(encoding="utf-8"))
        if not transfer.get("verified"):
            parser.error("Verified transfer manifest required")
        runs.append((path, run))
        transfers.append(transfer)
    metadata_path = runs[0][0]
    metadata = copy.deepcopy(runs[0][1])
    metadata["events"] = [{**item, "source_metadata_path": path.as_posix()}
                          for path, run in runs for item in run["events"]]
    metadata["read_failures"] = [item for _, run in runs for item in run.get("read_failures", [])]
    metadata["utc_start"] = min(run["utc_start"] for _, run in runs)
    metadata["utc_end"] = max(run["utc_end"] for _, run in runs)
    completed = [item for item in metadata["events"] if item["status"] == "complete"]
    metadata["status"] = "complete (combined)" if len(completed) == 96 else "partial"
    transfer = {"source_release": "; ".join(item["source_release"] for item in transfers)}
    events = []
    coupling = np.zeros((6, 6))
    for item in metadata["events"]:
        if item["status"] != "complete":
            continue
        path = Path(item["source_metadata_path"]).parent / item["file"]
        frame = pd.read_csv(path)
        q = frame[[f"q_actual_j{i}" for i in range(1, 7)]].to_numpy(float)
        if not np.isfinite(q).all() or not (np.diff(frame["t_s"]) > 0).all():
            raise ValueError("Invalid samples: " + str(path))
        joint = item["joint"]
        t = frame["t_s"].to_numpy(float) - item["command_start_s"]
        pre = frame["phase"].to_numpy() == "pre_step"
        y = q[:, joint - 1]
        y0 = float(np.median(y[pre]))
        event = {**item, "t": t, "y": y, "y0": y0,
                 "delta": item["to_deg"] - item["from_deg"],
                 "error": item["to_deg"] - y0, "frame": frame,
                 "pre_std": float(np.std(y[pre], ddof=1)), "path": path}
        other_drift = np.max(np.abs(q - np.median(q[pre], axis=0)), axis=0)
        other_drift[joint - 1] = 0
        coupling[joint - 1] = np.maximum(coupling[joint - 1], other_drift)
        events.append(event)
    if not events:
        parser.error("No completed events to identify; inspect aborted-run metadata")
    table = pd.DataFrame([metrics(event) for event in events])
    table.to_csv(output / "step_metrics.csv", index=False)
    results = {}
    for joint in sorted({event["joint"] for event in events}):
        selected = [event for event in events if event["joint"] == joint]
        # Before the first command of a joint, the firmware's previous target is
        # not known (especially after a stopped pilot). Keep that record in the
        # report but exclude it from fitting every candidate consistently.
        train = [event for event in selected if event["split"] == "train" and event["step"] != 1]
        validation = [event for event in selected if event["split"] == "validation"]
        if not train or not validation:
            continue
        models = {}
        for model in MODELS:
            fit = fit_model(model, train)
            residual = errors(model, fit["parameter_vector"], validation)
            fit.update(validation_rmse_deg=float(np.sqrt(np.mean(residual**2))),
                       validation_mae_deg=float(np.mean(np.abs(residual))),
                       validation_max_abs_error_deg=float(np.max(np.abs(residual))))
            models[model] = fit
        best = min(models, key=lambda model: models[model]["validation_rmse_deg"])
        results[f"J{joint}"] = {"models": models, "lowest_validation_rmse_model": best,
                                 "train_events": len(train), "validation_events": len(validation)}
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, step in zip(axes.flat, (1, 3, 5, 7)):
            event = next(event for event in validation if event["step"] == step)
            t, y0 = event["t"], event["y0"]
            ax.step([-0.5, 0.0, 2.1], [event["from_deg"] - y0, event["to_deg"] - y0,
                                      event["to_deg"] - y0], where="post", linestyle="--", color="gray", label="command")
            ax.plot(t, event["y"] - y0, ".-", markersize=3, label="actual")
            for model, color in (("second_order", "tab:green"), (best, "tab:red")):
                delta = event["error"] if model == "deadzone_velocity" else event["delta"]
                prediction = predict(model, models[model]["parameter_vector"], t, delta, y0)
                ax.plot(t, prediction - y0, linewidth=1.2, color=color, label=model)
            ax.set(title=f"J{joint}, validation step {step}, command delta {event['delta']:+.1f} deg",
                   xlabel="Time from command start (s)", ylabel="Change from measured start (deg)", xlim=(-0.15, 2.05))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / f"J{joint}_validation.png", dpi=160)
        plt.close(fig)
        print(f"J{joint}: best={best}, validation RMSE={models[best]['validation_rmse_deg']:.4f} deg", flush=True)
    (output / "model_comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(coupling, index=[f"command_J{i}" for i in range(1, 7)],
                 columns=[f"measured_J{i}" for i in range(1, 7)]).to_csv(output / "other_joint_motion.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ax.imshow(coupling, cmap="YlOrRd", vmin=0, vmax=max(0.1, float(coupling.max())))
    ax.set_xticks(range(6), [f"J{i}" for i in range(1, 7)])
    ax.set_yticks(range(6), [f"J{i}" for i in range(1, 7)])
    ax.set(xlabel="Measured joint", ylabel="Commanded joint")
    for row in range(6):
        for col in range(6):
            ax.text(col, row, "-" if row == col else f"{coupling[row, col]:.2f}", ha="center", va="center")
    fig.colorbar(colors, ax=ax, label="Max change of other joint (deg)")
    ax.set_title("Single-joint tests: observed motion of other joints")
    fig.tight_layout()
    fig.savefig(figures / "other_joint_motion.png", dpi=160)
    plt.close(fig)
    baseline_rows = []
    if args.baseline_download:
        for path in sorted(Path(args.baseline_download).resolve().rglob("*.csv")):
            frame = pd.read_csv(path)
            intervals = np.diff(frame["t_s"].to_numpy(float)) * 1000
            baseline_rows.append({"file": path.name, "samples": len(frame),
                "interval_mean_ms": float(np.mean(intervals)),
                "interval_std_ms": float(np.std(intervals, ddof=1)),
                "interval_p99_ms": float(np.percentile(intervals, 99)),
                "interval_max_ms": float(np.max(intervals)),
                "all_joints_constant": all(frame[f"q_actual_j{i}"].nunique() == 1 for i in range(1, 7))})
    (output / "baseline_timing.json").write_text(json.dumps(baseline_rows, indent=2), encoding="utf-8")
    start = datetime.fromisoformat(metadata["utc_start"]).astimezone(ZoneInfo("Asia/Seoul"))
    end = datetime.fromisoformat(metadata["utc_end"]).astimezone(ZoneInfo("Asia/Seoul"))
    link = lambda path: f"<{Path(path).resolve().as_posix()}>"
    lines = ["# myCobot 280 JN5 실측 및 모델 식별 보고서", "",
        f"실험 시각: {start:%Y-%m-%d %H:%M:%S}–{end:%H:%M:%S} (한국시간)", "",
        "## 1. 이번 실험으로 확인한 것", "",
        f"Jetson에서 명령 전송과 관절각 기록을 실행하고, 노트북에서 분석했다. 완료된 step 기록은 **{len(events)}개**다. "
        f"실험 상태는 `{metadata['status']}`이며, 유효 관절각 샘플은 총 **{sum(len(event['frame']) for event in events):,}개**다. "
        "모델 추정에는 첫 번째 반복, 검증에는 두 번째 반복을 사용했다.", "",
        "같은 명령을 반복해도 이전 동작에서 남은 위치 오차 때문에 시작 실측각은 달라질 수 있다. "
        "각 step의 명령 전 실측각을 별도로 읽어 이를 분석에 반영했다. 각 관절의 첫 명령은 "
        "직전 펌웨어 목표값을 확인할 수 없어 모든 후보의 fitting에서 제외했다. 따라서 관절당 학습 7개, 검증 8개다.", "",
        "| 관절 | 관측된 이동 / step 수 | 정상상태 절대오차 중앙값 (°) | 2차 모델 검증 RMSE (°) | 최저 RMSE 후보 | 해당 RMSE (°) |",
        "|---|---:|---:|---:|---|---:|"]
    for key, result in results.items():
        joint = int(key[1:])
        rows = table[table.joint == joint]
        best = result["lowest_validation_rmse_model"]
        models = result["models"]
        lines.append(f"| {key} | {int(rows.observed_moving.sum())} / {len(rows)} | {rows.steady_error_deg.abs().median():.3f} | {models['second_order']['validation_rmse_deg']:.3f} | {LABELS[best]} | {models[best]['validation_rmse_deg']:.3f} |")
    lines += ["", "RMSE는 검증 데이터의 명령 이후 모든 샘플에 대한 예측 오차의 제곱평균제곱근이다. "
              "작을수록 관측값에 가깝다. ‘관측된 이동’은 최종 위치 변화가 0.15° 이상인 경우이며, "
              "움직이지 않은 명령도 삭제하지 않고 모델 평가에 포함했다.", "",
              "표의 최저 RMSE는 이번 후보 중 상대 비교 결과다. 한 자세·한 speed·작은 각도에서 얻은 값이므로 "
              "6축 전체 작업영역에서 검증된 MPPI 모델로 해석하면 안 된다. 후보 선택에도 검증 반복을 사용했으므로 "
              "최종 모델에는 세 번째 독립 실험이 필요하다.", "",
              "**현재 판단:** 작은 위치 명령에서는 지연과 응답 속도뿐 아니라 목표 근처의 유효 데드밴드와 "
              "방향·이력에 따른 위치 오차를 반영해야 한다. 6개 독립 2차 모델을 그대로 MPPI에 채택할 근거는 "
              "이번 데이터에서 충분하지 않다. 특히 다른 관절의 관측 변화가 1° 이상인 경우가 있어 결합 검증을 우선해야 한다.", "",
              "## 2. 실험 환경과 데이터 신뢰성", "",
              f"- 로봇 실행 호스트: `{metadata['host']}`, Python `{metadata['python'].split()[0]}`, pymycobot `{metadata['pymycobot']}`.",
              f"- 시리얼: `{metadata['port']}`, `{metadata['baud']}` baud.",
              f"- 고정 기준 자세: `{metadata.get('reference_angles_deg', metadata['initial_angles_deg'])}` °.",
              f"- 실제 실험 시작각: `{metadata['initial_angles_deg']}` °.",
              "- 한 번에 한 관절만 위치 명령을 전송했다. 전체 영점 이동, 전원 재설정, 서보 해제는 실행하지 않았다.",
              f"- speed={metadata['settings']['speed']}, 기준 자세의 ±1.5°·±3° 및 기준 복귀, 2회 반복.",
              "- step 전 0.5초, 명령 후 약 2초 기록. 명령 후 50 ms는 통신을 쉬고 이후 읽기를 수행했다.",
              f"- 분석 대상 수집 묶음의 읽기 실패: **{len(metadata.get('read_failures', []))}회**. 누락된 값을 보간하거나 정상 샘플로 대체하지 않았다.",
              f"- [전송 검증 기록]({link(source / 'transfer.json')}): 원격·로컬 SHA-256 일치 확인.",
              f"- [실험 메타데이터]({link(metadata_path)}): 명령 시각, 실험 순서, 초기 자세, 실패·종료 상태.", "",
              "각 시각은 Jetson의 monotonic clock으로 기록했다. `get_angles()` 전후 시각의 중간을 샘플 시각으로 "
              "사용했다. 센서 내부 취득 시각은 알 수 없으므로 센서값의 실제 나이와 일치한다고 보장할 수 없다.", "",
              "### 수집 묶음과 중단 이력", "",
              "| 수집 ID | 완료 명령 기록 | 원본 종료 상태 | 종료 원인 |",
              "|---|---:|---|---|"]
    for path, run in runs:
        lines.append(f"| [{run['run_id']}]({link(path)}) | {sum(item['status'] == 'complete' for item in run['events'])} | {run['status']} | {run.get('error', '정상 종료')} |")
    lines += ["", "각 수집 묶음의 원본 메타데이터와 전송 검증 기록은 별도로 보존했다. "
              "앞선 사전 시도에서는 목표 오차 한도, 단발 통신 응답 누락, 감시 기준 때문에 실험이 중단됐다. "
              "J1 완료 후 J2로 넘어갈 때 원래 기준 자세와 비교하던 검사를 실제 유지 자세 기준으로 수정했다. "
              "J2 중 J4의 약 0.79° 변화가 관측된 뒤에는 현장의 이상 없음 확인을 받고 비명령 관절 감시 한도만 "
              "0.75°에서 1.5°로 조정했다. 명령 크기 ±3°와 speed=20은 유지했다. 중단된 미완료 기록은 모델 추정에서 제외했다.", "",
              "각 관절에는 16개 명령을 보냈지만 첫 번째 명령 직전의 펌웨어 목표가 불명확하므로 "
              "이 첫 기록의 Δq_cmd를 검증된 입력 변화로 취급하지 않았다. 원본의 `from_deg`는 계획 기준값이며, "
              "첫 명령에 대해서는 실제 이전 목표였다고 보장할 수 없다.", "",
              "## 3. 통신 주기와 정지 데이터", "",
              "| 파일 | 샘플 수 | 평균 간격 (ms) | 간격 표준편차 (ms) | P99 (ms) | 최대 (ms) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in baseline_rows:
        lines.append(f"| `{row['file']}` | {row['samples']} | {row['interval_mean_ms']:.3f} | {row['interval_std_ms']:.3f} | {row['interval_p99_ms']:.3f} | {row['interval_max_ms']:.3f} |")
    lines += ["", "연속 읽기의 최대 처리율과 50 ms로 예약한 기록 주기는 다른 값이다. "
              "이번 실험은 고정 간격을 가정하지 않고 매 샘플의 실제 시각을 분석에 사용했다. "
              "이 결과만으로 MPPI의 실시간 제어 주기가 보장되는 것은 아니다.", "",
              "정지 데이터에서 보고된 각도가 일정하더라도 실제 물리적 잡음이 0이라는 뜻은 아니다. "
              "반올림·양자화·내부 갱신 방식 아래에서 변화가 관측되지 않았다는 뜻이며, "
              "정지 기록만으로 엔코더 분해능이나 내부 센서 갱신 주기를 결정할 수 없다.", "",
              "## 4. 관절별 step 응답", "",
              "아래 그래프는 **fitting에 쓰지 않은 두 번째 반복**이다. 파란 점은 실측값, 회색 점선은 명령값, "
              "초록 선은 지연 포함 2차 모델, 빨간 선은 검증 RMSE가 가장 낮은 후보다. "
              "세로축은 각 실험 시작 실측각에 대한 변화량이다."]
    for key in results:
        joint = int(key[1:])
        rows = table[(table.joint == joint) & table.observed_moving]
        lines += ["", f"### {key}", "",
                  f"![{key} 검증 응답]({link(figures / f'{key}_validation.png')})", "",
                  f"움직임이 관측된 step의 첫 반응 시각 중앙값은 {text_number(rows.onset_s.median())} s, "
                  f"10–90% 상승시간 중앙값은 {text_number(rows.rise_10_90_s.median())} s다."]
    lines += ["", "반응 시작은 시작각 대비 `max(0.09°, 정지 표준편차의 3배)`를 넘는 첫 샘플로 정의했다. "
              "이는 물리적 모터 기동 지연의 정확한 측정값이 아니다. 10–90%는 명령 크기가 아닌 **관측된 최종 변화량** 기준이다. "
              "최종 오차가 클 때 이 상승시간을 목표 도달시간으로 해석하면 안 된다.", "",
              "[전체 step 지표 CSV](" + link(output / "step_metrics.csv") + ")에는 방향·명령 크기·최종 오차·오버슈트·정착시간을 기록했다. "
              "정착시간은 관측 최종값의 `max(0.1°, 관측 변화량의 2%)` 밴드에 들어간 뒤 0.3초 이상 유지한 시각이다. "
              "관측 변화가 0.15° 미만이면 지연·상승시간을 산출하지 않았다.", "",
              "### 움직이지 않은 작은 복귀 명령의 예", "",
              "| 관절·반복·순서 | 명령 변화 (°) | 시작 실측각 (°) | 새 목표각 (°) | 최종 실측각 (°) | 실측 변화 (°) |",
              "|---|---:|---:|---:|---:|---:|"]
    for event in events:
        if event["joint"] in (1, 5) and event["repeat"] == 2 and event["step"] in (2, 4):
            row = table[table.file == event["file"]].iloc[0]
            lines.append(f"| J{event['joint']} · 2 · {event['step']} | {event['delta']:+.2f} | {event['y0']:.2f} | {event['to_deg']:.2f} | {row.final_angle_deg:.2f} | {row.output_delta_deg:+.2f} |")
    lines += ["", "명령값이 바뀌어도 **새 목표와 현재 실측각의 차이**가 작으면 추가 변화가 보고되지 않는 예다. "
              "실측 반응이 없는 기록도 유효한 관찰이며, 임의의 상승시간을 만들거나 fitting에서 제거하지 않았다. "
              "이 현상을 마찰·백래시·펌웨어 중 한 원인으로 단정하지 않는다.", "",
              "## 5. 모델 비교와 파라미터", "",
              "표준 2차 모델은 다음 식의 파라미터를 첫 반복에서 추정했다.", "",
              "```text", "G(s) = Kdc * wn² / (s² + 2*zeta*wn*s + wn²) * exp(-L*s)", "```", "",
              "각 step 직전 속도는 0으로 놓고, 직전 실측각을 위치 초기값으로 사용했다. "
              "2차·1차·일반 속도 제한 모델에는 명령의 실제 변화량 Δq_cmd를 입력했다. "
              "이 가정이 정상상태 오차·이력 효과를 설명하지 못하면 검증 오차로 드러난다. "
              "직전 실측각을 매번 다시 제공한 약 2초의 국소 예측 평가이며, 전체 연속 명령열을 관측 갱신 없이 "
              "예측한 open-loop rollout 검증은 아니다.", "",
              "| 관절 | wn (rad/s) | zeta | L (s) | Kdc | 학습 RMSE (°) | 검증 RMSE (°) | 경계 근처 파라미터 |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for key, result in results.items():
        fit = result["models"]["second_order"]
        p = fit["parameters"]
        lines.append(f"| {key} | {p['omega_n_rad_s']:.3f} | {p['zeta']:.3f} | {p['delay_s']:.3f} | {p['dc_gain']:.3f} | {fit['train_rmse_deg']:.3f} | {fit['validation_rmse_deg']:.3f} | {', '.join(fit['near_bound_parameters']) or '없음'} |")
    lines += ["", "이 값들은 **후보 모델의 추정 결과**이며 자동 채택한 제어 파라미터가 아니다. "
              "파라미터가 탐색 경계에 붙거나, 응답 변화가 몇 개 센서 샘플에만 나타나면 wn·zeta·L을 각각 정확히 구분하기 어렵다. "
              "명령 직후 50 ms 무조회 구간과 시리얼 호출 시간 때문에, 표의 L을 밀리초 단위의 물리적 지연으로 해석하지 않는다.", "",
              "추가 후보는 1차계, 속도 제한계, 데드밴드가 포함된 속도 제한계다. 마지막 후보는 다음 국소 모델을 사용했다.", "",
              "```text", "e = q_cmd(t - L) - q", "dq/dt = sign(e) * min(vmax, k * max(abs(e) - d, 0))", "```", "",
              "d는 목표 근처에서 추가 움직임이 발생하지 않는 유효 오차 폭이다. "
              "기어 마찰·백래시·펌웨어 허용오차 중 무엇이 원인인지는 이 실험만으로 분리할 수 없다. "
              "이 모델은 초기 실측각과 새 목표각 사이의 오차를 사용한다. 단순 전달함수의 DC gain과 d를 같은 뜻으로 해석하면 안 된다.", "",
              "| 관절 | 2차 RMSE | 1차 RMSE | 속도 제한 RMSE | 데드밴드·속도 제한 RMSE | d (°) |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, result in results.items():
        values = result["models"]
        scores = " | ".join(f"{values[m]['validation_rmse_deg']:.3f}" for m in MODELS)
        lines.append(f"| {key} | {scores} | {values['deadzone_velocity']['parameters']['deadband_deg']:.3f} |")
    lines += ["", "데드밴드 후보의 동특성 추정값은 다음과 같다. k는 데드밴드 밖 오차가 줄어드는 속도를 정한다. "
              "이번 작은 step에서는 속도 제한이 활성화되지 않아 vmax를 분리 추정하지 못했다. "
              "따라서 아래 k·L·d는 국소 후보이며, 수치 최적화가 반환한 vmax를 실제 관절 최대 속도로 사용하지 않는다.", "",
              "| 관절 | k (1/s) | 유효 지연 L (s) | 유효 데드밴드 d (°) | vmax 분리 식별 |",
              "|---|---:|---:|---:|---|"]
    for key, result in results.items():
        p = result["models"]["deadzone_velocity"]["parameters"]
        lines.append(f"| {key} | {p['rate_gain_per_s']:.2f} | {p['delay_s']:.3f} | {p['deadband_deg']:.3f} | 불가 |")
    lines += ["", "[전체 모델 파라미터](" + link(output / "model_comparison.json") + ")에 "
              "최적화 수렴 여부, 검증 최대 오차, 경계 근접 여부, Jacobian 조건수를 포함했다. "
              "최저 RMSE 모델도 경계 파라미터가 있으면 해당 파라미터를 확정값으로 사용하면 안 된다. "
              "Jacobian 조건수가 null이면 일부 파라미터가 이번 데이터에 영향을 거의 주지 않아 분리 추정되지 않았다는 뜻이다. "
              "특히 vmax 제한이 활성화되지 않은 후보의 vmax는 측정된 최대 관절 속도가 아니다.", "",
              "## 6. 다른 관절의 움직임", "",
              f"![다른 관절 변화]({link(figures / 'other_joint_motion.png')})", "",
              f"한 관절을 명령했을 때 다른 관절에서 관측한 최대 변화는 **{coupling.max():.3f}°**다. "
              "각 step 직전 다른 관절의 중앙값을 기준으로 계산했으며 대각선은 제외했다. "
              "이는 이번 작은 단일축 실험에서의 관측량이다. 다관절 동시 운동이나 다른 자세의 동역학 결합이 없다는 증거는 아니다.", "",
              f"J2 명령 중 J5에서 약 {coupling[1, 4]:.2f}°, J4에서 약 {coupling[1, 3]:.2f}°가 관측됐고, "
              f"J3 명령 중 J5에서도 약 {coupling[2, 4]:.2f}°가 관측됐다. "
              "명령 크기 3°에 비해 무시하기 어려운 크기다. 이는 센서값에서 관측된 상호 변화이며, "
              "동역학 결합 계수나 원인을 확정한 것은 아니다. 다음 검증에서는 해당 쌍의 반복성과 다관절 예측 오차를 확인해야 한다.", "",
              "## 7. 연구에 적용할 판단", "",
              "1. **입출력 구조는 q_cmd → q_actual로 유지한다.** 이번 실측은 내부 서보를 포함한 폐루프 응답이다.",
              "2. **작은 명령에서 남는 위치 오차와 움직이지 않는 명령을 우선 반영한다.** 목표 근처 오차를 버리거나 단위 DC gain을 강제하면 MPPI가 예측하는 미세 운동이 실제로 일어나지 않을 수 있다.",
              "3. **독립 관절 모델의 충분성 판단은 보류한다.** J2·J3 명령에서 다른 관절의 1° 이상 변화가 관측됐으므로, 이 관절 쌍의 반복 실험과 결합을 포함한 예측 검증을 우선한다. 더 큰 step, 다른 자세, 다른 speed 검증도 남아 있다.",
              "4. **이번 ±3°·speed=20 범위에서만 모델을 해석한다.** 더 큰 동작이나 높은 speed에서 속도·가속도 제한이 드러날 수 있다. 현재 후보의 vmax가 경계에 있으면 최대 속도가 측정된 것이 아니다.",
              "5. **현재 데이터로 물리적 마찰 계수나 내부 PID gain을 역산하지 않는다.** 신호가 계단형으로 보이거나 멈추는 현상에는 센서 양자화와 펌웨어도 관여할 수 있다. 상승 구간이 수 개 샘플에만 나타나므로 1차·2차 형태의 차이와 개별 동특성 파라미터에도 불확실성이 크다.", "",
              "### 다음 실험의 우선순위", "",
              "| 우선순위 | 확인할 내용 | 목적 |",
              "|---|---|---|",
              "| 1 | J2↔J4·J5, J3↔J5의 동일 조건 반복 | 다른 관절 변화가 재현되는지 확인 |",
              "| 2 | 같은 시작 실측각에서 양방향 미세 명령·복귀 | 데드밴드와 방향·이력 의존성을 구분 |",
              "| 3 | 세 번째 독립 반복과 연속 명령열 예측 | 후보 선택에 쓰지 않은 데이터와 실제 MPPI rollout 방식 검증 |",
              "| 4 | 다른 자세·speed·더 큰 step | 자세 의존성 및 속도·가속도 제한 검증 |", "",
              "위 후속 실험은 이 보고서에 포함된 완료 작업이 아니라 다음 연구 단계다.", "",
              "## 8. 재현과 원본", "",
              f"- 원격 릴리스: `{transfer['source_release']}`.",
              "- 노트북 원본: " + "; ".join(f"`{directory.as_posix()}`" for directory in sources) + ".",
              f"- 원본 CSV 수: {len(events)}개 완료 step. 중단된 사전 시도는 별도 다운로드 폴더에 보존했으며 학습에 혼합하지 않았다.",
              "- 분석 명령:", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -m analysis.report_identification " + " ".join(f'"{directory.as_posix()}"' for directory in sources) + f' --output "{output.as_posix()}"' +
              (f" --baseline-download \"{Path(args.baseline_download).resolve().as_posix()}\"" if args.baseline_download else ""),
              "```", "",
              "### 공식 API 참고", "",
              "실험 데이터와 수치의 근거는 위 CSV·메타데이터다. API 명령의 의미는 "
              "[Elephant Robotics Python API](https://github.com/elephantrobotics/pymycobot/blob/main/docs/MyCobot_280_en.md)를 참고했다. "
              "현재 공식 문서의 최신 클래스와 이번 장치에 설치된 3.5.3의 MyCobot 클래스는 버전이 다르므로, "
              "실행에는 장치에 설치된 메서드 구현을 확인해 사용했다."]
    final_status_path = output / "final_robot_status.json"
    if final_status_path.exists():
        status = json.loads(final_status_path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(status["utc"]).astimezone(ZoneInfo("Asia/Seoul"))
        reference = metadata.get("reference_angles_deg", metadata["initial_angles_deg"])
        lines += ["", "## 9. 실험 종료 후 로봇 상태", "",
                  f"{timestamp:%Y-%m-%d %H:%M:%S} 한국시간의 읽기 전용 상태 확인 결과: "
                  f"`is_moving={status['is_moving']}`, `power_on={status['power_on']}`, "
                  f"`all_servos_enabled={status['all_servos_enabled']}`.", "",
                  "각 관절의 마지막 목표는 기준각으로 복귀하도록 보냈으나, 실측각에는 아래의 잔류 오차가 남았다. "
                  "전체 영점 자세로 이동한 것은 아니다.", "",
                  "| 관절 | 기준각 (°) | 종료 후 실측각 (°) | 실측−기준 (°) |",
                  "|---|---:|---:|---:|"]
        for i, (target, actual) in enumerate(zip(reference, status["angles_deg"]), start=1):
            lines.append(f"| J{i} | {target:.2f} | {actual:.2f} | {actual-target:+.2f} |")
        lines += ["", f"[종료 상태 원본]({link(final_status_path)})"]
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REPORT", report_path)


if __name__ == "__main__":
    main()
