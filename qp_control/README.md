# myCobot OSQP 제어

젯슨의 `/home/er/workspace/urop2-osqp`에 별도 가상환경과 버전별 배포 디렉터리를 둔다.
젯슨이 관절각 읽기, QP 풀이, 각도 명령 전송을 수행한다. 노트북은 SSH로 실험을 시작하고 CSV를 회수한다.

## 노트북에서 실행

프로젝트 루트에서 아래 명령을 사용한다. 배포 중 실행하는 자체 검사는 시리얼 포트를 열지 않는다.

```powershell
.\.venv\Scripts\python.exe tools/jetson.py qp-deploy
```

다른 로봇 제어 세션을 종료하고, 작업 공간을 비운 상태에서 현재 플랜지 위치에 가까운 목표를 입력한다.
`X Y Z`는 로봇 베이스 기준 플랜지 원점의 mm 좌표다. 그리퍼 끝점을 목표로 쓰려면 별도의 TCP 적용이 필요하다.

```powershell
.\.venv\Scripts\python.exe tools/jetson.py qp-run --goal-mm X Y Z --duration 10 --confirm-motion I_UNDERSTAND
.\.venv\Scripts\python.exe tools/jetson.py qp-fetch
```

기본 주기는 0.05초이며 주기 초과가 반복되면 다음 실행에 `--period 0.10`을 추가한다.
목표 도달 기준은 3 mm다. 3회 연속 주기 초과, 실측 각도 한계 이탈, 읽기·QP 실패, 또는 시간 제한에 이르면 제어기를 종료하고 로봇에 정지 명령을 보낸다.
QP는 관절별 명령 속도를 ±10°/s로 제한한다. 실제 관절 속도는 CSV에서 확인한다.
QP 제약은 관절 각도와 명령 속도에 적용되므로, 첫 실험은 장애물이 없는 공간에서 짧은 이동으로 수행한다.

## CSV 분석

`qp-fetch`가 출력한 디렉터리의 `data/raw`에 CSV가 있다. `work_ms`로 읽기 시작부터 명령 API 반환까지의 작업 시간을 확인하고,
`deadline_missed`를 합산해 주기 초과를 센다. 50 ms에서 작업 시간 P99가 40 ms 이내이고 초과가 없으면 20 Hz를 유지한다.
`ee_error_mm`로 목표 오차를, 인접한 `q_actual_j*_deg`의 차를 `sample_s`의 차로 나눠 실제 관절 속도를 확인한다.
