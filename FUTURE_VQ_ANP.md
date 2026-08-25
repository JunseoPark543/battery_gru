# Future V-Q Latent ANP

이 패키지는 **최근 100개 완료 cycle의 전체 V-Q 곡선**만 보고 이후 모든 cycle의
완전한 V-Q 곡선과 각 곡선의 종료점 `q_end`를 예측한다. 기존
`partial_vq_forecasting`, `streaming_soh`, `streaming_soh_anp` 코드는 수정하지 않았다.

## 구조

```text
100 completed V-Q curves
  -> shared 1-D CNN + masked point attention (curve encoder)
  -> 2-layer GRU (inter-cycle ageing history)
  -> future-cycle cross attention
  -> latent ANP z ~ p(z | observed history)
  -> coordinate decoder V(future cycle, Q) + q_end decoder
```

- `Q = discharge_capacity_in_Ah / nominal_capacity_in_Ah`인 고정 좌표를 사용한다.
- 개별 곡선의 마지막 Q로 정규화하지 않아 미래 `q_end` 누수를 막는다.
- 미래 곡선은 예측값을 다시 입력하지 않고 `(future cycle, Q)` 좌표에서 직접 생성한다.
- 하나의 latent sample을 모든 미래 좌표가 공유하므로 uncertainty가 미래 surface 전체에서
  일관된다.
- 학습 때만 미래 정답이 posterior `q(z|context,target)`에 들어간다. validation/test는
  prior `p(z|context)`만 사용한다.
- train/validation/test와 voltage scaler는 모두 cell 단위로 분리된다.

## 전체 학습 + test

```bash
python -m battery_weighted_maml.future_vq_anp.run_suite \
  --config configs/matr_future_vq_anp.yaml \
  --fold 0 \
  --device cuda
```

실험 폴더는 `outputs/future_vq_anp/matr_future_vq_anp_f0_s42`이다. `last.pt`는
500 step/validation마다 저장되고, `best.pt`는 validation CRPS가 개선될 때 저장된다.

## 중단된 학습 재개

```bash
python -m battery_weighted_maml.future_vq_anp.run_suite \
  --config configs/matr_future_vq_anp.yaml \
  --fold 0 \
  --device cuda \
  --resume outputs/future_vq_anp/matr_future_vq_anp_f0_s42/checkpoints/last.pt
```

## test만 실행

```bash
python -m battery_weighted_maml.future_vq_anp.evaluate \
  --checkpoint outputs/future_vq_anp/matr_future_vq_anp_f0_s42/checkpoints/best.pt \
  --device cuda \
  --cut-cycles 100 130 135 140 200
```

`cut=130`은 cycle 31~130의 완료 곡선을 입력해 131부터 해당 cell의 마지막 cycle까지
예측한다. `cut=135`에서는 cycle 36~135로 context만 교체한다. 재학습이나 온라인
weight update는 필요 없다.

주요 결과는 `evaluation/best/aggregate_metrics.csv`, `episode_metrics.csv`,
`curve_metrics.csv`, `plots/*_surface.png`, `plots/*_curves.png`,
`plots/calibration.png`에 저장된다.
