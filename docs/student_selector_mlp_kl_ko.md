# Student Object Selector의 MLP와 KL Divergence

## SameSpaceSelector부터 실제 baseline16 파이프라인까지

작성일: 2026-07-18
대상 저장소: `.`

---

## 1. 가장 중요한 결론

이 파이프라인에 대한 이해는 다음 두 문장으로 정리할 수 있다.

1. 기본 student selector는 scene graph에서 추출한 object별 수치 feature를 MLP에 넣어 object별 logit과 확률분포를 만든다.
2. teacher distribution은 추론 시 student distribution을 직접 덮어쓰는 입력이 아니라, 학습 시 KL divergence loss를 통해 student MLP의 파라미터를 업데이트하는 target이다.

실제 motion student에 사용된 `baseline16_torso_cone_w2`는 기본 MLP 한 번으로 끝나는 모델이 아니다. 이전 baseline의 score를 residual MLP로 보정하고, 마지막에 torso-cone feature를 선형으로 더하는 다단계 selector이다.

전체 흐름은 다음과 같다.

```text
scene graph / precomputed feature CSV
        |
        v
object별 numeric feature tensor [B, N, D]
        |
        v
student selector 또는 residual reranker
        |
        v
object별 logits [B, N]
        |
        v
masked softmax
        |
        v
student distribution [B, N]
        |
        +---- 학습 시: KL(P_teacher || P_student)
        |
        v
argmax object 선택
        |
        v
선택 object의 point cloud export
        |
        v
SIF3D motion student 입력
```

---

## 2. MLP는 raw scene graph를 직접 받지 않는다

`SameSpaceSelector.forward()`가 받는 입력은 JSON이나 scene graph object가 아니다.

```python
candidate_features: torch.Tensor  # [B, N, D]
valid_mask: torch.Tensor          # [B, N]
```

- `B`: batch size
- `N`: 한 sample 안의 최대 object candidate 수
- `D`: object 하나를 표현하는 feature 개수

Dataset 단계가 먼저 `scene_graphs/{train,test}/samples.jsonl`과 candidate CSV를 읽고 object별 feature vector를 만든다.

관련 코드:

- `object_selector/same_space/dataset.py:103` — scene graph의 object metrics 로딩
- `object_selector/same_space/dataset.py:151` — sample별 candidate feature tensor 생성
- `object_selector/same_space/dataset.py:257` — train mean/std 계산
- `object_selector/same_space/dataset.py:266` — feature 정규화
- `object_selector/same_space/dataset.py:319` — 서로 다른 candidate 수를 batch로 padding

기본 feature 목록은 다음과 같다.

```python
DEFAULT_SAME_SPACE_FEATURE_COLUMNS = (
    "face_alignment_norm",
    "body_alignment_norm",
    "face_line_score",
    "future_hand_contact_score",
    "future_root_endpoint_score",
    "root_alignment",
    "torso_alignment",
    "root_torso_alignment",
    "face_normal_alignment",
)
```

요청한 feature가 candidate CSV에 있으면 CSV 값을 사용한다. CSV에 없으면 scene graph JSON의 각 object 안에 있는 `metrics`에서 값을 찾는다.

학습 전에 train split 통계로 정규화한다.

```text
x_normalized = (x - train_mean) / train_std
```

따라서 모델이 실제로 받는 것은 "scene graph 자체"가 아니라 "scene graph 및 추가 motion feature에서 추출·정규화된 수치 tensor"이다.

---

## 3. SameSpaceSelector의 구조

구현 파일:

```text
object_selector/same_space/model.py
```

핵심 클래스:

```python
class SameSpaceSelector(nn.Module):
```

모델은 크게 세 부분으로 나뉜다.

```text
candidate_encoder
context_encoder
score_head
```

### 3.1 candidate_encoder: 각 object를 독립적으로 인코딩

```python
self.candidate_encoder = _make_mlp(
    input_dim,
    hidden_dim,
    hidden_dim,
    dropout,
)
```

구조는 다음과 같다.

```text
D
-> Linear(D, H)
-> ReLU
-> Linear(H, H)
-> ReLU
```

입력과 출력 shape:

```text
candidate_features: [B, N, D]
encoded:            [B, N, H]
```

동일한 `candidate_encoder`가 모든 object에 공유된다. chair, table, bed마다 별도 MLP가 있는 것이 아니다.

### 3.2 mean/max pooling: scene 전체 후보 요약

각 object를 개별적으로만 보면 다른 후보와의 상대적인 관계를 알기 어렵다. 따라서 모든 valid candidate embedding을 mean pooling과 max pooling으로 요약한다.

```python
pooled_mean = mean(encoded over valid candidates)
pooled_max = max(encoded over valid candidates)
```

각각의 shape은 `[B, H]`이고, 둘을 연결하면 `[B, 2H]`가 된다.

Padding candidate는 `valid_mask=False`이므로 pooling에서 제외된다.

### 3.3 context_encoder: sample 전체 context 생성

```python
context = self.context_encoder(
    torch.cat([pooled_mean, pooled_max], dim=-1)
)
```

구조:

```text
2H
-> Linear(2H, H)
-> ReLU
-> Linear(H, H)
-> ReLU
```

출력 `context`의 shape은 `[B, H]`이다. 이 vector는 한 sample 안의 object 후보 집합 전체를 표현한다.

### 3.4 score_head: object별 logit 생성

Scene context를 각 object 위치에 복사한다.

```python
expanded_context = context[:, None, :].expand(-1, N, -1)
```

각 candidate embedding과 context를 연결한다.

```python
pair = torch.cat([encoded, expanded_context], dim=-1)
```

Shape은 `[B, N, 2H]`이다.

이 tensor를 score head에 넣는다.

```text
2H
-> Linear(2H, H)
-> ReLU
-> Linear(H, 1)
```

최종 출력은 object별 logit이다.

```text
logits: [B, N]
```

마지막 Linear에는 ReLU가 없으므로 logit은 음수 또는 양수 모두 가능하다.

중요한 점은 object score가 해당 object feature만으로 결정되지 않는다는 것이다. `context`가 함께 들어가므로 다른 후보들의 feature도 해당 object score에 영향을 준다.

---

## 4. Masked softmax와 student distribution

Logit을 probability distribution으로 바꾸는 코드는 다음과 같다.

```python
distribution = masked_softmax(logits, valid_mask)
```

일반 softmax 수식은 다음과 같다.

```text
P_student(i) = exp(z_i) / sum_j exp(z_j)
```

여기서 `z_i`는 i번째 candidate의 logit이다.

Candidate 수가 sample마다 다르기 때문에 batch에서 padding이 필요하다. `masked_softmax()`는 padding 위치의 logit을 `-inf`로 바꾼다.

```python
masked_logits = logits.masked_fill(~valid, -torch.inf)
probs = torch.softmax(masked_logits, dim=1)
```

따라서 invalid candidate의 probability는 0이 된다.

예를 들어 다음과 같은 logit이 있다면:

```text
logits = [2.0, 1.0, -0.5]
```

Softmax 결과는 대략 다음과 같다.

```text
student distribution = [0.69, 0.25, 0.06]
```

가장 높은 probability를 가진 object가 student top-1 선택이다.

```python
selected_idx = distribution.argmax(dim=1)
```

---

## 5. Teacher distribution 생성

기본 student trainer는 다음 파일에 있다.

```text
scripts/train_same_space_student_selector.py
```

Teacher candidate CSV의 `fusion_total` 같은 score를 읽은 뒤 temperature softmax를 적용한다.

```python
teacher_posterior = torch.softmax(
    teacher_scores / temperature,
    dim=0,
)
```

수식:

```text
P_teacher(i)
  = exp(teacher_score_i / T)
    / sum_j exp(teacher_score_j / T)
```

기본 student trainer의 temperature 기본값은 `T=2.0`이다.

- 작은 T: teacher top-1에 probability가 집중된다.
- 큰 T: 다른 후보에도 probability가 분산된다.
- 분산된 target은 top-1 label 하나보다 후보 간 상대적 선호 정보를 더 많이 전달한다.

예를 들어 teacher score가 다음과 같다고 하자.

```text
teacher scores = [5.0, 4.0, 1.0]
```

Temperature softmax 후에는 다음과 비슷한 분포가 만들어질 수 있다.

```text
teacher distribution = [0.57, 0.35, 0.08]
```

이는 첫 번째 object가 가장 좋지만 두 번째도 어느 정도 가능성이 있다는 정보를 student에게 전달한다.

관련 코드:

- `scripts/train_same_space_student_selector.py:278` — score softmax
- `scripts/train_same_space_student_selector.py:299` — teacher target sample 생성
- `scripts/train_same_space_student_selector.py:489` — teacher score/posterior 로딩

---

## 6. KL Divergence의 정확한 의미

구현:

```text
scripts/train_same_space_student_selector.py:133
```

코드:

```python
return (
    teacher
    * (teacher_safe.log() - student_safe.log())
).sum(dim=1).mean()
```

수식으로는 다음과 같다.

```text
KL(P_teacher || P_student)
  = sum_i P_teacher(i)
      * [log P_teacher(i) - log P_student(i)]
```

KL divergence는 비대칭이므로 방향이 중요하다. 이 코드가 계산하는 값은 다음이다.

```text
KL(teacher || student)
```

다음 값과는 다르다.

```text
KL(student || teacher)
```

### 6.1 KL이 student를 수정하는 방식

Teacher와 student distribution이 다음과 같다고 가정한다.

```text
teacher = [0.70, 0.20, 0.10]
student = [0.30, 0.40, 0.30]
```

KL loss를 줄이는 방향으로 역전파하면 첫 번째 candidate의 student logit은 올라가고, 두 번째와 세 번째 logit은 상대적으로 내려간다. 결과적으로 student distribution이 teacher distribution에 가까워진다.

하지만 teacher distribution이 `SameSpaceSelector.forward()`의 입력으로 들어가는 것은 아니다.

학습 시:

```text
student features -> student forward -> student distribution
teacher scores   -> teacher distribution
student distribution + teacher distribution -> KL loss
KL loss -> backward -> student MLP weights update
```

추론 시:

```text
student features -> trained student MLP -> student distribution
```

추론에는 teacher distribution이 필요 없다.

학습 코드:

```python
outputs = model(features, valid_mask)

kl_value = distribution_kl(
    outputs["distribution"],
    teacher_posterior,
    valid_mask,
)

loss = lambda_kl * kl_value + other_losses

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

관련 위치:

- `scripts/train_same_space_student_selector.py:554` — training batch
- `scripts/train_same_space_student_selector.py:564` — student forward
- `scripts/train_same_space_student_selector.py:565` — KL 계산
- `scripts/train_same_space_student_selector.py:582` — 전체 loss
- `scripts/train_same_space_student_selector.py:590` — backward 및 optimizer update

---

## 7. 기본 student의 전체 training loss

기본 trainer의 전체 loss는 다음과 같다.

```text
L_total
  = lambda_KL         * L_KL
  + lambda_teacher_CE * L_teacher_CE
  + lambda_rank       * L_rank
  + lambda_pairgap    * L_pairgap
  + lambda_GT_CE      * L_GT_CE
```

각 항의 목적:

- `L_KL`: 전체 probability distribution을 teacher와 맞춘다.
- `L_teacher_CE`: teacher top-1 candidate를 맞춘다.
- `L_rank`: teacher의 candidate 순서를 유지한다.
- `L_pairgap`: teacher 후보 사이의 score 간격을 학습한다.
- `L_GT_CE`: 실제 GT object를 맞추도록 한다.

KL은 top-1 하나만 가르치지 않고 전체 분포를 전달한다. CE와 ranking loss는 top-1 및 순서 학습을 보강한다.

---

## 8. 실제 baseline16은 residual reranker이다

Motion student 입력에 사용된 실제 baseline16은 `SameSpaceSelector`가 처음부터 새 distribution을 만드는 구조와 다르다.

실제 구현:

```text
scripts/train_same_space_residual_reranker.py
```

핵심 클래스:

```python
class ResidualReranker(nn.Module):
```

Baseline16은 이전 baseline9가 export한 candidate CSV에서 다음 값을 읽는다.

```text
student_score     -> base_logits
student_posterior -> base_distribution
teacher_posterior -> teacher_distribution
```

따라서 baseline16 학습에서 말하는 "초기 distribution"은 baseline16 MLP가 처음부터 계산하는 것이 아니다. 이미 학습된 이전 baseline의 결과를 CSV에서 불러온다.

관련 코드:

- `scripts/train_same_space_residual_reranker.py:290` — baseline candidate rows를 sample로 구성
- `scripts/train_same_space_residual_reranker.py:310` — base logits
- `scripts/train_same_space_residual_reranker.py:311` — base distribution
- `scripts/train_same_space_residual_reranker.py:316` — teacher distribution

### 8.1 Residual MLP의 동작

Residual reranker도 candidate encoder, context encoder, score head를 사용한다. 단, 최종 head 출력은 완전한 새 logit이 아니라 기존 logit에 더할 보정값이다.

```python
raw_residual = self.head(pair).squeeze(-1)
residual = max_residual * torch.tanh(raw_residual)
final_logits = base_logits + residual
final_distribution = masked_softmax(final_logits, valid_mask)
```

Baseline16 설정은 다음과 같다.

```text
hidden_dim = 128
max_residual = 0.6
```

따라서 각 candidate logit 보정값은 대략 `[-0.6, +0.6]` 범위로 제한된다.

```text
z_final(i) = z_base(i) + r_theta(i)
P_final = softmax(z_final)
```

여기서는 "student의 기존 distribution을 수정한다"는 설명이 비교적 직접적으로 맞다. 더 정확히는 probability를 직접 더하는 것이 아니라 기존 `base_logits`에 residual을 더한 후 다시 softmax한다.

---

## 9. Baseline16의 KL과 anchor KL

Residual용 KL 구현:

```text
scripts/train_same_space_residual_reranker.py:399
```

이 함수도 `KL(second argument || first argument)`를 계산한다.

Baseline16 학습에서는 두 종류의 KL을 사용한다.

```python
kl = distribution_kl(
    final_student_distribution,
    teacher_distribution,
)

anchor = distribution_kl(
    final_student_distribution,
    base_distribution,
)
```

따라서 실제 수식은 다음과 같다.

```text
kl     = KL(P_teacher || P_final)
anchor = KL(P_base    || P_final)
```

- Teacher KL은 최종 student를 teacher 쪽으로 이동시킨다.
- Anchor KL은 최종 student가 기존 baseline에서 지나치게 멀어지지 않게 한다.

실제 baseline16 loss weight:

```text
lambda_KL          = 1.0
lambda_anchor      = 0.7
lambda_residual_L2 = 0.015
lambda_rank        = 0.7
lambda_teacher_CE  = 0.8
```

전체 loss:

```text
L_baseline16
  = 1.0   * KL(P_teacher || P_final)
  + 0.7   * KL(P_base    || P_final)
  + 0.015 * ||residual||^2
  + 0.7   * ranking_loss
  + 0.8   * teacher_top1_CE
```

이는 다음 두 목표 사이의 균형이다.

```text
teacher의 더 좋은 분포를 따라가기
                  vs.
기존 baseline의 안정적인 분포를 보존하기
```

관련 코드:

- `scripts/train_same_space_residual_reranker.py:622` — training loop
- `scripts/train_same_space_residual_reranker.py:633` — residual model forward
- `scripts/train_same_space_residual_reranker.py:634` — teacher KL
- `scripts/train_same_space_residual_reranker.py:635` — base anchor KL
- `scripts/train_same_space_residual_reranker.py:647` — 전체 loss
- `scripts/train_same_space_residual_reranker.py:654` — backward

실제 checkpoint:

```text
object_selector/same_space_runs/
  baseline16_temporal_attention_combo_seed7/
  hand_face_mean_head/best_model.pt
```

---

## 10. 최종 baseline16_torso_cone_w2 단계

Motion student가 사용한 object cache 이름은 다음과 같다.

```text
final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_points_1024.pth
```

이 최종 단계에는 추가 MLP 또는 새로운 KL 학습이 없다. 학습된 baseline16 score에 정규화된 torso-cone feature를 2.0배하여 더한다.

```text
torso_cone_z
  = (torso_cone_raw - train_mean) / train_std

final_score
  = baseline16_score + 2.0 * torso_cone_z

final_distribution
  = softmax(final_score)
```

해당 run의 train 통계:

```text
train_mean = 0.3263414139
train_std  = 0.3143460090
weight     = 2.0
```

설정 파일:

```text
object_selector/same_space_runs/
  baseline16_plus_torso_cone_w2_seed7/summary.json
```

최종 파이프라인은 다음과 같다.

```text
이전 baseline9 logits/distribution
        |
        v
baseline16 residual MLP
  - KL(P_teacher || P_final)
  - KL(P_base || P_final)
  - rank / CE / residual regularization
        |
        v
baseline16 score
        |
        +---- 2.0 * normalized torso-cone feature
        |
        v
최종 score 및 softmax distribution
        |
        v
argmax object 선택
        |
        v
1024-point object point cloud export
        |
        v
SIF3D motion student 입력
```

---

## 11. Training과 inference의 차이

### Training

Training에는 teacher information이 존재한다.

```text
student-side features
teacher distribution
base distribution
GT / teacher top-1 index
```

Teacher distribution은 KL, ranking, CE loss를 계산하는 데 사용된다. 이 loss의 gradient로 student 또는 residual MLP weight가 업데이트된다.

### Inference

Inference에는 teacher distribution이 없다.

```text
student-side features
base model output (residual chain인 경우)
trained MLP weights
```

학습된 student가 스스로 distribution을 출력한다. 즉 knowledge distillation의 결과는 teacher probability를 저장해 두고 매번 섞는 것이 아니라, teacher의 행동이 student weight 안에 학습되는 것이다.

---

## 12. Motion checkpoint와 selector checkpoint의 차이

다음 motion checkpoint에는 selector MLP가 포함되어 있지 않다.

```text
new_runs/ours_student_train_dual_old_0351_kd_third1423/
  best_loss_trans.pth
```

이 파일은 SIF3D motion prediction model의 checkpoint이다.

Selector는 먼저 object distribution을 만들고 top-1 object를 선택한다. 이후 선택한 object point cloud가 별도 `.pth` cache로 export된다.

```text
selector checkpoint
  -> selected object id
  -> selected object point cloud cache
  -> motion model input
```

따라서 motion model은 raw scene graph나 teacher distribution을 받지 않는다. 다음처럼 이미 만들어진 object point cloud를 받는다.

```text
final_object_student_baseline16_torso_cone_w2_..._points_1024.pth
```

Checkpoint의 역할은 다음과 같이 구분해야 한다.

```text
best_model.pt
  = object selector residual MLP checkpoint

best_loss_trans.pth
  = SIF3D motion student checkpoint
```

---

## 13. 자주 헷갈리는 부분

### 오해 1: Teacher distribution이 student forward 입력이다

아니다. Teacher distribution은 학습 loss target이다. Student forward 입력은 candidate feature와 valid mask이다.

### 오해 2: KL이 매 inference마다 student probability를 직접 바꾼다

아니다. KL은 training 중 gradient를 통해 MLP weight를 바꾼다. Inference에는 KL 계산이 없다.

### 오해 3: Baseline16이 SameSpaceSelector 한 개이다

아니다. 실제 baseline16은 이전 baseline score를 residual MLP로 보정하는 reranker이다.

### 오해 4: Residual은 probability에 직접 더한다

아니다. Residual은 base logit에 더해진다. 이후 softmax로 새로운 distribution을 만든다.

```text
correct:   softmax(base_logits + residual)
incorrect: base_probability + residual
```

### 오해 5: 최종 torso-cone w2도 KL로 학습된다

아니다. 최종 w2 단계는 정규화된 torso-cone feature를 baseline16 score에 선형으로 더하는 후처리이다.

---

## 14. 핵심 코드 위치 요약

```text
기본 selector MLP
  object_selector/same_space/model.py:34

scene graph -> numeric feature
  object_selector/same_space/dataset.py:103
  object_selector/same_space/dataset.py:151

teacher score -> teacher distribution
  scripts/train_same_space_student_selector.py:278
  scripts/train_same_space_student_selector.py:299

기본 student KL
  scripts/train_same_space_student_selector.py:133
  scripts/train_same_space_student_selector.py:565

baseline16 residual MLP
  scripts/train_same_space_residual_reranker.py:71
  scripts/train_same_space_residual_reranker.py:95

baseline16 teacher/base KL
  scripts/train_same_space_residual_reranker.py:399
  scripts/train_same_space_residual_reranker.py:634
  scripts/train_same_space_residual_reranker.py:635

baseline16 전체 loss
  scripts/train_same_space_residual_reranker.py:647

baseline16 checkpoint
  object_selector/same_space_runs/
    baseline16_temporal_attention_combo_seed7/
    hand_face_mean_head/best_model.pt

최종 torso-cone w2 설정
  object_selector/same_space_runs/
    baseline16_plus_torso_cone_w2_seed7/summary.json
```

---

## 15. 한 문장 요약

기본 student는 scene-derived object feature를 MLP로 encode하여 distribution을 만들고 `KL(P_teacher || P_student)`로 그 MLP를 학습하며, 실제 baseline16은 이전 baseline logits에 학습된 residual을 더해 분포를 보정하고, 최종 `torso_cone_w2` 단계에서 torso-cone feature를 선형으로 한 번 더 반영한 뒤 top-1 object point cloud를 SIF3D motion student에 전달한다.
