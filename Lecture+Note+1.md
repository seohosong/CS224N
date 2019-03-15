
---

# 2. Word Vector

- 단어를 차원공간을 나타내는 벡터로 인코딩(Encoding) 하는 작업
- 원래 단어의 갯수(|V|)보다 작은 N(N<=|V|)차원의 벡터로 나타내는 것
- 각 차원은 음성이 내포하는 의미를 인코딩 가능
    - Semantic Dimension: Tense(past vs future), Count(singular vs plural), gender(masculine vs feminine)

---

## 2-1. One-hot Encoder

- 사전 안의 모든 단어(|V|: size of vocabulary)를 단어의 갯수와 같은 |V|차원으로 표현하는 방식
- 해당 방식에서 단어 간 관계는 서로 독립적이며, 따라서 단어 간 유사도 개념을 반영하지 못함
- 예를 들어, 

$$W^{hotel} = [1,0,0], W^{motel} = [0, 1, 0], (W^{cat} = [0, 0, 1]$$

- 위와 같은 3개의 단어벡터로 표현되는 사전이 있다면, 

$$ (W^{hotel})^{T}W^{motel} = (W^{hotel})^{T}(W^{cat})=0$$

---

## 2-2. SVD Based Model

- 비슷한 혹은 유사한 단어들은 같은 문서에 자주 존재한다라는 Bold한 가정을 전제로 함
    - Bank, Bond, Stock은 같은 문서 내 자주 등장할 가능성이 높지만 Bank, Fish, Soccer 은 상대적으로 같은 문서 내 등장할 가능성이 낮다는 주장

### (1) Word Document Matrix

- M개의 문서에 대해 단어들의 출현 빈도를 나타내는 행렬 $X (R^{|V|* M}$)을 생성

$$X=
\left[\begin{array}{rrr} 
x_{11}&x_{12}&...&x_{1j}&...&x_{1m}\\
...&&\\
x_{i1}&x_{i2}&...&x_{ij}&...&x_{1m}\\
...&\\
x_{|V|1}&...&...&...&...&x_{|V|m}
\end{array}\right]
$$


- 여기서 $x_{ij}$ 는 i번째 단어가 j번째 문서에 출현한 빈도를 의미

### (2) Window Based Co-occurance Matrix

- 단어 간 Affinity Matrix(친밀도 행렬) 생성
- 말뭉치(Corpus)에서 임의 설정한 Window Size 내 두 단어들의 동시 출현 빈도를 행렬로 표현
- 여기서 Window Size는 우리가 들여다볼 범위를 의미

- Ex. 한 말뭉치 안에 다음과 같은 3개의 문장 존재
    - Sentence 1: I like swimming
    - Sentence 2: I like soccer
    - Sentence 3: I hate deep learning
- 그리고 Window Size를 1로 설정한다

- 그렇다면, 다음과 같은 친밀도 행렬을 생성 가능

![친밀도 행렬](affinitymatrix.png)

- Window Size 내 동시출현빈도가 높을수록 두 단어가 유사하다고 가정

### (3) Applying SVD to the Co-Occurance Matrix

- 원래 단어 차원행렬($R^{|V|*|V|}$)을 그보다 낮은 차원행렬($R^{|V|*k}$)로 차원축소하는 방법

#### **(3)-1. Applying SVD to X**

![SVD](bySVD.jpg)

#### **(3)-2. K개의 Singular Vector로 차원 축소**

![Singular](bySingular.jpg)

- 두 방식 모두 의미정보(Semantic)와 통사정보(Syntactic)를 가지고 있는 Word Vector를 반환

- **단점**
    1. 단어가 추가되는 속도가 빨라질수록 말뭉치 크기 및 행렬의 차원 또한 자주 바뀜
    - 동시출현하는 경우가 없으면 데이터가 Sparse해짐
    - 일반적으로 고차원의 행렬로 표현됨
    - SVD를 위한 Quadratic Cost 발생
    - 단어 빈도간 Unimbalance 문제

- 그리고 이를 해결하기 위해 다음과 같은 방법들이 고안됨
    - 문법 기능 위주 단어제외(he, she, is 등)
    - Ramp Window: Window 안에서도 단어간의 거리가 가까울수록 동시출현 Weight를 더 많이 부여
    - 피어슨 상관계수 및 Negative Cost를 0으로 부여

---

# 3. Iteration Based Methods - Word2Vec

- SVD 기반 기법들은 대용량의 데이터셋을 한번에 저장 및 계산해야 한다는 단점이 있었음
- 따라서, 한번에 하나의 반복작업 및 문맥에서 단어의 확률을 인코딩하는 방법을 고안
    1. Word Vector를 Parameter로 가지는 모델
    - 각 iteration마다 학습->오차평가->Error 유발 Parameter에대한 페널티 부과 규칙 생성 -> 규칙 업데이트 과정을 거침
    - Backpropagation(역전파)개념과 유사

## Word2Vec

- 단어를 Vector로 변환하는 자연어처리 모델의 한 기법

### 1. 알고리즘

1. CBOW(Continuous Bag-of-Words) : 주변 단어로 중심 단어(Center Word)를 예측
- Skip Gram : 중심 단어로 주변 단어를 예측(Distribution Probability 계산)

### 2. 학습 방법

1. Negative Sampling
2. Hierarchical Softmax: 사전 내 모든 단어의 확률을 계산하기 위해 tree Structure을 활용

### 3-1. Language Model(Unigram, Bigram 등)

- 여기 두 문장이 있음

$$" The\,cat\,jumped\,over\,the\,puddle"$$
$$$$
$$ "Stock\,boil\,fish\,is\,toy" $$

- 좋은 언어모델이라면 두번째 문장보다는 첫번째 문장에 더 높은 확률을 부여할 것
- 단어 시퀀수가 주어졌을 때 확률을 수식으로 표현한다면 다음과 같음

$$
P(w_1, w_2, ..., w_n)
$$

- 만약 단어 간 관계를 독립적으로 본다면, 다음과 같이 표현 가능

$$P(w_1, w_2,..., w_n) = \Pi_{i=1}^{n}P(w_i) $$

- 하지만 일반적으로 단어는 그 이전 단어에 따라 영향을 받기 때문에 독립적이지 않음. 따라서, 다음과 같은 좀더 개선된 방법이 고안됨

$$P(w_1, w_2,..., w_n) = \Pi_{i=1}^{n}P(w_i|w_{i-1}) $$

- 개선되긴 했지만 여전히 전체 문장이나 문맥을 파악하지 못하고 이웃한 단어만 고려했다는 단점이 존재. 연산량 또한 크다는 문제가 발생

### 3-2.CBOW(Continuous Bag-of-Words)

- {"The", "cat", "over", "the", "puddle"} 이라는 문맥이 주어질때, "jump" 이라는 중심단어(Center Word)를 출력하는 접근방식
- 크게 입력벡터(input vector)와 출력벡터(output vetor)로 구분할 수 있으며, 입력벡터는 문맥에 있는 단어를, 출력벡터는 센터에 있는 단어를 의미

#### Model Processing Breakdown

Step1: 입력 context size에 따라 one-hot 벡터 생성
$$ m: (x^{c-m}, ..., x^{c-1}, x^{c+1}, x^{c+m} \in R^{|v|}) $$

Step2: Embedded Word Vector 생성

$$ (\nu_{c-m} = V\nu_{c-m}, \nu_{c-m+1} = V\nu_{c-m+1},..., \nu_{c+m} = V\nu_{c+m} \in R^{|v|}) $$

Step3: 벡터들의 평균인 $\nu^{\wedge}$ 도출

$$\nu^{\wedge} = \frac{\nu_{c-m}+ \nu_{c-m+1}+...\nu_{c+m}}{2m} \in R^n$$

Step4: 스코어 벡터인 z = $U\nu^{\wedge}$ 생성(유사 벡터들의 내적값이 클수록, 스코어벡터는 해당 단어들의 유사도를 더 높게 판단)

Step5: 스코어를 확률로 변환 $y^{\wedge} = softmax(z) \in R^{|v|}$

Step6: 계산한 확률을 실제확률값과 비교

#### 입력($U$) 및 출력($V$) 메트릭스의 학습 방법

- 목적함수(Objective Function)로 Cross Entropy(손실 함수 공식에서 도출 )

$$ H(y^{\wedge}, y) = -y_{i}log(y^{\wedge}_i) $$

- 만약 우리의 예측이 정확해서 $y^{\wedge}_c$=1() 이라면, $H(y^{\wedge}, y)$=0 이됨. 즉, 어떠한 페널티나 손실이 발생하지 않음. 반면 0.01이라면, 값은 4.6정도로 손실이 발생함
- 따라서 목적 최적화를 다음과 같이 정의

$$minimize\,J = -logP(w_c|w_{c-m}, ..., w_{c-1}, w_{c+1}, ..., w_{c+m})$$

$$= -logP(u_c|v^{\wedge})$$
