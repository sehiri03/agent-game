# streamlit_app.py – Cultural Ethics Simulator (Scenario A/B 버전)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy, pearsonr
import openai  # GPT 요약 기능 쓰면 필요, 안 쓰면 에러 나도 무시 가능

# ==================== App Config ====================
st.set_page_config(page_title="Ethics GPT Sim", layout="wide")
st.title("🌍 Global AI Ethics Simulator")

# ==================== Configuration ====================
CULTURES = {
    "USA":       {"emotion": 0.3,  "social": 0.1,  "identity": 0.3,  "moral": 0.3},
    "CHINA":     {"emotion": 0.1,  "social": 0.5,  "identity": 0.2,  "moral": 0.2},
    "EUROPE":    {"emotion": 0.3,  "social": 0.2,  "identity": 0.2,  "moral": 0.3},
    "KOREA":     {"emotion": 0.2,  "social": 0.2,  "identity": 0.4,  "moral": 0.2},
    "LATIN_AM":  {"emotion": 0.4,  "social": 0.2,  "identity": 0.2,  "moral": 0.2},
    "MIDDLE_E":  {"emotion": 0.1,  "social": 0.2,  "identity": 0.2,  "moral": 0.5},
    "AFRICA":    {"emotion": 0.2,  "social": 0.4,  "identity": 0.2,  "moral": 0.2},
    # ✅ 추가된 남아시아 문화권
    "SOUTH_ASIA": {"emotion": 0.15, "social": 0.30, "identity": 0.30, "moral": 0.25},
}

# ==================== Scenario Definitions ====================
SCENARIOS = {
    "AI Hiring Bias": {
        "A": [0.20, 0.20, 0.10, 0.50],  # AI 점수 그대로 반영 (효율/성과, 규범 준수 위주)
        "B": [0.20, 0.40, 0.30, 0.10],  # 형평성, 사회적 영향, 정체성 보호 강화
    },
    "AI Facial Recognition": {
        "A": [0.10, 0.50, 0.10, 0.30],  # AI 명단 그대로 체포 (공공질서/사회 안전 중시)
        "B": [0.30, 0.20, 0.20, 0.30],  # 재검증, 인권/개인 보호를 더 반영
    },
    "Surveillance vs Freedom": {
        "A": [0.10, 0.50, 0.20, 0.20],  # 감시 강화 (사회 안전/통제 위주)
        "B": [0.30, 0.20, 0.10, 0.40],  # 자유/권리, 도덕적 원칙 중심
    }
}

# ==================== Sidebar UI ====================
scenario = st.sidebar.selectbox(
    "시나리오 선택",
    list(SCENARIOS.keys()),
    index=0
)

selected = st.sidebar.multiselect(
    "문화권 선택",
    list(CULTURES.keys()),
    default=["SOUTH_ASIA"]  # ✅ 기본값: 남아시아만 선택
)

steps = st.sidebar.slider("반복 수", 50, 500, 200, step=50)
manual = st.sidebar.checkbox("🎮 사용자 정의 가중치", False)

# ==================== Helper Functions ====================
def normalize(w: dict):
    s = sum(w.values())
    # 0 division 방지 + 최소값 하한선
    return {k: max(0.001, v) / s for k, v in w.items()}

# ==================== Agents 초기화 ====================
AGENTS = selected
AGENT_WEIGHTS = {}

for a in AGENTS:
    if manual:
        st.sidebar.markdown(f"**{a} 가중치 설정**")
        sliders = {}
        for k in ["emotion", "social", "identity", "moral"]:
            sliders[k] = st.sidebar.slider(
                f"{a} - {k.capitalize()}",
                0.0, 1.0,
                float(CULTURES[a][k])
            )
        AGENT_WEIGHTS[a] = normalize(sliders)
    else:
        # 기본 문화값 그대로 사용
        AGENT_WEIGHTS[a] = dict(CULTURES[a])

AGENT_SCORES = {a: [] for a in AGENTS}
AGENT_HISTORY = {a: [dict(AGENT_WEIGHTS[a])] for a in AGENTS}
AGENT_ENTROPIES = {a: [] for a in AGENTS}
AGENT_MOVEMENT = {a: [] for a in AGENTS}
GROUP_DIVERGENCE = []
GROUP_AVG_REWARDS = []

# ==================== Simulation Logic ====================
def simulate():
    chosen_scenario = SCENARIOS[scenario]

    for _ in range(steps):
        for a in AGENTS:
            prev = list(AGENT_WEIGHTS[a].values())

            # 1️⃣ 현재 가치 weight 벡터
            weights_list = list(AGENT_WEIGHTS[a].values())

            # 2️⃣ 선택지 A/B에 대한 점수 계산 (내적)
            scoreA = float(np.dot(weights_list, chosen_scenario["A"]))
            scoreB = float(np.dot(weights_list, chosen_scenario["B"]))
            choice = "A" if scoreA >= scoreB else "B"

            # 3️⃣ 선택된 보상 벡터
            reward = chosen_scenario[choice]
            keys = list(AGENT_WEIGHTS[a].keys())

            # 4️⃣ 보상에 비례해서 가중치 업데이트
            for i, k in enumerate(keys):
                AGENT_WEIGHTS[a][k] += 0.05 * reward[i]

            # 5️⃣ 정규화 및 기록
            AGENT_WEIGHTS[a] = normalize(AGENT_WEIGHTS[a])
            curr = list(AGENT_WEIGHTS[a].values())

            AGENT_HISTORY[a].append(dict(AGENT_WEIGHTS[a]))
            AGENT_SCORES[a].append(float(np.dot(weights_list, reward)))
            AGENT_ENTROPIES[a].append(entropy(curr))
            AGENT_MOVEMENT[a].append(
                np.linalg.norm(np.array(curr) - np.array(prev))
            )

        # 6️⃣ 그룹 차원 지표
        if len(AGENTS) > 1:
            mat = np.array([list(AGENT_WEIGHTS[a].values()) for a in AGENTS])
            GROUP_DIVERGENCE.append(float(np.mean(pdist(mat))))
        else:
            GROUP_DIVERGENCE.append(0.0)

        GROUP_AVG_REWARDS.append(
            float(np.mean([np.mean(AGENT_SCORES[a]) for a in AGENTS]))
        )

# ==================== Alerts ====================
def show_alerts():
    for a in AGENTS:
        if len(AGENT_ENTROPIES[a]) > 1:
            delta = AGENT_ENTROPIES[a][-2] - AGENT_ENTROPIES[a][-1]
            if delta > 0.1:
                st.warning(
                    f"⚠️ {a}: 전략이 급격히 한 방향으로 쏠리고 있습니다 (entropy ↓ {delta:.2f})"
                )

# ==================== Caption (고정 텍스트 캐싱) ====================
@st.cache_data(show_spinner=False)
def generate_caption():
    return {
        "fig1": "Figure 1: 문화권별 가치 차원(Emotion, Social, Identity, Moral) 가중치 궤적",
        "fig2": "Figure 2a: 엔트로피(전략 내부 다양성) 변화, 2b: 누적 전략 변화량",
        "fig3": "Figure 3a: 문화권 간 전략 분기 정도(Divergence), 3b: 평균 보상과의 상관관계",
    }

# ==================== (옵션) GPT 요약 ====================
def gpt_summary():
    try:
        openai.api_key = st.secrets.get("OPENAI_API_KEY")
        trend = pd.DataFrame(GROUP_DIVERGENCE).diff().mean().values[0]
        agents = list(AGENT_HISTORY.keys())
        prompt = (
            f"문화권 에이전트 {agents}가 시나리오 '{scenario}'에서 "
            f"A/B 전략 선택을 반복 학습한 결과를 요약해줘. "
            f"전략 다양성(엔트로피)와 보상(평균 보상)의 관계도 포함해서 5줄로 정리해줘."
        )
        out = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        st.info(out["choices"][0]["message"]["content"])
    except Exception as e:
        st.error(f"GPT 요약 실패: {e}")

# ==================== Run & Display ====================
if st.button("▶️ 시뮬레이션 시작"):
    if len(AGENTS) == 0:
        st.error("최소 1개 이상의 문화권을 선택하세요.")
    else:
        simulate()
        captions = generate_caption()

        # ---------- Figure 1: 가치 차원 궤적 ----------
        st.subheader("📊 " + captions["fig1"])
        for dim in ["emotion", "social", "identity", "moral"]:
            fig, ax = plt.subplots()
            for a in AGENT_HISTORY:
                ax.plot([w[dim] for w in AGENT_HISTORY[a]], label=a)
            ax.set_title(f"{dim.capitalize()} Weight Trajectory")
            ax.legend()
            st.pyplot(fig)

        # ---------- Figure 2: 엔트로피 / 누적 변화 ----------
        st.subheader("📈 " + captions["fig2"])

        # 2a. 엔트로피
        fig1, ax1 = plt.subplots()
        for a in AGENT_ENTROPIES:
            ax1.plot(AGENT_ENTROPIES[a], label=a)
        ax1.set_title("Entropy of Strategy Distribution")
        ax1.legend()
        st.pyplot(fig1)

        # 2b. 누적 변화량
        fig2, ax2 = plt.subplots()
        for a in AGENT_MOVEMENT:
            ax2.plot(np.cumsum(AGENT_MOVEMENT[a]), label=a)
        ax2.set_title("Cumulative Strategic Change")
        ax2.legend()
        st.pyplot(fig2)

        # ---------- Figure 3: Divergence & Reward ----------
        st.subheader("📉 " + captions["fig3"])

        # 3a. 그룹 Divergence
        fig3, ax3 = plt.subplots()
        ax3.plot(GROUP_DIVERGENCE, label="Ethical Divergence")
        ax3.set_title("Group Ethical Divergence Over Time")
        ax3.legend()
        st.pyplot(fig3)

        # 3b. Divergence vs Avg Reward
        fig4, ax4 = plt.subplots()
        ax4.scatter(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
        if len(GROUP_DIVERGENCE) > 1 and len(set(GROUP_DIVERGENCE)) > 1:
            r, p = pearsonr(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
            ax4.set_title(f"Divergence vs Avg Reward (r={r:.2f}, p={p:.3f})")
        else:
            ax4.set_title("Divergence vs Avg Reward (표본이 부족하거나 상수가 많음)")
        st.pyplot(fig4)

        # ---------- 최종 전략 요약 ----------
        st.subheader("📄 최종 전략 요약 (마지막 스텝 기준)")
        df = pd.DataFrame(
            [{"Agent": a, **AGENT_HISTORY[a][-1]} for a in AGENTS]
        )
        st.dataframe(df.set_index("Agent"))

        st.download_button(
            "📥 Save CSV",
            data=df.to_csv(index=False),
            file_name="final_strategies.csv"
        )

        # ---------- 경고 ----------
        st.subheader("📡 전략 분기 경고")
        show_alerts()

        # ---------- GPT 요약 (옵션 버튼) ----------
        if st.button("🧠 GPT로 결과 요약받기"):
            gpt_summary()
