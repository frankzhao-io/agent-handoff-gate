# 文獻查重與差異化定位（2026-11-16）

## 一、2025–2026 核心相關工作

| 論文／基準 | 年份／會議 | 核心機制 | 我們的差異化缺口（Gap） |
| :--- | :--- | :--- | :--- |
| **MAST** (UC Berkeley) | 2025–2026 | 分析 1,600+ 多 Agent 軌跡，建立結構/對齊/驗證失敗分類學 | 屬事後診斷分類，未提出執行期阻斷機制 |
| **AgentProp-Bench** | 2026 | 分階段拆解工具呼叫與規劃錯誤的傳播鏈與放大效應 | 屬評測基準，未設計雙向歸因閘門 |
| **ToolGate** | ACL 2026 | 以 Hoare-style 前置/後置條件驗證單一工具呼叫與符號狀態 | 屬**單向（Forward-only）**閘門，未區分跨 Agent 的 `ASSUMED_*` vs `VERIFIED_*` 與逆向回滾 |
| **SWE-bench Multi-Agent Empirical Studies** | 2025–2026 | 實證無約束 sub-agent 導致成功率下降（83%→59%）、Token 膨脹 1.4x–3.3x 與 False Consensus | 指出問題與代價，尚未以雙向知識狀態閘門解決 |

## 二、設計原則（來自獨立審稿人 Subagent 壓力測試）
1. **避開玩具基準**：不使用 `HumanEval+ / MBPP` 單函式題；改用具備隱藏狀態的**多步驟表格/資料分析管線（`DA-Code` / `InfiAgent-DABench` 風格）**＋ **可控錯誤注入壓力測試（Corrupted-Plan Stress Test）**。
2. **必須證明「雙向歸因」優於「單向閘門」**：必須與 `Single-Agent + Reflexion`（同 Token 預算）及 `Forward-Only Gate (ToolGate baseline)` 做嚴格消融。
3. **核心指標**：`Pass@1`、`False Consensus Rate (FCR)`、`Error Propagation Rate`、`Token Cost Pareto Front`。
