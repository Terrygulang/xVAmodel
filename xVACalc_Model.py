from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.ma.core import negative
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CounterpartyProfile:
    """counterparty profile"""
    name: str
    credit_rating: str
    pd_curve: List[float]
    recovery_rate: float
    funding_spread: float

@dataclass
class TradePortfolio:
    trades: List[Dict]
    netting_agreement: bool=True
    margin_agreement: bool=False
    initial_margin: float =0
    variation_margin: float=0

class xVACalculator:
    def __init__(self,bank_pd_curve, bank_recovery_rate, risk_free_curve):
        self.bank_pd_curve= bank_pd_curve
        self.bank_recovery_rate = bank_recovery_rate
        self.risk_free_curve = risk_free_curve
        self.time_grid = np.array([0.5, 1, 2, 3, 5, 7, 10])

    def calculate_exposure_profile(self, portfolio: TradePortfolio,
                                   counterparty: CounterpartyProfile,
                                   n_simulations: int = 10000) -> Dict:
        """计算暴露分布"""

        results = {}

        for i, time in enumerate(self.time_grid):
            exposures = []

            # Monte Carlo模拟
            for _ in range(n_simulations):
                portfolio_mtm = 0

                # 模拟每个交易的MTM
                for trade in portfolio.trades:
                    mtm = self._simulate_trade_mtm(trade, time)
                    portfolio_mtm += mtm

                # 考虑净额结算
                if portfolio.netting_agreement:
                    gross_exposure = max(portfolio_mtm, 0)
                else:
                    gross_exposure = sum([max(self._simulate_trade_mtm(t, time), 0)
                                          for t in portfolio.trades])

                # 考虑保证金
                net_exposure = max(gross_exposure - portfolio.variation_margin, 0)
                exposures.append(net_exposure)

            exposures = np.array(exposures)

            results[time] = {
                'expected_exposure': np.mean(exposures),
                'pfe_95': np.percentile(exposures, 95),
                'pfe_99': np.percentile(exposures, 99),
                'expected_negative_exposure': np.mean(np.minimum(exposures, 0)),  # 用于DVA
                'exposures': exposures
            }

        return results

    def _simulate_trade_mtm(self, trade: Dict, time: float) -> float:
        """MTM per trade"""
        trade_type = trade['type']
        notional = trade['notional']
        maturity = trade['maturity']

        if time >= maturity:
            return 0

        if trade_type == 'interest_rate_swap':
            rate_shock = np.random.normal(0, 0.01 * np.sqrt(time))
            duration = (maturity - time) *0.8
            mtm = notional * rate_shock * duration

        elif trade_type == 'fx_forward':
            fx_shock = np.random.normal(0, 0.15 * np.sqrt(time))
            mtm = notional * fx_shock

        elif trade_type == 'credit_default_swap':
            spread_shock = np.random.normal(0, 0.005 * np.sqrt(time))
            mtm = notional * spread_shock * (maturity - time)

        else:
            mtm = np.random.normal(0, notional * 0.1 * np.sqrt(time))

        return mtm

    def calculate_cva(self, exposure_profile: Dict,
                      counterparty: CounterpartyProfile,
                      ):
        cva = 0
        lgd = 1 - counterparty.recovery_rate

        for i, time in enumerate(self.time_grid):
            if i ==0:
                dt=time
                survival_prob_start = 1.0
            else:
                dt = time - self.time_grid[i-1]
                survival_prob_start = 1 - counterparty.pd_curve[i-1]

            survival_prob_end = 1 - counterparty.pd_curve[i]
            default_prob = survival_prob_start - survival_prob_end

            expected_exposure = exposure_profile[time]['expected_exposure']
            discount_factor = np.exp(-self.risk_free_curve[i] * time)

            cva += lgd * default_prob * expected_exposure * discount_factor
        return cva

    def calculate_dva(self, exposure_profile:Dict,
                      counterparty: CounterpartyProfile):
        dva = 0
        lgd_bank = 1- self.bank_recovery_rate

        for i, time in enumerate(self.time_grid):
            if i ==0:
                survival_prob_start = 1.0
            else:
                survival_prob_start = 1- self.bank_pd_curve[i-1]

            survival_prob_end = 1 - self.bank_pd_curve[i]
            default_prob = survival_prob_start - survival_prob_end

            # DVA使用负暴露（银行对对手方的负债）
            negative_exposure = abs(exposure_profile[time]['expected_negative_exposure'])
            discount_factor = np.exp(-self.risk_free_curve[i] * time)

            dva += lgd_bank * default_prob * negative_exposure * discount_factor
        return dva

    def calculate_fva(self, exposure_profile: Dict,
                      counterparty:CounterpartyProfile):
        fca = 0 # Funding Cost Adjustments
        fba = 0 # Funding Benefit Adjustments

        for i, time in enumerate(self.time_grid):
            expected_exposure = exposure_profile[time]['expected_exposure']
            negative_exposure = exposure_profile[time]['expected_negative_exposure']
            discount_factor = np.exp(-self.risk_free_curve[i] * time)

            # 融资成本（正暴露需要融资）
            fca += counterparty.funding_spread * expected_exposure * discount_factor

            # 融资收益（负暴露投资获利）
            fba += counterparty.funding_spread * negative_exposure * discount_factor

        return {'FCA': fca, 'FBA': fba, 'FVA': fca-fba}

    def calculate_kva(self, exposure_profile: Dict, capital_requirement_rate: float = 0.12) -> float:
        """计算资本价值调整 KVA (简化版本)"""
        kva = 0
        cost_of_capital = 0.15  # 15%的资本成本

        for i, time in enumerate(self.time_grid):
            # 使用EE作为资本需求的代理
            capital_required = exposure_profile[time]['expected_exposure'] * capital_requirement_rate
            discount_factor = np.exp(-self.risk_free_curve[i] * time)

            if i == 0:
                dt = time
            else:
                dt = time - self.time_grid[i - 1]

            kva += cost_of_capital * capital_required * dt * discount_factor

        return kva


def create_sample_data():
    """创建示例数据"""

    # 对手方信用档案
    counterparties = {
        'AAA_Bank': CounterpartyProfile(
            name='AAA ranking Bank',
            credit_rating='AAA',
            pd_curve=[0.001, 0.002, 0.005, 0.008, 0.015, 0.025, 0.040],  # 累积PD
            recovery_rate=0.60,
            funding_spread=0.0050
        ),
        'BBB_Corp': CounterpartyProfile(
            name='BBB ranking Bank',
            credit_rating='BBB',
            pd_curve=[0.005, 0.012, 0.025, 0.040, 0.070, 0.100, 0.150],
            recovery_rate=0.40,
            funding_spread=0.0150
        )
    }

    # 交易组合
    portfolio = TradePortfolio(
        trades=[
            {
                'type': 'interest_rate_swap',
                'notional': 50_000_000,
                'maturity': 5,
                'fixed_rate': 0.025,
                'description': '5 yr int swap'
            },
            {
                'type': 'fx_forward',
                'notional': 20_000_000,
                'maturity': 2,
                'strike': 1.35,
                'description': 'EUR/USD Forward'
            },
            {
                'type': 'credit_default_swap',
                'notional': 10_000_000,
                'maturity': 3,
                'spread': 0.0200,
                'description': 'CDS'
            }
        ],
        netting_agreement=True,
        margin_agreement=True,
        variation_margin=1_000_000
    )

    return counterparties, portfolio


def main():
    print("=== xVA Case Analysis ===\n")

    # 创建示例数据
    counterparties, portfolio = create_sample_data()

    # 银行自身参数和市场数据
    bank_pd_curve = [0.002, 0.004, 0.008, 0.012, 0.020, 0.030, 0.045]
    bank_recovery_rate = 0.40
    risk_free_curve = [0.015, 0.018, 0.022, 0.025, 0.028, 0.030, 0.032]

    # 创建xVA计算器
    xva_calc = xVACalculator(bank_pd_curve, bank_recovery_rate, risk_free_curve)

    # 计算结果汇总
    results_summary = []

    for cp_name, counterparty in counterparties.items():
        print(f"\n=== counterparty: {counterparty.name} ({counterparty.credit_rating}) ===")

        # 计算暴露分布
        print("exposure calculation...")
        exposure_profile = xva_calc.calculate_exposure_profile(portfolio, counterparty, n_simulations=5000)

        # 计算各种xVA
        cva = xva_calc.calculate_cva(exposure_profile, counterparty)
        dva = xva_calc.calculate_dva(exposure_profile, counterparty)
        fva_components = xva_calc.calculate_fva(exposure_profile, counterparty)
        kva = xva_calc.calculate_kva(exposure_profile)

        # 总xVA
        total_xva = cva - dva + fva_components['FVA'] + kva

        # 结果展示
        print(f"\nxVA:")
        print(f"CVA (Credit Value Adjustments): ${cva:,.0f}")
        print(f"DVA (Debit Value Adjustments): ${dva:,.0f}")
        print(f"FCA (Funding Cost Adjustments): ${fva_components['FCA']:,.0f}")
        print(f"FBA (Funding Benefit Adjustments): ${fva_components['FBA']:,.0f}")
        print(f"FVA (Funding Value Adjustments): ${fva_components['FVA']:,.0f}")
        print(f"KVA (Capital Value Adjustments): ${kva:,.0f}")
        print(f"Total xVA: ${total_xva:,.0f}")

        # 保存结果用于对比
        results_summary.append({
            'counterparty': counterparty.name,
            'credit rating': counterparty.credit_rating,
            'CVA': cva,
            'DVA': dva,
            'FVA': fva_components['FVA'],
            'KVA': kva,
            'Total xVA': total_xva,
            'exposure profile': exposure_profile
        })

    # 结果对比分析
    print("\n" + "=" * 60)
    print("xVA comparison analysis")
    print("=" * 60)

    df_summary = pd.DataFrame([
        {
            'counterparty': r['counterparty'],
            'credit rating': r['credit rating'],
            'CVA': f"${r['CVA']:,.0f}",
            'DVA': f"${r['DVA']:,.0f}",
            'FVA': f"${r['FVA']:,.0f}",
            'KVA': f"${r['KVA']:,.0f}",
            'Total xVA': f"${r['Total xVA']:,.0f}"
        }
        for r in results_summary
    ])

    print(df_summary.to_string(index=False))

    # 可视化分析
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('xVA results', fontsize=16)

    # 1. xVA组成对比
    categories = ['CVA', 'DVA', 'FVA', 'KVA']
    x = np.arange(len(categories))
    width = 0.35

    aaa_values = [results_summary[0]['CVA'], -results_summary[0]['DVA'],
                  results_summary[0]['FVA'], results_summary[0]['KVA']]
    bbb_values = [results_summary[1]['CVA'], -results_summary[1]['DVA'],
                  results_summary[1]['FVA'], results_summary[1]['KVA']]

    axes[0, 0].bar(x - width / 2, aaa_values, width, label='AAA Bank', alpha=0.8)
    axes[0, 0].bar(x + width / 2, bbb_values, width, label='BBB Bank', alpha=0.8)
    axes[0, 0].set_xlabel('xVA components')
    axes[0, 0].set_ylabel('Amount ($)')
    axes[0, 0].set_title('Counter party xVA vs.')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # # 9. 风险报告总结
    # print(f"\n=== 风险管理总结 ===")
    # print(f"• 最大95% PFE: ${max(pfe_95):,.0f} (发生在{times[pfe_95.index(max(pfe_95))]}年)")
    # print(f"• 预期平均暴露峰值: ${max(mean_exp):,.0f}")
    # print(f"• 总CVA成本: ${cva:,.2f}")
    # print(f"• CVA占名义本金比例: {cva / swap.notional:.4%}")


if __name__ == "__main__":
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
