import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.ma.core import swapaxes
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class InterestRateSwap:
    """Interest Rate Swap"""

    def __init__(self, notional, fixed_rate, floating_rate, maturity_years, payment_freq=2):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.floating_rate = floating_rate
        self.maturity_years = maturity_years
        self.payment_freq = payment_freq
        self.payments_per_year = payment_freq

    def current_mtm(self, discount_curve):
        """ Mark to Market"""
        time_points = np.arange(0.25, self.maturity_years + 0.25, 1/self.payment_freq)
        fixed_cf = self.notional * self.fixed_rate/self.payment_freq
        floating_cf = self.notional * self.floating_rate/self.payment_freq

        discount_factor = [np.exp(-r * t) for r, t in zip(discount_curve, time_points)]
        fixed_pv = sum([fixed_cf * factor for factor in discount_factor])
        floating_pv = sum([floating_cf * factor for factor in discount_factor])

        mtm = floating_pv - fixed_pv
        return mtm

class CCRCalculator:
    """counterparty credit risk"""
    def __init__(self, swap, counterparty_pd, recovery_rate):
        self.swap = swap
        self.counterparty_pd = counterparty_pd
        self.recovery_rate = recovery_rate
        self.lgd = 1 - recovery_rate

    @staticmethod
    def calculate_current_exposure(current_mtm):
        return max(current_mtm, 0)

    def simulate_future_exposures(self, time_horizons, n_simulations=10000):
        """Monte Carlo"""
        results = {}

        for horizon in time_horizons:
            # 简化的利率路径模拟 (使用Vasicek模型)
            dt = 0.25  # 时间间隔节点
            n_steps = int(horizon /dt)

            exposures = []
            for _ in range(n_simulations):
                # calc rate path
                rates = self.simulate_rate_path(n_steps, dt)
                # calc MTM based on the path
                mtm = self.calculate_mtm_at_time(rates[-1], horizon)
                exposure = max(mtm, 0)
                exposures.append(exposure)
            exposures = np.array(exposures)

            # key metrics for risk exposure
            results[horizon] = {
                'mean_exposure': np.mean(exposures),
                'max_exposure':np.max(exposures),
                'pfe_95':np.percentile(exposures, 95),
                'pfe_99': np.percentile(exposures, 99),
                'eepe': np.mean(exposures),
                'exposures':exposures
            }

        return results

    def simulate_rate_path(self, n_steps, dt):
        r0 = 0.05 # 初始利率
        alpha = 0.1  # 均值回归速度
        theta = 0.05  # 长期均值
        sigma = 0.02  # 波动率

        rates =[r0]
        for _ in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            dr = alpha *(theta -rates[-1]) *dt + sigma * dW
            rates.append(max(rates[-1]+dr, 0)) # 确保非负利率
        return rates

    def calculate_mtm_at_time(self, current_rate, time_to_maturity):
        """calc MTM at time"""
        remaining_years = self.swap.maturity_years - time_to_maturity
        if remaining_years <=0:
            return 0

        rate_diff = current_rate - self.swap.fixed_rate
        duration = remaining_years * 0.8 #  simplify duration
        mtm = self.swap.notional * rate_diff * duration

        return mtm

    def calculate_cva(self, exposure_profile):
        """calc credit value adjustments"""
        cva = 0
        for horizon, metrics in exposure_profile.items():
            time_pd = 1 -(1-self.counterparty_pd) ** horizon
            cva +=self.lgd * time_pd * metrics['mean_exposure']

        return cva

def main():
    # 1. create swap contract
    swap = InterestRateSwap(
        notional=10_000_000,
        fixed_rate=0.03,
        floating_rate=0.035,
        maturity_years=5
    )
    #2. calc MTM
    discount_curve =[0.035 , 0.036, 0.037, 0.038, 0.039] *4 # Simplify discount curve
    current_mtm = swap.current_mtm(discount_curve)
    print(f"MTM: ${current_mtm:,.2f}")

    #3. calc CCR
    ccr_calc = CCRCalculator(
        swap=swap, counterparty_pd=0.02, recovery_rate=0.4
    )

    #4. calc current exposures
    current_exposure = ccr_calc.calculate_current_exposure(current_mtm)
    print(f"current exposures(CE): ${current_exposure:,.2f}\n")

    #5. calc simulated exposures
    time_horizons = [0.5, 1.0, 2.0, 3.0, 5.0]  # 6个月到5年
    exposure_results = ccr_calc.simulate_future_exposures(time_horizons, n_simulations=10000)

    #6. metrics presenting
    results_df = pd.DataFrame()

    for horizon in time_horizons:
        metrics = exposure_results[horizon]
        results_df = pd.concat([results_df, pd.DataFrame({
            'Time(Yr)': [horizon],
            'mean_exposure': [f"${metrics['mean_exposure']:,.0f}"],
            '95% PFE': [f"${metrics['pfe_95']:,.0f}"],
            '99% PFE': [f"${metrics['pfe_99']:,.0f}"],
            'max_exposure': [f"${metrics['max_exposure']:,.0f}"]
        })], ignore_index=True)
    print(results_df.to_string(index=False))

    # 7. 计算CVA
    cva = ccr_calc.calculate_cva(exposure_results)
    print(f"\n (CVA): ${cva:,.2f}")

    # 8. charts
    fig, axes = plt.subplots(3,2, figsize=(15,10))
    fig.suptitle('credit risk analysis results', fontsize=16)

    # Monte Carlo模拟演示 - 显示实际模拟路径
    print("生成Monte Carlo模拟演示图...")
    n_demo_paths = 3000  # 用于演示的路径数量
    demo_horizon = 1.0  # 1年期演示
    demo_exposures = []
    demo_rate_paths = []

    for i in range(n_demo_paths):
        dt = 0.25
        n_steps = int(demo_horizon / dt)
        rates = ccr_calc.simulate_rate_path(n_steps, dt)
        demo_rate_paths.append(rates)

        mtm = ccr_calc.calculate_mtm_at_time(rates[-1], demo_horizon)
        exposure = max(mtm, 0)
        demo_exposures.append(exposure)

    # 绘制利率路径演示
    time_steps = np.linspace(0, demo_horizon, len(demo_rate_paths[0]))

    # 显示前20条路径
    for i in range(min(200, n_demo_paths)):
        axes[0, 0].plot(time_steps, demo_rate_paths[i], alpha=0.2, color='blue', linewidth=0.8)

    # 显示平均路径
    mean_path = np.mean(demo_rate_paths, axis=0)
    axes[0, 0].plot(time_steps, mean_path, 'r-', linewidth=3, label='avg path')

    # 显示置信区间
    upper_95 = np.percentile(demo_rate_paths, 97.5, axis=0)
    lower_95 = np.percentile(demo_rate_paths, 2.5, axis=0)
    axes[0, 0].fill_between(time_steps, lower_95, upper_95, alpha=0.2, color='red', label='95% p')

    axes[0, 0].set_xlabel('Time(yr)')
    axes[0, 0].set_ylabel('rate')
    axes[0, 0].set_title('Vasicek model - Monte Carlo')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Monte Carlo暴露分布演示
    axes[0, 1].scatter(range(len(demo_exposures)), demo_exposures, alpha=0.1, s=10, color='blue')
    axes[0, 1].axhline(y=np.mean(demo_exposures), color='red', linestyle='-', linewidth=2,
                       label=f'Mean: ${np.mean(demo_exposures):,.0f}')
    axes[0, 1].axhline(y=np.percentile(demo_exposures, 95), color='green', linestyle='--', linewidth=2,
                       label=f'95%pct: ${np.percentile(demo_exposures, 95):,.0f}')
    axes[0, 1].set_xlabel('n_simulation')
    axes[0, 1].set_ylabel('Exposure ($)')
    axes[0, 1].set_title(f'Monte Carlo result (1 Yr, N={n_demo_paths})')
    axes[0, 1].legend()
    axes[0, 1].grid(True)



    # PFE时间序列
    times = list(time_horizons)
    pfe_95 = [exposure_results[t]['pfe_95'] for t in times]
    pfe_99 = [exposure_results[t]['pfe_99'] for t in times]
    mean_exp = [exposure_results[t]['mean_exposure'] for t in times]

    axes[1, 0].plot(times, mean_exp, 'b-o', label='mean_exposure')
    axes[1, 0].plot(times, pfe_95, 'r-s', label='95% PFE')
    axes[1, 0].plot(times, pfe_99, 'g-^', label='99% PFE')
    axes[1, 0].set_xlabel('time(yr)')
    axes[1, 0].set_ylabel('exposure ($)')
    axes[1, 0].set_title('future exposures')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 1年期暴露分布
    exposures_1y = exposure_results[1.0]['exposures']
    axes[1, 1].hist(exposures_1y, bins=50, alpha=0.7, density=True)
    axes[1, 1].axvline(np.percentile(exposures_1y, 95), color='r', linestyle='--', label='95% PFE')
    axes[1, 1].axvline(np.percentile(exposures_1y, 99), color='g', linestyle='--', label='99% PFE')
    axes[1, 1].set_xlabel('exposure ($)')
    axes[1, 1].set_ylabel('histgram')
    axes[1, 1].set_title('1yr exposure distribution')
    axes[1, 1].legend()

    # 不同时间点的分位数对比
    percentiles = [50, 75, 90, 95, 99]
    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for i, p in enumerate(percentiles):
        values = [np.percentile(exposure_results[t]['exposures'], p) for t in times]
        axes[2, 0].plot(times, values, 'o-', color=colors[i], label=f'{p}%pct')

    axes[2, 0].set_xlabel('Time(yr)')
    axes[2, 0].set_ylabel('exposure ($)')
    axes[2, 0].set_title('exposure pct mvt')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # CVA构成
    cva_components = []
    for horizon in time_horizons:
        metrics = exposure_results[horizon]
        time_pd = 1 - (1 - ccr_calc.counterparty_pd) ** horizon
        component = ccr_calc.lgd * time_pd * metrics['mean_exposure']
        cva_components.append(component)

    axes[2, 1].bar(range(len(times)), cva_components)
    axes[2, 1].set_xlabel('Time Horizon')
    axes[2, 1].set_ylabel('CVA contribution ($)')
    axes[2, 1].set_title('horizon vs CVA contribution')
    axes[2, 1].set_xticks(range(len(times)))
    axes[2, 1].set_xticklabels([f'{t}Y' for t in times])

    plt.tight_layout()
    plt.show()

    # 9. 风险报告总结
    print(f"\n=== Summary ===")
    print(f"• max95% PFE: ${max(pfe_95):,.0f} (at {times[pfe_95.index(max(pfe_95))]} Yr)")
    print(f"• est. peak exposure: ${max(mean_exp):,.0f}")
    print(f"• Total CVA cost: ${cva:,.2f}")
    print(f"• CVA against notional: {cva / swap.notional:.4%}")

if __name__ == '__main__':
    main()
