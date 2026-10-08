#!/usr/bin/env python3
"""Plot and summarize saved multifield ensemble/theory comparisons (no reruns)."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tomllib


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_toml(path):
    with path.open('rb') as f:
        return tomllib.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    root = args.batch.resolve()
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
    import numpy as np
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    env = read_toml(root / 'environment.toml')
    theory_check = read_toml(root / 'theory_diagnostics.toml')
    dirs = sorted(root.glob('trial_*'))
    statuses = [read_toml(d/'status.toml') for d in dirs]
    config = read_toml(dirs[0]/'config.toml')
    rows = read_csv(root / 'spectra_comparison.csv')
    numeric = ('efolds','k_bin','n_modes','k_comoving','observed','theory_mean',
               'theory_sigma','conditioned_prediction','prediction_with_lattice_projection',
               'residual_fraction','ensemble_ratio')
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        row['trial'] = int(row['trial'])
    ntrials = len(dirs)
    plt.rcParams.update({'font.size':10,'figure.dpi':120,'savefig.dpi':180,
                         'axes.grid':True,'grid.alpha':.2})
    label = 'SHORT SOFTWARE TEST' if env['smoke'] else 'INFLATION END'
    quantities = ('phi','chi','zeta_linear')
    aggregate = []
    for snapshot in dict.fromkeys(r['snapshot'] for r in rows):
        for quantity in quantities:
            selected = [r for r in rows if r['snapshot']==snapshot and r['quantity']==quantity]
            for binid in sorted({r['k_bin'] for r in selected}):
                group = [r for r in selected if r['k_bin']==binid]
                assert len(group) == ntrials
                observed = np.array([r['observed'] for r in group])
                conditioned = np.array([r['conditioned_prediction'] for r in group])
                sigma_mean = np.sqrt(sum(r['theory_sigma']**2 for r in group))/ntrials
                expected = np.mean([r['theory_mean'] for r in group])
                obsvar = np.var(observed, ddof=1) if ntrials>1 else 0.
                aggregate.append(dict(snapshot=snapshot,quantity=quantity,k_bin=int(binid),
                    k_comoving=group[0]['k_comoving'],n_modes=int(group[0]['n_modes']),
                    conservative_band=group[0]['conservative_band'],
                    observed_mean=observed.mean(),observed_std=observed.std(ddof=1) if ntrials>1 else 0.,
                    theory_mean=expected,theory_sigma_of_mean=sigma_mean,
                    ensemble_sigma_deviation=(observed.mean()-expected)/sigma_mean,
                    mean_residual_percent=100*np.mean([r['residual_fraction'] for r in group]),
                    min_residual_percent=100*min(r['residual_fraction'] for r in group),
                    max_residual_percent=100*max(r['residual_fraction'] for r in group),
                    min_explained_variance_fraction=(1-np.var(observed-conditioned,ddof=1)/obsvar) if obsvar>0 else float('nan'),
                    correlation=np.corrcoef(observed,conditioned)[0,1] if obsvar>0 else float('nan')))
    write_csv(root/'spectrum_seed_attribution.csv',aggregate)
    colors = plt.get_cmap('tab10')

    def power_panel(ax, quantity, snapshot='final'):
        selected = [r for r in rows if r['quantity']==quantity and r['snapshot']==snapshot]
        for trial in range(1,ntrials+1):
            g=sorted((r for r in selected if r['trial']==trial),key=lambda r:r['k_bin'])
            ax.loglog([r['k_comoving'] for r in g],[r['observed'] for r in g],
                       marker='o',ms=3,lw=.8,color=colors((trial-1)%10))
            ax.loglog([r['k_comoving'] for r in g],[r['conditioned_prediction'] for r in g],
                       ls='--',lw=.9,color=colors((trial-1)%10))
        a=sorted((r for r in aggregate if r['quantity']==quantity and r['snapshot']==snapshot),key=lambda r:r['k_bin'])
        ax.loglog([r['k_comoving'] for r in a],[r['theory_mean'] for r in a],color='black',lw=1.6)
        ax.set(xlabel=r'$k_{\rm lat}$ (comoving, Planck units)',ylabel='Dimensionless power')
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1.,2.,5.)))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(handles=[Line2D([],[],color='gray',marker='o',ms=3,label='Lattice: each trial'),
                            Line2D([],[],color='gray',ls='--',label='Linear: same initial arrays'),
                            Line2D([],[],color='black',label='Linear: ensemble expectation')],fontsize=8)

    def residual_panel(ax, quantity, snapshot='final'):
        selected=[r for r in rows if r['quantity']==quantity and r['snapshot']==snapshot]
        for trial in range(1,ntrials+1):
            g=sorted((r for r in selected if r['trial']==trial),key=lambda r:r['k_bin'])
            ax.plot([r['k_comoving'] for r in g],[100*r['residual_fraction'] for r in g],
                    marker='o',ms=2,lw=.8,color=colors((trial-1)%10),label=f'Trial {trial}')
        ax.axhline(0,color='black',ls=':',lw=1)
        ax.set(xlabel=r'$k_{\rm lat}$ (comoving)',ylabel='100 × (lattice / same-draw linear − 1) [%]')

    def mean_panel(ax, quantity, snapshot='final'):
        a=sorted((r for r in aggregate if r['quantity']==quantity and r['snapshot']==snapshot),key=lambda r:r['k_bin'])
        x=np.array([r['k_comoving'] for r in a]);expected=np.array([r['theory_mean'] for r in a])
        width=np.array([r['theory_sigma_of_mean'] for r in a])/expected
        ax.fill_between(x,1-width,1+width,color='gray',alpha=.25,label='Theory ±1σ of mean')
        ax.errorbar(x,[r['observed_mean']/r['theory_mean'] for r in a],
                    yerr=[r['observed_std']/np.sqrt(ntrials)/r['theory_mean'] for r in a],
                    marker='o',ms=3,lw=1,capsize=2,label='Trial mean ± sample SEM')
        ax.axhline(1,color='black',ls=':',lw=1)
        ax.set(xlabel=r'$k_{\rm lat}$ (comoving)',ylabel='Mean lattice / expected power')
        ax.legend(fontsize=8)

    def save(fig,name):
        fig.savefig(figures/(name+'.png'));fig.savefig(figures/(name+'.pdf'));plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    power_panel(axes[0,0],'zeta_linear');axes[0,0].set_title('Final curvature power: each trial')
    mean_panel(axes[0,1],'zeta_linear');axes[0,1].set_title(f'Mean of {ntrials} trials vs expectation')
    residual_panel(axes[1,0],'zeta_linear');axes[1,0].set_title('Residual using the actual initial arrays')
    axes[1,0].legend(fontsize=7,ncol=2)
    decomp=read_csv(root/'curvature_decomposition.csv')
    g=sorted((r for r in decomp if r['snapshot']=='final' and int(r['trial'])==1),key=lambda r:int(r['k_bin']))
    bins=[int(r['k_bin']) for r in g]
    for key,title,color in [('observed_phi',r'$c_\phi^2 P_{\phi\phi}$','#b23a48'),
                             ('observed_chi',r'$c_\chi^2 P_{\chi\chi}$','#287d69'),
                             ('observed_cross_twice',r'$2c_\phi c_\chi P_{\phi\chi}$','#7653a3')]:
        axes[1,1].plot(bins,[float(r[key])/float(r['observed_total']) for r in g],marker='o',ms=3,label=title,color=color)
    axes[1,1].set(xlabel='Wave-number bin',ylabel='Contribution / total power',title='Curvature decomposition (trial 1)')
    axes[1,1].legend(fontsize=9)
    fig.suptitle(f'Two-field curvature: lattice vs no-metric linear theory | {label}\n'
                 f'{ntrials} distinct seeds; final N ≈ {statuses[0]["final_efolds"]:.5f}',fontsize=14)
    save(fig,'curvature_power_comparison')

    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for q,ax in zip(('phi','chi'),axes):
        power_panel(ax[0],q);ax[0].set_title(f'Final field {q} power')
        residual_panel(ax[1],q);ax[1].set_title(f'Field {q}: residual for the same initial arrays')
    fig.suptitle('Field powers | no metric perturbations')
    save(fig,'field_power_comparison')

    snapshots=list(dict.fromkeys(r['snapshot'] for r in rows))
    fig,axes=plt.subplots(len(snapshots),2,figsize=(12,3.1*len(snapshots)),squeeze=False,constrained_layout=True)
    for snap,ax in zip(snapshots,axes):
        mean_panel(ax[0],'zeta_linear',snap);residual_panel(ax[1],'zeta_linear',snap)
        ax[0].set_title(f'{snap}: curvature ensemble mean')
        ax[1].set_title(f'{snap}: residual for the same initial arrays')
    save(fig,'curvature_comparison_over_time')

    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    all_stats=[]
    for trial,d in enumerate(dirs,1):
        for r in read_csv(d/'initial_statistics.csv'):all_stats.append(dict(trial=trial,**r))
        hist=read_csv(d/'initial_histograms.csv')
        for qi,q in enumerate(('phi','chi')):
            for ci,quantity in enumerate(('mode_real_z','mode_imag_z','mode_power_ratio')):
                g=[r for r in hist if r['field']==q and r['quantity']==quantity]
                x=np.array([(float(r['left'])+float(r['right']))/2 for r in g])
                axes[qi,ci].plot(x,[float(r['density']) for r in g],alpha=.5,lw=.8,color=colors((trial-1)%10))
                if trial==1:
                    theory=np.exp(-x) if ci==2 else np.exp(-x*x/2)/np.sqrt(2*np.pi)
                    axes[qi,ci].plot(x,theory,'k--',lw=2,label='Expected density')
                axes[qi,ci].set(title=f'{q}: {quantity}',ylabel='Probability density')
    axes[0,0].legend(fontsize=8);save(fig,'initial_random_statistics')
    write_csv(root/'initial_statistics_all.csv',all_stats)

    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    trial_rows=[]
    for trial,s in enumerate(statuses,1):
        bg=read_csv(Path(s['run_dir'])/'background.csv')
        sr=read_csv(Path(s['run_dir'])/'slowroll.csv')
        diag=read_csv(Path(s['run_dir'])/'diagnostics.csv')
        for ax,col in zip(axes.flat,('phi_mean','chi_mean','H','epsilon_H')):
            series=sr if col=='epsilon_H' else bg
            ax.plot([float(r['efolds']) for r in series],[float(r[col]) for r in series],lw=.8,label=f'Trial {trial}')
            ax.set(xlabel='N (e-folds)',ylabel=col)
        trial_rows.append(dict(trial=trial,seed=s['seed'],end_reason=s['end_reason'],final_efolds=s['final_efolds'],
            steps=s['steps'],elapsed_seconds=s['elapsed_seconds'],
            max_saved_friedmann_residual=max(float(r['friedmann_residual']) for r in diag)))
    axes[1,1].axhline(1,color='black',ls=':');axes[0,0].legend(fontsize=7,ncol=2)
    save(fig,'background_evolution');write_csv(root/'trials.csv',trial_rows)

    final=[r for r in rows if r['snapshot']=='final']
    finala=[r for r in aggregate if r['snapshot']=='final']
    summaries=[]
    for quantity in quantities:
        for scope in ('all_bins','conservative_bins'):
            group=[r for r in final if r['quantity']==quantity and (scope=='all_bins' or r['conservative_band']=='true')]
            ag=[r for r in finala if r['quantity']==quantity and (scope=='all_bins' or r['conservative_band']=='true')]
            summaries.append(dict(quantity=quantity,scope=scope,
                min_residual_percent=100*min(r['residual_fraction'] for r in group),
                max_residual_percent=100*max(r['residual_fraction'] for r in group),
                max_abs_ensemble_sigma_deviation=max(abs(r['ensemble_sigma_deviation']) for r in ag),
                min_explained_variance_percent=100*min(r['min_explained_variance_fraction'] for r in ag),
                min_correlation=min(r['correlation'] for r in ag)))
    write_csv(root/'comparison_summary.csv',summaries)
    metadata={'runs':ntrials,'label':label,'summary':summaries,'theory_checks':theory_check,
              'source_csv_sha256':hashlib.sha256((root/'spectra_comparison.csv').read_bytes()).hexdigest()}
    (root/'analysis_summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    table='\n'.join(f"|{r['quantity']}|{'全ビン' if r['scope']=='all_bins' else '保守的範囲'}|"
                    f"{r['min_residual_percent']:+.5g}～{r['max_residual_percent']:+.5g}%|"
                    f"{r['max_abs_ensemble_sigma_deviation']:.3f}σ|{r['min_explained_variance_percent']:.6f}%|" for r in summaries)
    report=rf'''# 二場二次ポテンシャル：乱数反復と曲率パワーの検証

## 計算条件

- V = (m²φ² + M²χ²)/2、m=9×10⁻⁶、M=10⁻⁶、還元Planck単位。
- φ₀=χ₀=13、両方の初期宇宙時間速度=0、a₀=1。
- 格子{config['lattice']['size']}³、内部箱長{config['lattice']['box_size']}、周期境界、7点差分ラプラシアン。
- 異なるシード{ntrials}試行、親シード `{env['master_seed']}`、場ごとにも別の乱数列。
- 真空揺らぎ有効、初期周波数の質量項なし、δN無効、計量摂動なし。
- leapfrog、timestep_factor={config['evolution']['timestep_factor']}。本番の終了は各試行でε_Hが初めて1に達したステップ。
- 本文の状態：**{label}**。短い動作試験の場合、終了N≈0.02へ変更され、本番検証ではありません（格子数は上記の値）。
- 終了N：{min(s['final_efolds'] for s in statuses):.10f}～{max(s['final_efolds'] for s in statuses):.10f}。
- 保存点でのFriedmann残差の最大：{max(r['max_saved_friedmann_residual'] for r in trial_rows):.4e}。

## 曲率パワーの比較

[曲率パワー比較PNG]({figures/'curvature_power_comparison.png'}) / [PDF]({figures/'curvature_power_comparison.pdf'})

左上は各試行（実線・点）、その実際の初期配列からの線形予測（同色破線）、初期分布の理論期待値（黒線）。
右上は{ntrials}試行平均と理論期待値の比。誤差棒は試行間の標準偏差/√試行数、灰色帯は理論上の平均の±1σです。
左下は初期配列に条件付けた線形予測からの残差（%）。右下は試行1の最終曲率パワーの各場・相互相関の寄与。

|物理量|波数範囲|同じ初期配列の予測からの残差|試行平均の期待値からの最大偏差|試行間分散の説明割合の最小|
|---|---|---:|---:|---:|
{table}

保守的範囲は既往検証に合わせ2≤k_bin≤8（8³動作試験では≤2）。全ビンも除外せず掲載しています。
端のビンはモード数が少なく、1σ帯からの逸脱だけで失敗とは判定できません。帯は各ビンの幅であり、同時信頼区間ではありません。
分散の説明割合は `1−Var(P_lattice−P_conditioned)/Var(P_lattice)`。各ビンで試行間平均を引いた変動を評価し、共通バイアスとは区別します。
相関が高くても残差がゼロとは限りません。数値結果は [要約CSV]({root/'comparison_summary.csv'}) に保存しています。

## 同じ近似に合わせた理論

各場の線形モードについて、独立に計算した一様背景上で

$$Q_{{I,NN}}+(3-epsilon_H)Q_{{I,N}}+left[rac{{k_{{lat}}^2}}{{a^2H^2}}+rac{{m_I^2}}{{H^2}}ight]Q_I=0$$

を数値積分しました。これは時空の揺らぎを省いた場の摂動方程式です。
標準の多場Mukhanov–Sasaki方程式にある計量由来の混合項は含めません。
初期の場と時間微分の両方を各モードで伝播し、殻平均初期パワーを倍率補正するだけの近似は使いません。
各波数ビンを代表する一波数への置き換えもせず、格子に実在する{theory_check['distinct_lattice_momenta']}種類の対称性で分類した有効波数を用いています。

曲率は格子コードの定義に合わせ、u_I=dφ̄_I/dN として

$$zeta_{{lin}}=-rac{{u_phi Q_phi+u_chi Q_chi}}{{u_phi^2+u_chi^2}},qquad
P_zeta=c_phi^2P_{{phiphi}}+c_chi^2P_{{chichi}}+2c_phi c_chi P_{{phichi}}.$$

分離型ポテンシャル・独立初期場では相互相関の**アンサンブル期待値**は0ですが、有限実現では0とは限りません。
条件付き予測にはこの実現ごとの相互相関を含めています。初期速度0の瞬間は曲率の分母が0のため比較対象から外しています。
このζ_linは格子コードの線形射影量です。計量摂動を解いた物理的曲率や、非線形δNと同一視しません。

## 数値確認と解釈の範囲

- 全試行のシードおよび両場・両時間微分の初期ハッシュを照合。乱数を再生成した実現を使っています。
- 理論の刻み半減によるパワーの最大相対変化：期待値 {theory_check['max_mean_power_relative_change']:.3e}、条件付き {theory_check['max_conditioned_power_relative_change']:.3e}。
- 格子の曲率パワーと「各場自己項＋2×交差項」の恒等式の最大相対残差：{theory_check['max_curvature_decomposition_relative_residual']:.3e}。
- 時間は各試行の実出力Nに合わせて理論を積分。最終スペクトルは出力スケジュールに依存せず、終了ステップの状態から計算。
- 背景は同じ初期条件から独立なRK4で計算し、格子の背景データを理論に代入していません。
  したがって、格子の背景の離散化誤差や揺らぎの平均エネルギーの効果も残差に入り得ます。
- 初期モードの短波長カットはありません。箱長を固定して格子点数を増やすと、新たな短波長真空モードのエネルギーも加わります。
  異なる格子間の差は、同じ物理的UVカットを保った空間離散化の収束試験とは区別してください。
- CSVの `prediction_with_lattice_projection` は最後の曲率射影係数だけを格子値に置き換えた補助診断。
  モード発展は独立背景のままです。
- 初期真空条件は有限時刻に設定したコードの規約に一致させています。漸近的BD真空や観測量との一致を保証する比較ではありません。

## 出力

- [曲率比較の時間変化]({figures/'curvature_comparison_over_time.png'})
- [両場のパワー比較]({figures/'field_power_comparison.png'})
- [初期乱数分布]({figures/'initial_random_statistics.png'})
- [背景の時間発展]({figures/'background_evolution.png'})
- [全試行・各時刻・各波数の比較CSV]({root/'spectra_comparison.csv'})
- [曲率の自己項・相互相関の分解]({root/'curvature_decomposition.csv'})
- [試行間変動の統計]({root/'spectrum_seed_attribution.csv'})
- [初期統計]({root/'initial_statistics_all.csv'}) / [試行情報]({root/'trials.csv'})
- [初期の場間相関]({root/'initial_cross_statistics.csv'})

## 参考

多場での断熱・等曲率分解は [Gordon et al.](https://arxiv.org/abs/astro-ph/0009131)、
計量摂動の有無による線形方程式の区別は [Barker et al.](https://arxiv.org/html/2606.14853v1) を参照。
今回の数値比較は保存した格子データと本フォルダの独立ODEの結果です。
'''
    (root/'REPORT_JA.md').write_text(report,encoding='utf-8')
    zband=next(s for s in summaries if s['quantity']=='zeta_linear' and s['scope']=='conservative_bins')
    zall=next(s for s in summaries if s['quantity']=='zeta_linear' and s['scope']=='all_bins')
    zdata=[r for r in final if r['quantity']=='zeta_linear']
    worst=max(zdata,key=lambda r:abs(r['residual_fraction']))
    conclusion=f'''\n## 結論：曲率パワーの差の内訳\n
- 保守的範囲の同一初期配列に対する残差は {zband['min_residual_percent']:+.6f}～{zband['max_residual_percent']:+.6f}%。
- 同範囲の試行間分散の説明割合は最小でも {zband['min_explained_variance_percent']:.6f}%。試行差は初期乱数からの予測に強く追随していますが、全試行に共通する上記残差は別に残ります。
- 全ビンの平均スペクトルと分布の期待値の最大偏差は {zall['max_abs_ensemble_sigma_deviation']:.3f}σ。全体を「乱数誤差だけで完全一致」とは結論しません。
- 最大の条件付き相対差は試行{worst['trial']}、ビン{int(worst['k_bin'])}、モード数{int(worst['n_modes'])}で {100*worst['residual_fraction']:+.6f}%。極端に少ないモード数の端ビンを、通常の殻平均と区別して確認してください。
'''
    if (root/'TIMESTEP_CHECK_JA.md').exists():
        conclusion+=f"\n[同じシードで時間刻みを半分にした追加検証]({root/'TIMESTEP_CHECK_JA.md'}) と [比較図]({figures/'lattice_timestep_convergence.png'}) を参照してください。\n"
    with (root/'REPORT_JA.md').open('a',encoding='utf-8') as f:
        f.write(conclusion)
    print(json.dumps(metadata,indent=2))
    print('REPORT',root/'REPORT_JA.md')


if __name__ == '__main__':
    main()
