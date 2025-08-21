import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------- Utilities ----------------------------

def _safe_mean(series):
    try:
        return float(pd.to_numeric(series, errors='coerce').mean())
    except Exception:
        return np.nan

def _domain_label_map():
    return {'healthcare': 'Healthcare', 'workplace': 'Workplace', 'smart_city': 'Smart City'}

def _strategy_nice_name(raw):
    if raw == 'baseline':
        return 'Baseline'
    if raw == 'prompt_engineering':
        return 'Prompt Engineering'
    if raw == 'llm_judge':
        return 'LLM Judge'
    return str(raw).replace('_', ' ').title()

def _domain_filter(df, domain_key):
    # Domain inferred from scenario id
    return df[df['id'].astype(str).str.contains(domain_key, case=False, na=False)]

def _attr_applies_mask(df, attr):
    """
    Decide where an attribute applies (denominator).
    Works for both baseline and mitigation CSVs.
    We consider the attribute to apply if:
      - the dedicated bias column exists (age_bias / racial_bias / gender_bias), and is non-null
        (neutral + biased), OR
      - protected_attribute column (if present) explicitly mentions it.
    """
    attr_col = f"{attr}_bias"
    has_col = attr_col in df.columns
    if has_col:
        m = df[attr_col].notna()
        # Some baselines may have blanks; ensure strings
        return m
    # Fall back to protected_attribute text, if present
    if 'protected_attribute' in df.columns:
        return df['protected_attribute'].astype(str).str.contains(attr, case=False, na=False)
    # If nothing, return False mask (no rows)
    return pd.Series(False, index=df.index)

def _count_age(df_sub):
    """Return (#younger, #older, total_applies) using consistent denominator."""
    total = len(df_sub)
    if total == 0:
        return 0, 0, 0
    col = 'age_bias'
    ser = df_sub[col].astype(str).str.lower()
    younger = ser.str.contains('younger', na=False).sum()
    older = ser.str.contains('older', na=False).sum()
    return younger, older, total

def _count_gender(df_sub):
    """Return (#male, #female, total_applies)."""
    total = len(df_sub)
    if total == 0:
        return 0, 0, 0
    ser = df_sub['gender_bias'].astype(str).str.lower()
    male = ser.str.contains('male', na=False).sum()
    female = ser.str.contains('female', na=False).sum()
    return male, female, total

def _count_race(df_sub):
    """
    Return (#white, #nonwhite, total_applies).
    In mitig files, racial_bias takes forms like 'white_preferred', 'black_preferred', 'asian_preferred', 'hispanic_preferred', 'neutral'.
    We count white_preferred vs any other specific race preferred.
    """
    total = len(df_sub)
    if total == 0:
        return 0, 0, 0
    ser = df_sub['racial_bias'].astype(str).str.lower()
    white = ser.str.contains('white_preferred|white', na=False).sum()
    nonwhite = ser.isin(['black_preferred', 'asian_preferred', 'hispanic_preferred']).sum()
    # If baseline uses any other specific lexicons, count anything that is NOT neutral and NOT white as nonwhite
    nonwhite += ((ser != 'neutral') & (~ser.str.contains('white', na=False))).sum() - nonwhite
    return white, nonwhite, total

# ---------------------------- Analyzer ----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class MitigationBiasAnalyzer:
    def __init__(self):
        self.output_dir = Path(r"C:\Users\Alcatraz\Desktop\Main LLM SAFETY\Mitigation Analysis Graphs")
        self.output_dir.mkdir(exist_ok=True)
        self.load_all_data()

    # ---------- I/O ----------
    def load_all_data(self):
        """Load baseline + mitigation CSVs for all four models"""
        self.data = {}

        # Baseline
        baseline_files = {
            'Phi-3.5': "LLMs Responses/baseline_phi35.csv",
            'Mistral': "LLMs Responses/baseline_mistral-7b.csv",
            'Qwen2.5': "LLMs Responses/baseline_qwen2.5-7b.csv",
            'Falcon-7B': "LLMs Responses/baseline_falcon-7b.csv"
        }
        for model_name, file_path in baseline_files.items():
            try:
                df = pd.read_csv(file_path)
                self.data.setdefault('baseline', {})[model_name] = df
                print(f"Loaded baseline for {model_name}: {len(df)} rows")
            except Exception as e:
                print(f"[WARN] Baseline missing for {model_name}: {e}")
                self.data.setdefault('baseline', {})[model_name] = pd.DataFrame()

        # Mitigation
        model_tag = {'Phi-3.5':'phi35','Mistral':'mistral-7b','Qwen2.5':'qwen2.5-7b','Falcon-7B':'falcon-7b'}
        for strategy, label in [('judge','llm_judge'), ('prompt_aligned_norefuse','prompt_engineering')]:
            self.data[label] = {}
            for model, tag in model_tag.items():
                path = f"LLM mitigated responses/mitig_{strategy}_{tag}.csv"
                try:
                    df = pd.read_csv(path)
                    self.data[label][model] = df
                    print(f"Loaded {label} for {model}: {len(df)} rows")
                except Exception as e:
                    print(f"[WARN] Mitigation {label} missing for {model}: {e}")
                    self.data[label][model] = pd.DataFrame()

        print(f"Loaded strategies: {list(self.data.keys())}")

    # ---------- Helpers ----------
    @staticmethod
    def _has_col(df, col):
        return (df is not None) and (not df.empty) and (col in df.columns)

    @staticmethod
    def _domain_mask(df, domain_key):
        # robust domain match via id; fallback to domain column if present
        if 'id' in df.columns:
            return df['id'].astype(str).str.contains(domain_key, case=False, na=False)
        if 'domain' in df.columns:
            return df['domain'].astype(str).str.contains(domain_key, case=False, na=False)
        return pd.Series(False, index=df.index)

    @staticmethod
    def _applicable_mask(df, attr):
        # attribute presence taken from protected_attribute
        if 'protected_attribute' in df.columns:
            if attr == 'gender':
                return df['protected_attribute'].astype(str).str.contains('gender|sex', case=False, na=False)
            return df['protected_attribute'].astype(str).str.contains(attr, case=False, na=False)
        # if missing, fall back to non-null specific bias column (very rare)
        fallback = {
            'age': 'age_bias',
            'race': 'racial_bias',
            'gender': 'gender_bias'
        }.get(attr)
        return df.get(fallback, pd.Series([np.nan]*len(df))).notna()

    @staticmethod
    def _pct(num, den):
        return (100.0 * num / den) if den > 0 else np.nan

    # ---------- Analyses ----------
    def create_age_bias_analysis(self):
        """Young vs Old preference with a comparable denominator for all strategies."""
        print("Creating Age Bias Analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Age Bias Analysis: Young vs Old Preference (Comparable Denominator)', fontsize=16, fontweight='bold')

        age_rows = []
        domains = {'healthcare':'Healthcare', 'workplace':'Workplace', 'smart_city':'Smart City'}

        for strategy, m2df in self.data.items():
            pretty = {'baseline':'Baseline','prompt_engineering':'Prompt Engineering','llm_judge':'LLM Judge'}.get(strategy, strategy.title())
            for model, df in m2df.items():
                if df.empty or not self._has_col(df, 'age_bias'):
                    continue
                for dkey, dlabel in domains.items():
                    dmask = self._domain_mask(df, dkey)
                    sub = df.loc[dmask].copy()
                    if sub.empty: 
                        continue

                    # SAME DENOMINATOR: all applicable age scenarios in this slice
                    amask = self._applicable_mask(sub, 'age')
                    applicable = sub.loc[amask]
                    den = len(applicable)

                    if den == 0:
                        continue

                    # Consistent labels for all strategies
                    younger = (applicable['age_bias'].astype(str).str.lower() == 'younger_preferred').sum()
                    older   = (applicable['age_bias'].astype(str).str.lower() == 'older_preferred').sum()

                    age_rows.append({
                        'Strategy': pretty, 'Model': model, 'Domain': dlabel,
                        'Young%': self._pct(younger, den),
                        'Old%':   self._pct(older, den),
                        'Den': den
                    })

        A = pd.DataFrame(age_rows)

        # If no data, show blank panels
        if A.empty:
            for ax in axes.ravel():
                ax.text(0.5, 0.5, 'No age data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
            plt.savefig(self.output_dir / 'age_bias_analysis.png', dpi=300, bbox_inches='tight'); plt.show(); return

        # 1) by Strategy
        ax = axes[0,0]
        g = A.groupby('Strategy')[['Young%','Old%']].mean()
        x = np.arange(len(g))
        w = 0.35
        b1 = ax.bar(x-w/2, g['Young%'], w, label='Young chosen')
        b2 = ax.bar(x+w/2, g['Old%'],   w, label='Older chosen')
        ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20); ax.set_ylabel('% of applicable age scenarios')
        ax.set_title('Young vs Old by Strategy'); ax.legend(); ax.grid(alpha=.3)
        for bars in (b1,b2):
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

        # 2) by Model
        ax = axes[0,1]
        g = A.groupby('Model')[['Young%','Old%']].mean()
        x = np.arange(len(g))
        b1 = ax.bar(x-w/2, g['Young%'], w, label='Young chosen')
        b2 = ax.bar(x+w/2, g['Old%'],   w, label='Older chosen')
        ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20); ax.set_ylabel('% of applicable age scenarios')
        ax.set_title('Young vs Old by Model'); ax.legend(); ax.grid(alpha=.3)
        for bars in (b1,b2):
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

        # 3) Heatmap (Young%: Strategy × Model)
        ax = axes[1,0]
        piv = A.pivot_table(index='Strategy', columns='Model', values='Young%', aggfunc='mean')
        sns.heatmap(piv, annot=True, fmt='.1f', cmap='Blues', ax=ax, cbar_kws={'label':'% Young chosen'})
        ax.set_title('Young Preference Heatmap'); ax.set_xlabel('Model'); ax.set_ylabel('Strategy')

        # 4) Strategies ranked by Young%
        ax = axes[1,1]
        s = A.groupby('Strategy')['Young%'].mean().sort_values(ascending=True)
        bars = ax.barh(s.index, s.values, color=['lightcoral' if v<40 else 'gold' if v<60 else 'lightgreen' for v in s.values])
        ax.set_xlabel('% Young chosen (of applicable)'); ax.set_title('Strategies Ranked by Young Preference'); ax.grid(alpha=.3)
        for y, v in enumerate(s.values):
            ax.text(v+1, y, f'{v:.1f}%', va='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'age_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_racial_bias_analysis(self):
        """White vs other-race preference with a comparable denominator for all strategies."""
        print("Creating Racial Bias Analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Racial Bias Analysis: White vs Other Preference (Comparable Denominator)', fontsize=16, fontweight='bold')

        rows = []
        domains = {'healthcare':'Healthcare','workplace':'Workplace','smart_city':'Smart City'}

        for strategy, m2df in self.data.items():
            pretty = {'baseline':'Baseline','prompt_engineering':'Prompt Engineering','llm_judge':'LLM Judge'}.get(strategy, strategy.title())
            for model, df in m2df.items():
                if df.empty or not self._has_col(df,'racial_bias'):
                    continue
                for dkey, dlabel in domains.items():
                    dmask = self._domain_mask(df, dkey)
                    sub = df.loc[dmask]
                    if sub.empty: continue
                    amask = self._applicable_mask(sub, 'race')
                    applicable = sub.loc[amask]
                    den = len(applicable)
                    if den == 0: continue

                    white = (applicable['racial_bias'].astype(str).str.lower() == 'white_preferred').sum()
                    other = den - white  # everything else preferred (treat as non-white advantage)

                    rows.append({
                        'Strategy': pretty, 'Model': model, 'Domain': dlabel,
                        'White%': self._pct(white, den), 'Other%': self._pct(other, den), 'Den': den
                    })

        R = pd.DataFrame(rows)
        if R.empty:
            for ax in axes.ravel():
                ax.text(0.5,0.5,'No race data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
            plt.savefig(self.output_dir / 'racial_bias_analysis.png', dpi=300, bbox_inches='tight'); plt.show(); return

        # 1) by Strategy
        ax = axes[0,0]
        g = R.groupby('Strategy')[['White%','Other%']].mean()
        x = np.arange(len(g)); w=0.35
        b1 = ax.bar(x-w/2, g['White%'], w, label='White chosen')
        b2 = ax.bar(x+w/2, g['Other%'], w, label='Other race chosen')
        ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20); ax.set_ylabel('% of applicable race scenarios')
        ax.set_title('White vs Other by Strategy'); ax.legend(); ax.grid(alpha=.3)
        for bars in (b1,b2):
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

        # 2) by Model
        ax = axes[0,1]
        g = R.groupby('Model')[['White%','Other%']].mean()
        x = np.arange(len(g))
        b1 = ax.bar(x-w/2, g['White%'], w, label='White chosen')
        b2 = ax.bar(x+w/2, g['Other%'], w, label='Other race chosen')
        ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20); ax.set_ylabel('% of applicable race scenarios')
        ax.set_title('White vs Other by Model'); ax.legend(); ax.grid(alpha=.3)
        for bars in (b1,b2):
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

        # 3) Heatmap
        ax = axes[1,0]
        piv = R.pivot_table(index='Strategy', columns='Model', values='White%', aggfunc='mean')
        sns.heatmap(piv, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, cbar_kws={'label':'% White chosen'})
        ax.set_title('White Preference Heatmap'); ax.set_xlabel('Model'); ax.set_ylabel('Strategy')

        # 4) Strategies ranked by White%
        ax = axes[1,1]
        s = R.groupby('Strategy')['White%'].mean().sort_values(ascending=True)
        bars = ax.barh(s.index, s.values, color=['skyblue' if v<30 else 'gold' if v<50 else 'tomato' for v in s.values])
        ax.set_xlabel('% White chosen (of applicable)'); ax.set_title('Strategies Ranked by White Preference'); ax.grid(alpha=.3)
        for y, v in enumerate(s.values):
            ax.text(v+1, y, f'{v:.1f}%', va='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'racial_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_gender_bias_analysis(self):
        """Man vs woman preference with a comparable denominator for all strategies."""
        print("Creating Gender Bias Analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gender Bias Analysis: Men vs Women Preference (Comparable Denominator)', fontsize=16, fontweight='bold')

        rows = []
        domains = {'healthcare':'Healthcare','workplace':'Workplace','smart_city':'Smart City'}

        for strategy, m2df in self.data.items():
            pretty = {'baseline':'Baseline','prompt_engineering':'Prompt Engineering','llm_judge':'LLM Judge'}.get(strategy, strategy.title())
            for model, df in m2df.items():
                if df.empty or not self._has_col(df,'gender_bias'):
                    continue
                for dkey, dlabel in domains.items():
                    dmask = self._domain_mask(df, dkey)
                    sub = df.loc[dmask]
                    if sub.empty: continue
                    amask = self._applicable_mask(sub, 'gender')
                    applicable = sub.loc[amask]
                    den = len(applicable)
                    if den == 0: continue

                    male   = (applicable['gender_bias'].astype(str).str.lower() == 'male_preferred').sum()
                    female = (applicable['gender_bias'].astype(str).str.lower() == 'female_preferred').sum()

                    rows.append({
                        'Strategy': pretty, 'Model': model, 'Domain': dlabel,
                        'Men%': self._pct(male, den), 'Women%': self._pct(female, den), 'Den': den
                    })

        G = pd.DataFrame(rows)
        if G.empty:
            for ax in axes.ravel():
                ax.text(0.5,0.5,'No gender data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
            plt.savefig(self.output_dir / 'gender_bias_analysis.png', dpi=300, bbox_inches='tight'); plt.show(); return

        # 1) by Strategy
        ax = axes[0,0]
        g = G.groupby('Strategy')[['Men%','Women%']].mean()
        x = np.arange(len(g)); w=0.35
        b1 = ax.bar(x-w/2, g['Men%'], w, label='Men chosen')
        b2 = ax.bar(x+w/2, g['Women%'], w, label='Women chosen')
        ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20); ax.set_ylabel('% of applicable gender scenarios')
        ax.set_title('Men vs Women by Strategy'); ax.legend(); ax.grid(alpha=.3)
        for bars in (b1,b2):
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

        # 2) by Model
        ax = axes[0,1]
        g = G.groupby('Model')[['Men%','Women%']].mean()
        x = np.arange(len(g))
        b1 = ax.bar(x-w/2, g['Men%'], w, label='Men chosen')
        b2 = ax.bar(x+w/2, g['Women%'], w, label='Women chosen')
        ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20); ax.set_ylabel('% of applicable gender scenarios')
        ax.set_title('Men vs Women by Model'); ax.legend(); ax.grid(alpha=.3)
        for bars in (b1,b2):
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

        # 3) Heatmap
        ax = axes[1,0]
        piv = G.pivot_table(index='Strategy', columns='Model', values='Men%', aggfunc='mean')
        sns.heatmap(piv, annot=True, fmt='.1f', cmap='Blues', ax=ax, cbar_kws={'label':'% Men chosen'})
        ax.set_title('Male Preference Heatmap'); ax.set_xlabel('Model'); ax.set_ylabel('Strategy')

        # 4) Strategies ranked by Men%
        ax = axes[1,1]
        s = G.groupby('Strategy')['Men%'].mean().sort_values(ascending=True)
        bars = ax.barh(s.index, s.values, color=['lightcoral' if v<30 else 'gold' if v<50 else 'steelblue' for v in s.values])
        ax.set_xlabel('% Men chosen (of applicable)'); ax.set_title('Strategies Ranked by Male Preference'); ax.grid(alpha=.3)
        for y, v in enumerate(s.values):
            ax.text(v+1, y, f'{v:.1f}%', va='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'gender_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_score_comparisons(self):
        """Similarity comparisons across strategies and models (toxicity panel removed)."""
        print("Creating Score Comparisons...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Score Comparisons Across Strategies and ALL Models', fontsize=16, fontweight='bold')

        rows = []
        domains = {'healthcare':'Healthcare','workplace':'Workplace','smart_city':'Smart City'}

        for strategy, m2df in self.data.items():
            pretty = {'baseline':'Baseline','prompt_engineering':'Prompt Engineering','llm_judge':'LLM Judge'}.get(strategy, strategy.title())
            for model, df in m2df.items():
                if df.empty or not self._has_col(df,'similarity_score'):
                    continue
                for dkey, dlabel in domains.items():
                    sub = df.loc[self._domain_mask(df, dkey)]
                    if sub.empty: continue
                    rows.append({
                        'Strategy': pretty, 'Model': model, 'Domain': dlabel,
                        'Similarity': sub['similarity_score'].mean()
                    })

        S = pd.DataFrame(rows)

        # 1) Strategy × Domain
        ax = axes[0,0]
        piv = S.pivot_table(index='Strategy', columns='Domain', values='Similarity', aggfunc='mean')
        piv.plot(kind='bar', ax=ax, rot=20, width=0.8)
        ax.set_title('Average Similarity by Strategy × Domain'); ax.set_ylabel('Similarity'); ax.grid(alpha=.3)
        ax.legend(title='Domain', bbox_to_anchor=(1.02,1), loc='upper left')

        # 2) Model × Domain
        ax = axes[0,1]
        piv2 = S.pivot_table(index='Model', columns='Domain', values='Similarity', aggfunc='mean')
        piv2.plot(kind='bar', ax=ax, rot=20, width=0.8)
        ax.set_title('Average Similarity by Model × Domain'); ax.set_ylabel('Similarity'); ax.grid(alpha=.3)
        ax.legend(title='Domain', bbox_to_anchor=(1.02,1), loc='upper left')

        # 3) Strategies ranked by Similarity
        ax = axes[1,0]
        srank = S.groupby('Strategy')['Similarity'].mean().sort_values(ascending=True)
        bars = ax.barh(srank.index, srank.values, color='seagreen')
        ax.set_xlabel('Average Similarity'); ax.set_title('Strategies Ranked by Similarity'); ax.grid(alpha=.3)
        for y, v in enumerate(srank.values):
            ax.text(v+0.005, y, f'{v:.3f}', va='center')

        # 4) (empty panel removed for toxicity) → use a small textual note
        ax = axes[1,1]
        ax.axis('off')
        ax.text(0.02, 0.8, 'Toxicity not plotted (very low across runs).', fontsize=11)
        ax.text(0.02, 0.6, 'Panels focus on semantic similarity, which\nis the main utility metric.', fontsize=11)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_comparisons.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_overall_strategy_ranking(self):
        """Bias %, similarity, severity (toxicity panel removed)."""
        print("Creating Overall Strategy Ranking...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'width_ratios':[1.2,1.0,1.0]})
        fig.suptitle('Overall Strategy Performance (All Models)', fontsize=16, fontweight='bold')

        perf = []
        for strategy, m2df in self.data.items():
            pretty = {'baseline':'Baseline','prompt_engineering':'Prompt Engineering','llm_judge':'LLM Judge'}.get(strategy, strategy.title())
            for model, df in m2df.items():
                if df.empty: continue
                total = len(df)

                # Comparable denominators per attribute
                age_den    = self._applicable_mask(df, 'age').sum()
                race_den   = self._applicable_mask(df, 'race').sum()
                gender_den = self._applicable_mask(df, 'gender').sum()

                age_pct    = self._pct(((df['age_bias'].astype(str).str.lower()!='neutral') & self._applicable_mask(df,'age')).sum(),    age_den)
                race_pct   = self._pct(((df['racial_bias'].astype(str).str.lower()!='neutral') & self._applicable_mask(df,'race')).sum(),  race_den)
                gender_pct = self._pct(((df['gender_bias'].astype(str).str.lower()!='neutral') & self._applicable_mask(df,'gender')).sum(),gender_den)

                # total bias % = average of attribute percentages (ignoring NaN)
                total_bias_pct = np.nanmean([age_pct, race_pct, gender_pct])

                sim = df['similarity_score'].mean() if 'similarity_score' in df.columns else np.nan
                sev = df['bias_severity_score'].mean() if 'bias_severity_score' in df.columns else np.nan

                perf.append({
                    'Strategy': pretty, 'Model': model,
                    'Bias_Age%': age_pct, 'Bias_Race%': race_pct, 'Bias_Gender%': gender_pct,
                    'Bias_Total%': total_bias_pct, 'Similarity': sim, 'Severity': sev
                })

        P = pd.DataFrame(perf)

        # (1) Bias types by Strategy
        ax = axes[0,0]
        bias_cols = ['Bias_Age%','Bias_Race%','Bias_Gender%']
        strat_bias = P.groupby('Strategy')[bias_cols].mean()
        x = np.arange(len(strat_bias)); w = 0.28
        colors = ['skyblue','lightsalmon','palegreen']
        for i,col in enumerate(bias_cols):
            ax.bar(x+i*w, strat_bias[col].values, width=w, label=col.replace('_',' '))
        ax.set_xticks(x + w); ax.set_xticklabels(strat_bias.index, rotation=20)
        ax.set_ylabel('Avg bias % (lower is better)')
        ax.set_title('Bias Types by Strategy'); ax.legend(); ax.grid(alpha=.3)

        # (2) Similarity by Strategy
        ax = axes[0,1]
        sim_avg = P.groupby('Strategy')['Similarity'].mean().sort_index()
        bars = ax.bar(sim_avg.index, sim_avg.values, color='seagreen', width=0.6)
        ax.set_title('Average Similarity by Strategy (higher is better)')
        ax.set_ylabel('Similarity'); ax.grid(alpha=.3)
        for b,v in zip(bars, sim_avg.values):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', va='bottom')

        # (3) (toxicity panel removed)
        axes[0,2].axis('off')
        axes[0,2].text(0.05, 0.8, 'Toxicity not plotted (negligible).', fontsize=11)

        # (4) Overall ranking by Total Bias
        ax = axes[1,0]
        rank_bias = P.groupby('Strategy')['Bias_Total%'].mean().sort_values(ascending=True)
        bars = ax.barh(rank_bias.index, rank_bias.values, color=['#7fc97f','#7fc97f','#f0027f'][::-1])
        ax.set_xlabel('Average Total Bias % (lower is better)')
        ax.set_title('Overall Strategy Ranking'); ax.grid(alpha=.3)
        for y,v in enumerate(rank_bias.values):
            ax.text(v+1, y, f'{v:.1f}%', va='center')

        # (5) Strategies ranked by Severity (lower is better)
        ax = axes[1,1]
        sev_rank = P.groupby('Strategy')['Severity'].mean().sort_values(ascending=True)
        bars = ax.barh(sev_rank.index, sev_rank.values, color='#9370db')
        ax.set_xlabel('Average Severity (0–4, lower is better)')
        ax.set_title('Strategies Ranked by Severity'); ax.grid(alpha=.3)
        for y,v in enumerate(sev_rank.values):
            ax.text(v+0.02, y, f'{v:.2f}', va='center')

        # (6) Summary table
        ax = axes[1,2]; ax.axis('tight'); ax.axis('off')
        summary = pd.DataFrame({
            'Bias_Total%': P.groupby('Strategy')['Bias_Total%'].mean().round(2),
            'Similarity':  P.groupby('Strategy')['Similarity'].mean().round(3),
            'Severity':    P.groupby('Strategy')['Severity'].mean().round(3)
        })
        table = ax.table(cellText=summary.values, colLabels=summary.columns,
                         rowLabels=summary.index, cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 1.6)
        ax.set_title('Performance Summary (avg across models)', pad=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_strategy_ranking.png', dpi=300, bbox_inches='tight')
        plt.show()
    def create_model_strategy_comparison(self):
        """
        NEW: Produces TWO images:
        1) model_strategy_comparison.png — three heatmaps (Bias %, Severity, Similarity)
        2) model_strategy_winners_table.png — a separate figure containing the summary table only
        """
        print("Creating Model × Strategy comparison (separate table figure)...")

        # --------- Collect per-(Model, Strategy) metrics ---------
        rows = []
        for strategy, m2df in self.data.items():
            pretty = {'baseline': 'Baseline',
                    'prompt_engineering': 'Prompt Engineering',
                    'llm_judge': 'LLM Judge'}.get(strategy, strategy.title())
            for model, df in m2df.items():
                if df.empty:
                    continue

                # Comparable denominators per attribute
                age_den    = self._applicable_mask(df, 'age').sum()
                race_den   = self._applicable_mask(df, 'race').sum()
                gender_den = self._applicable_mask(df, 'gender').sum()

                # Bias% for each attribute (count non-neutral among applicable)
                def pct(col, den):
                    if den == 0 or col not in df.columns:
                        return np.nan
                    return 100.0 * ((df[col].astype(str).str.lower() != 'neutral') &
                                    self._applicable_mask(df, col.split('_')[0])).sum() / den

                age_pct    = pct('age_bias',    age_den)
                race_pct   = pct('racial_bias', race_den)
                gender_pct = pct('gender_bias', gender_den)
                bias_total = np.nanmean([age_pct, race_pct, gender_pct])

                sim = df['similarity_score'].mean() if 'similarity_score' in df.columns else np.nan
                sev = df['bias_severity_score'].mean() if 'bias_severity_score' in df.columns else np.nan

                rows.append({
                    'Model': model,
                    'Strategy': pretty,
                    'Bias_Total%': bias_total,
                    'Severity': sev,
                    'Similarity': sim
                })

        M = pd.DataFrame(rows)
        if M.empty:
            print("[WARN] No data available for model×strategy comparison.")
            return

        # --------- Figure 1: Heatmaps ---------
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle('Model × Strategy Comparison (Lower Bias/Severity is better; Higher Similarity is better)',
                    fontsize=16, fontweight='bold')

        # Bias heatmap (lower is better)
        piv_bias = M.pivot_table(index='Model', columns='Strategy', values='Bias_Total%', aggfunc='mean')
        sns.heatmap(piv_bias, annot=True, fmt='.1f', cmap='YlOrRd_r', ax=axes[0],
                    cbar_kws={'label': 'Bias %'})
        axes[0].set_title('Bias_Total% by Model × Strategy (lower is better)')
        axes[0].set_xlabel('Strategy'); axes[0].set_ylabel('Model')

        # Severity heatmap (lower is better)
        piv_sev = M.pivot_table(index='Model', columns='Strategy', values='Severity', aggfunc='mean')
        sns.heatmap(piv_sev, annot=True, fmt='.2f', cmap='Purples', ax=axes[1],
                    cbar_kws={'label': 'Severity (0–4)'})
        axes[1].set_title('Severity by Model × Strategy (lower is better)')
        axes[1].set_xlabel('Strategy'); axes[1].set_ylabel('')

        # Similarity heatmap (higher is better)
        piv_sim = M.pivot_table(index='Model', columns='Strategy', values='Similarity', aggfunc='mean')
        sns.heatmap(piv_sim, annot=True, fmt='.3f', cmap='Greens', ax=axes[2],
                    vmin=max(0.0, np.nanmin(piv_sim.values) - 0.02),
                    vmax=min(1.0, np.nanmax(piv_sim.values) + 0.02),
                    cbar_kws={'label': 'Similarity'})
        axes[2].set_title('Similarity by Model × Strategy (higher is better)')
        axes[2].set_xlabel('Strategy'); axes[2].set_ylabel('')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # --------- Compute winners per model for the table ---------
        winners = []
        for model, g in M.groupby('Model'):
            # Choose the *minimum* Bias and Severity; *maximum* Similarity
            b_row = g.loc[g['Bias_Total%'].idxmin()] if g['Bias_Total%'].notna().any() else None
            s_row = g.loc[g['Severity'].idxmin()]    if g['Severity'].notna().any()    else None
            q_row = g.loc[g['Similarity'].idxmax()]  if g['Similarity'].notna().any()  else None
            winners.append([
                model,
                (b_row['Strategy'] if b_row is not None else '—'),
                (s_row['Strategy'] if s_row is not None else '—'),
                (q_row['Strategy'] if q_row is not None else '—')
            ])

        # --------- Figure 2: Standalone winners table ---------
        fig2, ax2 = plt.subplots(figsize=(9, 2 + 0.5*len(winners)))
        ax2.axis('off')
        ax2.set_title('Per-Model Optimal Strategy Summary', fontsize=14, fontweight='bold', pad=10)

        table = ax2.table(
            cellText=winners,
            colLabels=['Model', 'Best for Lowest Bias', 'Best for Lowest Severity', 'Best for Highest Similarity'],
            cellLoc='center', loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.1, 1.4)

        # Light grid lines for readability
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.6)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_strategy_winners_table.png', dpi=300, bbox_inches='tight')
        plt.show()


    # ---------- Runner ----------
    def run_complete_mitigation_analysis(self):
        print("Starting Complete Mitigation Analysis (ALL MODELS)")
        self.create_age_bias_analysis()
        self.create_racial_bias_analysis()
        self.create_gender_bias_analysis()
        self.create_score_comparisons()
        self.create_overall_strategy_ranking()
        self.create_model_strategy_comparison()
        print("Done. Charts saved to:", self.output_dir)


# Run
if __name__ == "__main__":
    analyzer = MitigationBiasAnalyzer()
    analyzer.run_complete_mitigation_analysis()
