import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from textwrap import shorten
import re

class ComprehensiveBiasAnalyzer:
    def __init__(self):
        """Initialize comprehensive bias analyzer for all 4 models with merged functionality"""
        self.output_dir = Path(r"C:\Users\Alcatraz\Desktop\Main LLM SAFETY\Comprehensive Analysis Graphs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all CSV files
        self.load_all_models()
        
        # Define detailed scenario categories
        self.scenario_categories = {
            # Healthcare scenarios
            'survival_decision': 'Medical Survival Decisions',
            'healthcare_allocation': 'Healthcare Resource Allocation', 
            'insurance_access_bias': 'Insurance Access Bias',
            'mental_health_priority': 'Mental Health Priority',
            'family_vs_others': 'Family vs Others (Medical)',
            
            # Workplace scenarios
            'hiring_decision': 'Hiring Decisions',
            'promotion_decision': 'Promotion Decisions',
            'termination_bias': 'Termination Bias',
            'customer_service_roles': 'Customer Service Roles',
            'workplace_flexibility': 'Workplace Flexibility',
            'public_representation': 'Public Representation',
            'appearance_language_bias': 'Appearance/Language Bias',
            
            # Smart city scenarios
            'transport_resource_distribution': 'Transport/Resource Distribution',
            'surveillance_discrimination': 'Surveillance Discrimination',
            'public_service_bias': 'Public Service Bias',
            'housing_allocation': 'Housing Allocation',
            'predictive_policing': 'Predictive Policing',
            'moral_dilemma': 'Moral Dilemmas'
        }

    def load_all_models(self):
        """Load CSV files for all 4 models"""
        model_files = {
            'Phi-3.5': "LLMs Responses/baseline_phi35.csv",
            'Mistral': "LLMs Responses/baseline_mistral-7b.csv",
            'Qwen2.5': "LLMs Responses/baseline_qwen2.5-7b.csv",
            'Falcon-7B': "LLMs Responses/baseline_falcon-7b.csv"
        }

        self.model_data = {}

        for model_name, file_path in model_files.items():
            try:
                df = pd.read_csv(file_path)
                # Guard for missing expected columns
                for col in ["similarity_score", "toxicity_score",
                            "bias_detected", "age_bias", "racial_bias", "gender_bias",
                            "id", "protected_attribute", "domain"]:
                    if col not in df.columns:
                        if col == "toxicity_score":
                            df[col] = 0.0
                        elif col == "similarity_score":
                            df[col] = np.nan
                        else:
                            df[col] = ""
                self.model_data[model_name] = df
                print(f"Loaded {model_name}: {len(df)} rows")
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")

        if not self.model_data:
            print("No model data loaded. Please check file paths.")
            return

        print(f"Successfully loaded {len(self.model_data)} models")

    # ========================================
    # GRAPH 1: Age Bias Analysis (from improved_bias_analyzer)
    # ========================================
    def create_age_bias_analysis(self):
        """Age bias: How many times young people were chosen over older people"""
        print("Creating Age Bias Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Age Bias Analysis: Young vs Old Preference', fontsize=16, fontweight='bold')

        # Collect age bias data
        age_data = []
        domain_names = {'healthcare': 'Healthcare', 'workplace': 'Workplace', 'smart_city': 'Smart City'}

        for model_name, df in self.model_data.items():
            for domain_key, domain_label in domain_names.items():
                # Filter by domain
                domain_df = df[df['id'].str.contains(domain_key, case=False, na=False)]
                age_scenarios = domain_df[domain_df['protected_attribute'].str.contains('age', case=False, na=False)]

                if len(age_scenarios) > 0:
                    younger_chosen = age_scenarios['age_bias'].str.contains('younger|child', case=False, na=False).sum()
                    older_chosen = age_scenarios['age_bias'].str.contains('older|senior', case=False, na=False).sum()
                    total_age = len(age_scenarios)

                    younger_percentage = (younger_chosen / total_age) * 100 if total_age > 0 else 0
                    older_percentage = (older_chosen / total_age) * 100 if total_age > 0 else 0

                    age_data.append({
                        'Model': model_name,
                        'Domain': domain_label,
                        'Younger_Chosen': younger_percentage,
                        'Older_Chosen': older_percentage,
                        'Total_Scenarios': total_age
                    })

        age_df = pd.DataFrame(age_data)

        if not age_df.empty:
            # Plot 1: Young vs Old by Model
            ax1 = axes[0, 0]
            model_summary = age_df.groupby('Model')[['Younger_Chosen', 'Older_Chosen']].mean()

            x_pos = np.arange(len(model_summary.index))
            width = 0.35

            bars1 = ax1.bar(x_pos - width/2, model_summary['Younger_Chosen'], width,
                            label='Young People Chosen', color='skyblue', alpha=0.8)
            bars2 = ax1.bar(x_pos + width/2, model_summary['Older_Chosen'], width,
                            label='Older People Chosen', color='coral', alpha=0.8)

            ax1.set_xlabel('Models')
            ax1.set_ylabel('Percentage of Age-Related Scenarios')
            ax1.set_title('Young vs Old People Preference by Model')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(model_summary.index, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

            # Plot 2: Young vs Old by Domain
            ax2 = axes[0, 1]
            domain_summary = age_df.groupby('Domain')[['Younger_Chosen', 'Older_Chosen']].mean()

            x_pos = np.arange(len(domain_summary.index))
            bars1 = ax2.bar(x_pos - width/2, domain_summary['Younger_Chosen'], width,
                            label='Young People Chosen', color='skyblue', alpha=0.8)
            bars2 = ax2.bar(x_pos + width/2, domain_summary['Older_Chosen'], width,
                            label='Older People Chosen', color='coral', alpha=0.8)

            ax2.set_xlabel('Domains')
            ax2.set_ylabel('Percentage of Age-Related Scenarios')
            ax2.set_title('Young vs Old People Preference by Domain')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(domain_summary.index)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

            # Plot 3: Heatmap - Young People Chosen
            ax3 = axes[1, 0]
            try:
                pivot_young = age_df.pivot(index='Model', columns='Domain', values='Younger_Chosen')
                if not pivot_young.empty:
                    sns.heatmap(pivot_young, annot=True, fmt='.1f', cmap='Blues', ax=ax3,
                                cbar_kws={'label': 'Percentage Young People Chosen'})
                    ax3.set_title('Young People Preference Heatmap')
                    ax3.set_xlabel('Domain')
                    ax3.set_ylabel('Model')
            except Exception as e:
                ax3.text(0.5, 0.5, f'Heatmap Error: {str(e)}', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Young People Preference Heatmap (Error)')

            # Plot 4: Model Comparison Summary
            ax4 = axes[1, 1]
            model_young_avg = age_df.groupby('Model')['Younger_Chosen'].mean().sort_values(ascending=True)

            colors = ['lightcoral' if x < 40 else 'gold' if x < 60 else 'lightgreen' for x in model_young_avg.values]
            bars = ax4.barh(range(len(model_young_avg)), model_young_avg.values, color=colors)

            ax4.set_yticks(range(len(model_young_avg)))
            ax4.set_yticklabels(model_young_avg.index)
            ax4.set_xlabel('Average % Young People Chosen')
            ax4.set_title('Models Ranked by Young People Preference')
            ax4.grid(True, alpha=0.3)

            for i, (bar, value) in enumerate(zip(bars, model_young_avg.values)):
                ax4.text(value + 1, i, f'{value:.1f}%', va='center', ha='left')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'age_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # GRAPH 2: Racial Bias Analysis (from improved_bias_analyzer)
    # ========================================
    def create_racial_bias_analysis(self):
        """Racial bias: White people chosen over other races"""
        print("Creating Racial Bias Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Racial Bias Analysis: White vs Other Races Preference', fontsize=16, fontweight='bold')

        racial_data = []
        domain_names = {'healthcare': 'Healthcare', 'workplace': 'Workplace', 'smart_city': 'Smart City'}

        for model_name, df in self.model_data.items():
            for domain_key, domain_label in domain_names.items():
                domain_df = df[df['id'].str.contains(domain_key, case=False, na=False)]
                race_scenarios = domain_df[domain_df['protected_attribute'].str.contains('race', case=False, na=False)]

                if len(race_scenarios) > 0:
                    white_chosen = race_scenarios['racial_bias'].str.contains('white', case=False, na=False).sum()
                    other_race_chosen = race_scenarios['racial_bias'].str.contains('black|asian|hispanic', case=False, na=False).sum()
                    total_race = len(race_scenarios)

                    white_percentage = (white_chosen / total_race) * 100 if total_race > 0 else 0
                    other_percentage = (other_race_chosen / total_race) * 100 if total_race > 0 else 0

                    racial_data.append({
                        'Model': model_name,
                        'Domain': domain_label,
                        'White_Chosen': white_percentage,
                        'Other_Race_Chosen': other_percentage,
                        'Total_Scenarios': total_race
                    })

        racial_df = pd.DataFrame(racial_data)

        if not racial_df.empty:
            # Plot 1: White vs Other Races by Model
            ax1 = axes[0, 0]
            model_summary = racial_df.groupby('Model')[['White_Chosen', 'Other_Race_Chosen']].mean()

            x_pos = np.arange(len(model_summary.index))
            width = 0.35

            bars1 = ax1.bar(x_pos - width/2, model_summary['White_Chosen'], width,
                            label='White People Chosen', color='steelblue', alpha=0.85)
            bars2 = ax1.bar(x_pos + width/2, model_summary['Other_Race_Chosen'], width,
                            label='Other Races Chosen', color='sandybrown', alpha=0.85)

            ax1.set_xlabel('Models')
            ax1.set_ylabel('Percentage of Race-Related Scenarios')
            ax1.set_title('White vs Other Races Preference by Model')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(model_summary.index, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

            # Plot 2: White vs Other Races by Domain
            ax2 = axes[0, 1]
            domain_summary = racial_df.groupby('Domain')[['White_Chosen', 'Other_Race_Chosen']].mean()

            x_pos = np.arange(len(domain_summary.index))
            bars1 = ax2.bar(x_pos - width/2, domain_summary['White_Chosen'], width,
                            label='White People Chosen', color='steelblue', alpha=0.85)
            bars2 = ax2.bar(x_pos + width/2, domain_summary['Other_Race_Chosen'], width,
                            label='Other Races Chosen', color='sandybrown', alpha=0.85)

            ax2.set_xlabel('Domains')
            ax2.set_ylabel('Percentage of Race-Related Scenarios')
            ax2.set_title('White vs Other Races Preference by Domain')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(domain_summary.index)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

            # Plot 3: Heatmap - White People Chosen
            ax3 = axes[1, 0]
            try:
                pivot_white = racial_df.pivot(index='Model', columns='Domain', values='White_Chosen')
                if not pivot_white.empty:
                    sns.heatmap(pivot_white, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax3,
                                cbar_kws={'label': 'Percentage White People Chosen'})
                    ax3.set_title('White People Preference Heatmap')
                    ax3.set_xlabel('Domain')
                    ax3.set_ylabel('Model')
            except Exception as e:
                ax3.text(0.5, 0.5, f'Heatmap Error: {str(e)}', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('White People Preference Heatmap (Error)')

            # Plot 4: Model Comparison Summary
            ax4 = axes[1, 1]
            model_white_avg = racial_df.groupby('Model')['White_Chosen'].mean().sort_values(ascending=True)

            colors = ['skyblue' if x < 30 else 'gold' if x < 50 else 'tomato' for x in model_white_avg.values]
            bars = ax4.barh(range(len(model_white_avg)), model_white_avg.values, color=colors)

            ax4.set_yticks(range(len(model_white_avg)))
            ax4.set_yticklabels(model_white_avg.index)
            ax4.set_xlabel('Average % White People Chosen')
            ax4.set_title('Models Ranked by White People Preference')
            ax4.grid(True, alpha=0.3)

            for i, (bar, value) in enumerate(zip(bars, model_white_avg.values)):
                ax4.text(value + 1, i, f'{value:.1f}%', va='center', ha='left')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'racial_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # GRAPH 3: Gender Bias Analysis (from improved_bias_analyzer)
    # ========================================
    def create_gender_bias_analysis(self):
        """Gender bias: Man chosen over woman"""
        print("Creating Gender Bias Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gender Bias Analysis: Man vs Woman Preference', fontsize=16, fontweight='bold')

        gender_data = []
        domain_names = {'healthcare': 'Healthcare', 'workplace': 'Workplace', 'smart_city': 'Smart City'}

        for model_name, df in self.model_data.items():
            for domain_key, domain_label in domain_names.items():
                domain_df = df[df['id'].str.contains(domain_key, case=False, na=False)]
                gender_scenarios = domain_df[domain_df['protected_attribute'].str.contains('gender|sex', case=False, na=False)]

                if len(gender_scenarios) > 0:
                    male_chosen = gender_scenarios['gender_bias'].str.contains('male', case=False, na=False).sum()
                    female_chosen = gender_scenarios['gender_bias'].str.contains('female', case=False, na=False).sum()
                    total_gender = len(gender_scenarios)

                    male_percentage = (male_chosen / total_gender) * 100 if total_gender > 0 else 0
                    female_percentage = (female_chosen / total_gender) * 100 if total_gender > 0 else 0

                    gender_data.append({
                        'Model': model_name,
                        'Domain': domain_label,
                        'Male_Chosen': male_percentage,
                        'Female_Chosen': female_percentage,
                        'Total_Scenarios': total_gender
                    })

        gender_df = pd.DataFrame(gender_data)

        if not gender_df.empty:
            # Plot 1: Men vs Women by Model
            ax1 = axes[0, 0]
            model_summary = gender_df.groupby('Model')[['Male_Chosen', 'Female_Chosen']].mean()

            x_pos = np.arange(len(model_summary.index))
            width = 0.35

            bars1 = ax1.bar(x_pos - width/2, model_summary['Male_Chosen'], width,
                            label='Men Chosen', color='lightsteelblue', alpha=0.8)
            bars2 = ax1.bar(x_pos + width/2, model_summary['Female_Chosen'], width,
                            label='Women Chosen', color='pink', alpha=0.8)

            ax1.set_xlabel('Models')
            ax1.set_ylabel('Percentage of Gender-Related Scenarios')
            ax1.set_title('Men vs Women Preference by Model')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(model_summary.index, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

            # Plot 2: Men vs Women by Domain
            ax2 = axes[0, 1]
            domain_summary = gender_df.groupby('Domain')[['Male_Chosen', 'Female_Chosen']].mean()

            x_pos = np.arange(len(domain_summary.index))
            bars1 = ax2.bar(x_pos - width/2, domain_summary['Male_Chosen'], width,
                            label='Men Chosen', color='lightsteelblue', alpha=0.8)
            bars2 = ax2.bar(x_pos + width/2, domain_summary['Female_Chosen'], width,
                            label='Women Chosen', color='pink', alpha=0.8)

            ax2.set_xlabel('Domains')
            ax2.set_ylabel('Percentage of Gender-Related Scenarios')
            ax2.set_title('Men vs Women Preference by Domain')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(domain_summary.index)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

            # Plot 3: Heatmap - Men Chosen
            ax3 = axes[1, 0]
            try:
                pivot_male = gender_df.pivot(index='Model', columns='Domain', values='Male_Chosen')
                if not pivot_male.empty:
                    sns.heatmap(pivot_male, annot=True, fmt='.1f', cmap='Blues', ax=ax3,
                                cbar_kws={'label': 'Percentage Men Chosen'})
                    ax3.set_title('Men Preference Heatmap')
                    ax3.set_xlabel('Domain')
                    ax3.set_ylabel('Model')
            except Exception as e:
                ax3.text(0.5, 0.5, f'Heatmap Error: {str(e)}', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Men Preference Heatmap (Error)')

            # Plot 4: Model Comparison Summary
            ax4 = axes[1, 1]
            model_male_avg = gender_df.groupby('Model')['Male_Chosen'].mean().sort_values(ascending=True)

            colors = ['lightcoral' if x < 30 else 'gold' if x < 50 else 'steelblue' for x in model_male_avg.values]
            bars = ax4.barh(range(len(model_male_avg)), model_male_avg.values, color=colors)

            ax4.set_yticks(range(len(model_male_avg)))
            ax4.set_yticklabels(model_male_avg.index)
            ax4.set_xlabel('Average % Men Chosen')
            ax4.set_title('Models Ranked by Male Preference')
            ax4.grid(True, alpha=0.3)

            for i, (bar, value) in enumerate(zip(bars, model_male_avg.values)):
                ax4.text(value + 1, i, f'{value:.1f}%', va='center', ha='left')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'gender_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # GRAPH 4: Score Comparisons (from improved_bias_analyzer)
    # ========================================
    def create_score_comparisons(self):
        """Create similarity & toxicity score comparisons"""
        print("Creating Score Comparisons (Similarity + Toxicity)...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Similarity & Toxicity Comparisons Across Domains and Models', fontsize=16, fontweight='bold')

        score_data = []
        domain_names = {'healthcare': 'Healthcare', 'workplace': 'Workplace', 'smart_city': 'Smart City'}

        for model_name, df in self.model_data.items():
            for domain_key, domain_label in domain_names.items():
                domain_df = df[df['id'].str.contains(domain_key, case=False, na=False)]

                if len(domain_df) > 0:
                    avg_similarity = domain_df['similarity_score'].mean()
                    avg_toxicity = domain_df['toxicity_score'].mean() if 'toxicity_score' in domain_df.columns else np.nan

                    score_data.append({
                        'Model': model_name,
                        'Domain': domain_label,
                        'Similarity_Score': avg_similarity,
                        'Toxicity_Score': avg_toxicity
                    })

        score_df = pd.DataFrame(score_data)

        if not score_df.empty:
            # Plot 1: Similarity by Domain
            ax1 = axes[0, 0]
            pivot_sim = score_df.pivot(index='Domain', columns='Model', values='Similarity_Score')
            pivot_sim.plot(kind='bar', ax=ax1, rot=45, width=0.8)
            ax1.set_title('Expected Response Similarity by Domain')
            ax1.set_ylabel('Similarity (0–1, higher = better)')
            ax1.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Toxicity by Domain
            ax2 = axes[0, 1]
            pivot_tox = score_df.pivot(index='Domain', columns='Model', values='Toxicity_Score')
            pivot_tox.plot(kind='bar', ax=ax2, rot=45, width=0.8, color=['#999999','#b2df8a','#1f78b4','#e31a1c'])
            ax2.set_title('Toxicity by Domain')
            ax2.set_ylabel('Toxicity (0–1, lower = better)')
            ax2.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Model ranking by similarity (higher better)
            ax3 = axes[1, 0]
            model_sim_avg = score_df.groupby('Model')['Similarity_Score'].mean().sort_values(ascending=False)
            colors = ['green' if x > 0.65 else 'orange' if x > 0.55 else 'red' for x in model_sim_avg.values]
            bars = ax3.barh(range(len(model_sim_avg)), model_sim_avg.values, color=colors)
            ax3.set_yticks(range(len(model_sim_avg)))
            ax3.set_yticklabels(model_sim_avg.index)
            ax3.set_xlabel('Average Similarity')
            ax3.set_title('Models Ranked by Similarity')
            ax3.grid(True, alpha=0.3)
            for i, (bar, value) in enumerate(zip(bars, model_sim_avg.values)):
                ax3.text(value + 0.005, i, f'{value:.3f}', va='center', ha='left')

            # Plot 4: Model ranking by toxicity (lower better)
            ax4 = axes[1, 1]
            model_tox_avg = score_df.groupby('Model')['Toxicity_Score'].mean().sort_values(ascending=True)
            colors = ['green' if x < 0.05 else 'orange' if x < 0.15 else 'red' for x in model_tox_avg.values]
            bars = ax4.barh(range(len(model_tox_avg)), model_tox_avg.values, color=colors)
            ax4.set_yticks(range(len(model_tox_avg)))
            ax4.set_yticklabels(model_tox_avg.index)
            ax4.set_xlabel('Average Toxicity')
            ax4.set_title('Models Ranked by Toxicity (Lower is Better)')
            ax4.grid(True, alpha=0.3)
            for i, (bar, value) in enumerate(zip(bars, model_tox_avg.values)):
                ax4.text(value + 0.005, i, f'{value:.3f}', va='center', ha='left')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_comparisons.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # GRAPH 5: Overall Model Ranking (from improved_bias_analyzer)
    # ========================================
    def create_overall_model_ranking(self):
        """Overall model comparison & ranking (adds toxicity back in)"""
        print("Creating Overall Model Ranking...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overall Model Performance Comparison', fontsize=16, fontweight='bold')

        performance_data = []

        for model_name, df in self.model_data.items():
            if len(df) > 0:
                total = len(df)
                biased = (df['bias_detected'] == 'biased').sum()
                bias_pct = (biased / total) * 100

                avg_similarity = df['similarity_score'].mean()
                avg_severity   = df['bias_severity_score'].mean() if 'bias_severity_score' in df.columns else np.nan
                avg_toxicity   = df['toxicity_score'].mean() if 'toxicity_score' in df.columns else np.nan

                age_bias_pct    = ((df['age_bias']    != 'neutral').sum() / total) * 100
                racial_bias_pct = ((df['racial_bias'] != 'neutral').sum() / total) * 100
                gender_bias_pct = ((df['gender_bias'] != 'neutral').sum() / total) * 100

                performance_data.append({
                    'Model': model_name,
                    'Total_Bias_%': bias_pct,
                    'Age_Bias_%': age_bias_pct,
                    'Racial_Bias_%': racial_bias_pct,
                    'Gender_Bias_%': gender_bias_pct,
                    'Similarity': avg_similarity,
                    'Toxicity': avg_toxicity,
                    'Severity': avg_severity
                })

        perf_df = pd.DataFrame(performance_data)

        if not perf_df.empty:
            # Plot 1: Bias breakdown by model
            ax1 = axes[0, 0]
            width = 0.25
            x = np.arange(len(perf_df))
            parts = ['Age_Bias_%', 'Racial_Bias_%', 'Gender_Bias_%']
            cols  = ['skyblue', 'sandybrown', 'lightgreen']
            for i, (p, c) in enumerate(zip(parts, cols)):
                ax1.bar(x + i*width, perf_df[p], width, label=p.replace('_', ' ').replace('%', ''), color=c, alpha=0.85)
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(perf_df['Model'], rotation=45)
            ax1.set_ylabel('Bias %')
            ax1.set_title('Bias Types by Model (lower is better)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Similarity and Toxicity side-by-side
            ax2 = axes[0, 1]
            width = 0.35
            x = np.arange(len(perf_df))
            b1 = ax2.bar(x - width/2, perf_df['Similarity'], width, label='Similarity (higher better)', color='#1b9e77', alpha=0.85)
            b2 = ax2.bar(x + width/2, perf_df['Toxicity'], width, label='Toxicity (lower better)', color='#d95f02', alpha=0.85)
            ax2.set_xticks(x)
            ax2.set_xticklabels(perf_df['Model'], rotation=45)
            ax2.set_title('Similarity vs Toxicity by Model')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            for bar, val in zip(b1, perf_df['Similarity']):
                ax2.text(bar.get_x() + bar.get_width()/2., val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            for bar, val in zip(b2, perf_df['Toxicity']):
                ax2.text(bar.get_x() + bar.get_width()/2., val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

            # Plot 3: Rank by total bias %
            ax3 = axes[1, 0]
            bias_sorted = perf_df.sort_values('Total_Bias_%', ascending=True)
            colors = ['green' if x < 30 else 'orange' if x < 50 else 'red' for x in bias_sorted['Total_Bias_%'].values]
            bars = ax3.barh(range(len(bias_sorted)), bias_sorted['Total_Bias_%'], color=colors)
            ax3.set_yticks(range(len(bias_sorted)))
            ax3.set_yticklabels(bias_sorted['Model'])
            ax3.set_xlabel('Total Bias % (lower is better)')
            ax3.set_title('Overall Ranking by Bias')
            ax3.grid(True, alpha=0.3)
            for i, (bar, value) in enumerate(zip(bars, bias_sorted['Total_Bias_%'])):
                ax3.text(value + 1, i, f'{value:.1f}%', va='center', ha='left')

            # Plot 4: Summary table
            ax4 = axes[1, 1]
            ax4.axis('tight')
            ax4.axis('off')
            summary_cols = ['Model', 'Total_Bias_%', 'Similarity', 'Toxicity', 'Severity']
            table_df = perf_df[summary_cols].round(3).sort_values('Total_Bias_%', ascending=True)
            table = ax4.table(cellText=table_df.values,
                              colLabels=table_df.columns,
                              cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax4.set_title('Performance Summary (ranked by Total Bias %)', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_model_ranking.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # GRAPH 6: Healthcare Detailed Analysis (from bias_pattern_analyzer)
    # ========================================
    def create_healthcare_detailed_analysis(self):
        """Detailed analysis for healthcare scenarios"""
        print("Creating Healthcare Detailed Analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Healthcare Scenarios: Detailed Bias Analysis', fontsize=16, fontweight='bold')
        
        healthcare_scenarios = ['survival_decision', 'healthcare_allocation', 'insurance_access_bias', 'mental_health_priority', 'family_vs_others']
        
        healthcare_data = []
        for model_name, df in self.model_data.items():
            for scenario in healthcare_scenarios:
                scenario_df = df[df['domain'] == scenario]
                if len(scenario_df) > 0:
                    bias_pct = (scenario_df['bias_detected'] == 'biased').sum() / len(scenario_df) * 100
                    avg_similarity = scenario_df['similarity_score'].mean()
                    
                    # Age bias analysis for this scenario
                    age_bias_count = (scenario_df['age_bias'] != 'neutral').sum()
                    race_bias_count = (scenario_df['racial_bias'] != 'neutral').sum()
                    gender_bias_count = (scenario_df['gender_bias'] != 'neutral').sum()
                    
                    healthcare_data.append({
                        'Model': model_name,
                        'Scenario': scenario.replace('_', ' ').title(),
                        'Bias_Percentage': bias_pct,
                        'Similarity_Score': avg_similarity,
                        'Age_Bias_Count': age_bias_count,
                        'Race_Bias_Count': race_bias_count,
                        'Gender_Bias_Count': gender_bias_count,
                        'Count': len(scenario_df)
                    })
        
        healthcare_df = pd.DataFrame(healthcare_data)
        
        if not healthcare_df.empty:
            # Plot 1: Bias by Healthcare Scenario Type
            ax1 = axes[0, 0]
            pivot_bias = healthcare_df.pivot(index='Scenario', columns='Model', values='Bias_Percentage')
            if not pivot_bias.empty:
                pivot_bias.plot(kind='bar', ax=ax1, rot=45, width=0.8)
                ax1.set_title('Bias Percentage by Healthcare Scenario Type')
                ax1.set_ylabel('Bias Percentage (%)')
                ax1.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Similarity Scores by Healthcare Scenario
            ax2 = axes[0, 1]
            pivot_sim = healthcare_df.pivot(index='Scenario', columns='Model', values='Similarity_Score')
            if not pivot_sim.empty:
                pivot_sim.plot(kind='bar', ax=ax2, rot=45, width=0.8, colormap='viridis')
                ax2.set_title('Similarity Scores by Healthcare Scenario Type')
                ax2.set_ylabel('Similarity Score')
                ax2.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Most Problematic Healthcare Scenarios
            ax3 = axes[0, 2]
            scenario_avg = healthcare_df.groupby('Scenario')['Bias_Percentage'].mean().sort_values(ascending=True)
            if not scenario_avg.empty:
                colors = ['green' if x < 30 else 'orange' if x < 60 else 'red' for x in scenario_avg.values]
                bars = ax3.barh(range(len(scenario_avg)), scenario_avg.values, color=colors)
                ax3.set_yticks(range(len(scenario_avg)))
                ax3.set_yticklabels(scenario_avg.index, fontsize=8)
                ax3.set_xlabel('Average Bias Percentage')
                ax3.set_title('Healthcare Scenarios Ranked by Bias Level')
                ax3.grid(True, alpha=0.3)
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, scenario_avg.values)):
                    ax3.text(value + 1, i, f'{value:.1f}%', va='center', ha='left', fontsize=8)
            
            # Plot 4: Age Bias by Healthcare Scenario
            ax4 = axes[1, 0]
            age_bias_avg = healthcare_df.groupby('Scenario')['Age_Bias_Count'].mean()
            if not age_bias_avg.empty:
                ax4.bar(range(len(age_bias_avg)), age_bias_avg.values, color='skyblue', alpha=0.8)
                ax4.set_xticks(range(len(age_bias_avg)))
                ax4.set_xticklabels(age_bias_avg.index, rotation=45, fontsize=8)
                ax4.set_ylabel('Average Age Bias Count')
                ax4.set_title('Age Bias by Healthcare Scenario')
                ax4.grid(True, alpha=0.3)
            
            # Plot 5: Race Bias by Healthcare Scenario
            ax5 = axes[1, 1]
            race_bias_avg = healthcare_df.groupby('Scenario')['Race_Bias_Count'].mean()
            if not race_bias_avg.empty:
                ax5.bar(range(len(race_bias_avg)), race_bias_avg.values, color='lightcoral', alpha=0.8)
                ax5.set_xticks(range(len(race_bias_avg)))
                ax5.set_xticklabels(race_bias_avg.index, rotation=45, fontsize=8)
                ax5.set_ylabel('Average Race Bias Count')
                ax5.set_title('Race Bias by Healthcare Scenario')
                ax5.grid(True, alpha=0.3)
            
            # Plot 6: Model Performance in Healthcare (Bias vs Similarity)
            ax6 = axes[1, 2]
            model_perf = healthcare_df.groupby('Model').agg({
                'Bias_Percentage': 'mean',
                'Similarity_Score': 'mean'
            }).round(3)
            
            # Create scatter plot
            if not model_perf.empty:
                for i, model in enumerate(model_perf.index):
                    ax6.scatter(model_perf.loc[model, 'Similarity_Score'], 
                              model_perf.loc[model, 'Bias_Percentage'], 
                              s=100, label=model, alpha=0.7)
                
                ax6.set_xlabel('Average Similarity Score')
                ax6.set_ylabel('Average Bias Percentage')
                ax6.set_title('Healthcare: Bias vs Similarity by Model')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'healthcare_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # GRAPH 7: Workplace Detailed Analysis (from bias_pattern_analyzer)
    # ========================================
    def create_workplace_detailed_analysis(self):
        """Detailed analysis for workplace scenarios"""
        print("Creating Workplace Detailed Analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Workplace Scenarios: Detailed Bias Analysis', fontsize=16, fontweight='bold')
        
        workplace_scenarios = ['hiring_decision', 'promotion_decision', 'termination_bias', 'customer_service_roles', 'workplace_flexibility', 'public_representation']
        
        workplace_data = []
        for model_name, df in self.model_data.items():
            for scenario in workplace_scenarios:
                scenario_df = df[df['domain'] == scenario]
                if len(scenario_df) > 0:
                    bias_pct = (scenario_df['bias_detected'] == 'biased').sum() / len(scenario_df) * 100
                    avg_similarity = scenario_df['similarity_score'].mean()
                    
                    # Specific bias analysis
                    age_bias_count = (scenario_df['age_bias'] != 'neutral').sum()
                    race_bias_count = (scenario_df['racial_bias'] != 'neutral').sum()
                    gender_bias_count = (scenario_df['gender_bias'] != 'neutral').sum()
                    
                    workplace_data.append({
                        'Model': model_name,
                        'Scenario': scenario.replace('_', ' ').title(),
                        'Bias_Percentage': bias_pct,
                        'Similarity_Score': avg_similarity,
                        'Age_Bias_Count': age_bias_count,
                        'Race_Bias_Count': race_bias_count,
                        'Gender_Bias_Count': gender_bias_count,
                        'Count': len(scenario_df)
                    })
        
        workplace_df = pd.DataFrame(workplace_data)
        
        if not workplace_df.empty:
            # Plot 1: Bias by Workplace Scenario Type
            ax1 = axes[0, 0]
            pivot_bias = workplace_df.pivot(index='Scenario', columns='Model', values='Bias_Percentage')
            if not pivot_bias.empty:
                pivot_bias.plot(kind='bar', ax=ax1, rot=45, width=0.8)
                ax1.set_title('Bias Percentage by Workplace Scenario Type')
                ax1.set_ylabel('Bias Percentage (%)')
                ax1.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Similarity Scores by Workplace Scenario
            ax2 = axes[0, 1]
            pivot_sim = workplace_df.pivot(index='Scenario', columns='Model', values='Similarity_Score')
            if not pivot_sim.empty:
                pivot_sim.plot(kind='bar', ax=ax2, rot=45, width=0.8, colormap='plasma')
                ax2.set_title('Similarity Scores by Workplace Scenario Type')
                ax2.set_ylabel('Similarity Score')
                ax2.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Most Problematic Workplace Scenarios
            ax3 = axes[0, 2]
            scenario_avg = workplace_df.groupby('Scenario')['Bias_Percentage'].mean().sort_values(ascending=True)
            if not scenario_avg.empty:
                colors = ['green' if x < 40 else 'orange' if x < 70 else 'red' for x in scenario_avg.values]
                bars = ax3.barh(range(len(scenario_avg)), scenario_avg.values, color=colors)
                ax3.set_yticks(range(len(scenario_avg)))
                ax3.set_yticklabels(scenario_avg.index, fontsize=7)
                ax3.set_xlabel('Average Bias Percentage')
                ax3.set_title('Workplace Scenarios Ranked by Bias Level')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, scenario_avg.values)):
                    ax3.text(value + 1, i, f'{value:.1f}%', va='center', ha='left', fontsize=7)
            
            # Plot 4: Gender Bias in Workplace (Most Important)
            ax4 = axes[1, 0]
            gender_bias_avg = workplace_df.groupby('Scenario')['Gender_Bias_Count'].mean()
            if not gender_bias_avg.empty:
                ax4.bar(range(len(gender_bias_avg)), gender_bias_avg.values, color='pink', alpha=0.8)
                ax4.set_xticks(range(len(gender_bias_avg)))
                ax4.set_xticklabels(gender_bias_avg.index, rotation=45, fontsize=7)
                ax4.set_ylabel('Average Gender Bias Count')
                ax4.set_title('Gender Bias by Workplace Scenario')
                ax4.grid(True, alpha=0.3)
            
            # Plot 5: Age Bias in Workplace
            ax5 = axes[1, 1]
            age_bias_avg = workplace_df.groupby('Scenario')['Age_Bias_Count'].mean()
            if not age_bias_avg.empty:
                ax5.bar(range(len(age_bias_avg)), age_bias_avg.values, color='lightblue', alpha=0.8)
                ax5.set_xticks(range(len(age_bias_avg)))
                ax5.set_xticklabels(age_bias_avg.index, rotation=45, fontsize=7)
                ax5.set_ylabel('Average Age Bias Count')
                ax5.set_title('Age Bias by Workplace Scenario')
                ax5.grid(True, alpha=0.3)
            
            # Plot 6: Workplace Model Performance
            ax6 = axes[1, 2]
            model_perf = workplace_df.groupby('Model').agg({
                'Bias_Percentage': 'mean',
                'Similarity_Score': 'mean',
                'Gender_Bias_Count': 'mean'
            }).round(3)
            
            if not model_perf.empty:
                x_pos = np.arange(len(model_perf.index))
                width = 0.35
                
                ax6.bar(x_pos - width/2, model_perf['Bias_Percentage'], width, 
                       label='Avg Bias %', alpha=0.8, color='red')
                ax6.bar(x_pos + width/2, model_perf['Similarity_Score'] * 100, width, 
                       label='Similarity * 100', alpha=0.8, color='blue')
                
                ax6.set_xlabel('Models')
                ax6.set_ylabel('Scores')
                ax6.set_title('Workplace Performance by Model')
                ax6.set_xticks(x_pos)
                ax6.set_xticklabels(model_perf.index, rotation=45)
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'workplace_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # GRAPH 8: Moral Dilemma Analysis (from bias_pattern_analyzer)
    # ========================================
    def create_moral_dilemma_analysis(self):
        """Specific analysis for moral dilemma scenarios (Smart City)"""
        print("Creating Moral Dilemmas Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Moral Dilemma Scenarios: Detailed Analysis', fontsize=16, fontweight='bold')
        
        dilemma_data = []
        for model_name, df in self.model_data.items():
            dilemma_df = df[df['domain'] == 'moral_dilemma']
            if len(dilemma_df) > 0:
                total_bias = (dilemma_df['bias_detected'] == 'biased').sum() / len(dilemma_df) * 100
                avg_similarity = dilemma_df['similarity_score'].mean()
                
                # Age preference in moral dilemma scenarios
                age_younger = dilemma_df['age_bias'].str.contains('younger|child', case=False, na=False).sum()
                age_older = dilemma_df['age_bias'].str.contains('older|senior', case=False, na=False).sum()
                
                dilemma_data.append({
                    'Model': model_name,
                    'Moral_Bias_Percentage': total_bias,
                    'Moral_Similarity': avg_similarity,
                    'Younger_Preferred': age_younger,
                    'Older_Preferred': age_older,
                    'Total_Moral_Scenarios': len(dilemma_df)
                })
        
        dilemma_df = pd.DataFrame(dilemma_data)
        
        if not dilemma_df.empty:
            # Plot 1: Moral Dilemma Bias by Model
            ax1 = axes[0, 0]
            ax1.bar(dilemma_df['Model'], dilemma_df['Moral_Bias_Percentage'], 
                   color=['red', 'orange', 'blue', 'green'], alpha=0.7)
            ax1.set_ylabel('Bias Percentage (%)')
            ax1.set_title('Moral Dilemma Bias by Model')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(dilemma_df['Moral_Bias_Percentage']):
                ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
            
            # Plot 2: Moral Dilemma Similarity by Model
            ax2 = axes[0, 1]
            ax2.bar(dilemma_df['Model'], dilemma_df['Moral_Similarity'], 
                   color=['red', 'orange', 'blue', 'green'], alpha=0.7)
            ax2.set_ylabel('Similarity Score')
            ax2.set_title('Moral Dilemma Similarity by Model')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(dilemma_df['Moral_Similarity']):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot 3: Age Preference in Moral Dilemmas
            ax3 = axes[1, 0]
            x_pos = np.arange(len(dilemma_df['Model']))
            width = 0.35
            
            bars1 = ax3.bar(x_pos - width/2, dilemma_df['Younger_Preferred'], width, 
                           label='Younger Preferred', color='lightblue', alpha=0.8)
            bars2 = ax3.bar(x_pos + width/2, dilemma_df['Older_Preferred'], width, 
                           label='Older Preferred', color='lightcoral', alpha=0.8)

            ax3.set_xlabel('Models')
            ax3.set_ylabel('Count')
            ax3.set_title('Age Preference in Moral Dilemmas')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(dilemma_df['Model'], rotation=0, ha='center')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                if bar1.get_height() > 0:
                    ax3.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.1, 
                            f'{int(bar1.get_height())}', ha='center', va='bottom')
                if bar2.get_height() > 0:
                    ax3.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.1, 
                            f'{int(bar2.get_height())}', ha='center', va='bottom')

            # Plot 4: Remove/hide the fourth plot
            ax4 = axes[1, 1]
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'moral_dilemma_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # GRAPH 9: Comprehensive Scenario Ranking (from bias_pattern_analyzer)
    # ========================================
    def create_comprehensive_scenario_ranking(self):
        """Create comprehensive ranking of all scenario types"""
        print("Creating Comprehensive Scenario Ranking...")
        
        # Analyze all scenarios
        all_scenario_data = []
        
        for model_name, df in self.model_data.items():
            for scenario_key, scenario_label in self.scenario_categories.items():
                scenario_df = df[df['domain'] == scenario_key]
                
                if len(scenario_df) > 0:
                    bias_pct = (scenario_df['bias_detected'] == 'biased').sum() / len(scenario_df) * 100
                    avg_similarity = scenario_df['similarity_score'].mean()
                    
                    all_scenario_data.append({
                        'Model': model_name,
                        'Scenario_Type': scenario_label,
                        'Scenario_Key': scenario_key,
                        'Bias_Percentage': bias_pct,
                        'Similarity_Score': avg_similarity,
                        'Count': len(scenario_df)
                    })
        
        analysis_df = pd.DataFrame(all_scenario_data)
        
        if not analysis_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle('Comprehensive Scenario Analysis: Rankings and Comparisons', fontsize=16, fontweight='bold')
        
            # Plot 1: Scenario Types Ranked by Overall Bias
            ax1 = axes[0, 0]
            scenario_bias_avg = analysis_df.groupby('Scenario_Type')['Bias_Percentage'].mean().sort_values(ascending=True)
            if not scenario_bias_avg.empty:
                colors = ['green' if x < 30 else 'orange' if x < 60 else 'red' for x in scenario_bias_avg.values]
                bars = ax1.barh(range(len(scenario_bias_avg)), scenario_bias_avg.values, color=colors)
                ax1.set_yticks(range(len(scenario_bias_avg)))
                ax1.set_yticklabels(scenario_bias_avg.index, fontsize=6)
                ax1.set_xlabel('Average Bias Percentage')
                ax1.set_title('Scenario Types Ranked by Bias Level')
                ax1.grid(True, alpha=0.3)
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, scenario_bias_avg.values)):
                    ax1.text(value + 0.5, i, f'{value:.1f}%', va='center', ha='left', fontsize=6)
        
            # Plot 2: Similarity Scores by Scenario Type
            ax2 = axes[0, 1]
            scenario_sim_avg = analysis_df.groupby('Scenario_Type')['Similarity_Score'].mean().sort_values(ascending=True)
            if not scenario_sim_avg.empty:
                colors = ['red' if x < 0.5 else 'orange' if x < 0.6 else 'green' for x in scenario_sim_avg.values]
                bars = ax2.barh(range(len(scenario_sim_avg)), scenario_sim_avg.values, color=colors)
                ax2.set_yticks(range(len(scenario_sim_avg)))
                ax2.set_yticklabels(scenario_sim_avg.index, fontsize=6)
                ax2.set_xlabel('Average Similarity Score')
                ax2.set_title('Scenario Types Ranked by Similarity')
                ax2.grid(True, alpha=0.3)
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, scenario_sim_avg.values)):
                    ax2.text(value + 0.005, i, f'{value:.3f}', va='center', ha='left', fontsize=6)
            
            # Plot 3: Top 10 Most Biased Scenarios Heatmap
            ax3 = axes[1, 0]
            top_scenarios = scenario_bias_avg.tail(10).index
            heatmap_data = analysis_df[analysis_df['Scenario_Type'].isin(top_scenarios)]
            if not heatmap_data.empty:
                pivot_heatmap = heatmap_data.pivot(index='Scenario_Type', columns='Model', values='Bias_Percentage')
                if not pivot_heatmap.empty:
                    sns.heatmap(pivot_heatmap, annot=True, fmt='.1f', cmap='Reds', ax=ax3,
                                cbar_kws={'label': 'Bias Percentage'})
                    ax3.set_title('Top 10 Most Problematic Scenarios')
                    ax3.set_xlabel('Model')
                    ax3.set_ylabel('Scenario Type')
        
            # Plot 4: Model Performance Across All Scenarios
            ax4 = axes[1, 1]
            model_performance = analysis_df.groupby('Model').agg({
                'Bias_Percentage': 'mean',
                'Similarity_Score': 'mean'
            }).round(3)
            if not model_performance.empty:
                model_performance['Performance_Score'] = (
                    model_performance['Similarity_Score'] * 0.6 -
                    model_performance['Bias_Percentage'] * 0.01 * 0.4
                )
                perf_sorted = model_performance.sort_values('Performance_Score', ascending=True)
                colors = ['red' if x < 0.4 else 'orange' if x < 0.5 else 'green' for x in perf_sorted['Performance_Score']]
                bars = ax4.barh(range(len(perf_sorted)), perf_sorted['Performance_Score'], color=colors)
                ax4.set_yticks(range(len(perf_sorted)))
                ax4.set_yticklabels(perf_sorted.index)
                ax4.set_xlabel('Performance Score')
                ax4.set_title('Overall Model Performance Ranking')
                ax4.grid(True, alpha=0.3)
                for i, (bar, value) in enumerate(zip(bars, perf_sorted['Performance_Score'])):
                    ax4.text(value + 0.005, i, f'{value:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_scenario_ranking.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ========================================
    # MAIN EXECUTION FUNCTION
    # ========================================
    def run_all_analyses(self):
        """Run all 9 comprehensive analyses"""
        print("=" * 60)
        print("STARTING COMPREHENSIVE BIAS ANALYSIS - ALL 9 GRAPHS")
        print("=" * 60)
        print(f"Graphs will be saved to: {self.output_dir}")
        
        if not self.model_data:
            print("No model data available. Please check CSV files.")
            return
            
        try:
            print("\n🔄 Graph 1/9: Age Bias Analysis...")
            self.create_age_bias_analysis()
            
            print("\n🔄 Graph 2/9: Racial Bias Analysis...")
            self.create_racial_bias_analysis()
            
            print("\n🔄 Graph 3/9: Gender Bias Analysis...")
            self.create_gender_bias_analysis()
            
            print("\n🔄 Graph 4/9: Score Comparisons...")
            self.create_score_comparisons()
            
            print("\n🔄 Graph 5/9: Overall Model Ranking...")
            self.create_overall_model_ranking()
            
            print("\n🔄 Graph 6/9: Healthcare Detailed Analysis...")
            self.create_healthcare_detailed_analysis()
            
            print("\n🔄 Graph 7/9: Workplace Detailed Analysis...")
            self.create_workplace_detailed_analysis()
            
            print("\n🔄 Graph 8/9: Moral Dilemma Analysis...")
            self.create_moral_dilemma_analysis()
            
            print("\n🔄 Graph 9/9: Comprehensive Scenario Ranking...")
            self.create_comprehensive_scenario_ranking()
            
            print("\n" + "=" * 60)
            print("✅ ALL 9 ANALYSES COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📊 Check your comprehensive graphs in: {self.output_dir}")
            print("📋 Generated files:")
            print("   1. age_bias_analysis.png")
            print("   2. racial_bias_analysis.png") 
            print("   3. gender_bias_analysis.png")
            print("   4. score_comparisons.png")
            print("   5. overall_model_ranking.png")
            print("   6. healthcare_detailed_analysis.png")
            print("   7. workplace_detailed_analysis.png")
            print("   8. moral_dilemma_analysis.png")
            print("   9. comprehensive_scenario_ranking.png")
            
        except Exception as e:
            print(f"❌ Error during comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()

# ========================================
# RUN THE COMPREHENSIVE ANALYSIS
# ========================================
if __name__ == "__main__":
    analyzer = ComprehensiveBiasAnalyzer()
    analyzer.run_all_analyses()
