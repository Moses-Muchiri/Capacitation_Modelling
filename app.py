import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pysb import Model, Monomer, Parameter, Initial, Rule, Observable, Expression
from pysb.simulator import ScipyOdeSimulator
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support
import warnings
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

TIME_POINTS = 300
TIME_END = 5000
time = np.linspace(0, TIME_END, TIME_POINTS)

def clean_cython_cache():
    
    cython_dir = Path.home() / '.cython'
    if cython_dir.exists():
        try:
            shutil.rmtree(cython_dir)
            logger.info("Cleaned Cython cache")
        except Exception as e:
            logger.warning(f"Could not clean Cython cache: {e}")

def build_model():
    model = Model('SpermMotilityModel')
    
    Monomer('Sperm', ['state'], {'state': ['Cap', 'Decap']})
    Monomer('DF', ['b'])
    Monomer('DF_Blocker', ['b'])
    Monomer('PKA', ['state'], {'state': ['inactive', 'active']})
    Monomer('Ca', ['level'], {'level': ['low', 'high']})
    Monomer('cAMP', ['level'], {'level': ['low', 'high']})
    Monomer('cAMP_Enhancer')

    Parameter('Sperm_0', 1000)
    Parameter('DF_0', 500)
    Parameter('PKA_0', 1000)
    Parameter('Ca_0', 1000)
    Parameter('cAMP_0', 1000)
    Parameter('DF_Blocker_0', 100)
    Parameter('cAMP_Enhancer_0', 100)

    Parameter('k_bind', 1e-4)           
    Parameter('k_unbind', 1e-3)         
    Parameter('k_df_block', 1e-4)       
    Parameter('k_df_unblock', 1e-3)     
    Parameter('k_activate_PKA', 1e-3)   
    Parameter('k_deactivate_PKA', 5e-4) 
    Parameter('k_cAMP_up', 1e-3)        
    Parameter('k_cAMP_down', 5e-4)      
    Parameter('k_Ca_up', 8e-4)          
    Parameter('k_Ca_down', 4e-4)        
    Parameter('k_cAMP_drug_boost', 1e-4)

    Initial(Sperm(state='Cap'), model.parameters['Sperm_0'])
    Initial(DF(b=None), model.parameters['DF_0'])
    Initial(DF_Blocker(b=None), model.parameters['DF_Blocker_0'])
    Initial(PKA(state='inactive'), model.parameters['PKA_0'])
    Initial(Ca(level='low'), model.parameters['Ca_0'])
    Initial(cAMP(level='low'), model.parameters['cAMP_0'])
    Initial(cAMP_Enhancer(), model.parameters['cAMP_Enhancer_0'])

    Rule('Bind_DF', 
         Sperm(state='Cap') + DF(b=None) >> Sperm(state='Decap') + DF(b=None), 
         model.parameters['k_bind'])
    
    Rule('Unbind_DF', 
         Sperm(state='Decap') >> Sperm(state='Cap'), 
         model.parameters['k_unbind'])
    
    Rule('DF_Block_bind', 
         DF(b=None) + DF_Blocker(b=None) >> DF(b=1) % DF_Blocker(b=1), 
         model.parameters['k_df_block'])
    
    Rule('DF_Block_unbind', 
         DF(b=1) % DF_Blocker(b=1) >> DF(b=None) + DF_Blocker(b=None), 
         model.parameters['k_df_unblock'])
    
    Rule('Activate_PKA', 
         PKA(state='inactive') + cAMP(level='high') >> PKA(state='active') + cAMP(level='high'), 
         model.parameters['k_activate_PKA'])
    
    Rule('Deactivate_PKA', 
         PKA(state='active') >> PKA(state='inactive'), 
         model.parameters['k_deactivate_PKA'])
    
    Rule('cAMP_up', 
         cAMP(level='low') + Sperm(state='Cap') >> cAMP(level='high') + Sperm(state='Cap'), 
         model.parameters['k_cAMP_up'])
    
    Rule('cAMP_down', 
         cAMP(level='high') >> cAMP(level='low'), 
         model.parameters['k_cAMP_down'])
    
    Rule('Ca_up', 
         Ca(level='low') + PKA(state='active') >> Ca(level='high') + PKA(state='active'), 
         model.parameters['k_Ca_up'])
    
    Rule('Ca_down', 
         Ca(level='high') >> Ca(level='low'), 
         model.parameters['k_Ca_down'])
    
    Rule('cAMP_Enhancer_Boost', 
         cAMP(level='low') + cAMP_Enhancer() >> cAMP(level='high') + cAMP_Enhancer(), 
         model.parameters['k_cAMP_drug_boost'])

    Observable('Cap_Sperm', Sperm(state='Cap'))
    Observable('Decap_Sperm', Sperm(state='Decap'))
    Observable('Ca_High', Ca(level='high'))
    Observable('cAMP_High', cAMP(level='high'))
    Observable('Active_PKA', PKA(state='active'))
    Observable('Free_DF', DF(b=None))
    
    Expression('Motility_Index', 
               (model.observables['Cap_Sperm'] * model.observables['Ca_High'] * 
                model.observables['cAMP_High'] * model.observables['Active_PKA']) / 1e9)

    return model

def create_simulator(model, verbose=True):
    
    if verbose:
        logger.info("Using Python compiler to avoid Cython compilation issues")
    return ScipyOdeSimulator(model, tspan=time, compiler='python', 
                           integrator='lsoda', integrator_options={'atol': 1e-10, 'rtol': 1e-8})

global_model = build_model()
global_sim = create_simulator(global_model)

def simulate(param_tuple):
    k_bind, k_unbind, k_df_block, k_cAMP_boost = param_tuple
    
    param_values = {
        'k_bind': k_bind,
        'k_unbind': k_unbind,
        'k_df_block': k_df_block,
        'k_cAMP_drug_boost': k_cAMP_boost
    }
    
    try:
        out = global_sim.run(param_values=param_values)
        
        motility_final = out.expressions['Motility_Index'][-1]
        motility_max = np.max(out.expressions['Motility_Index'])
        motility_mean = np.mean(out.expressions['Motility_Index'][TIME_POINTS//2:])          
        
        return {
            'k_bind': k_bind,
            'k_unbind': k_unbind,
            'k_df_block': k_df_block,
            'k_cAMP_boost': k_cAMP_boost,
            'Motility_Index_final': motility_final,
            'Motility_Index_max': motility_max,
            'Motility_Index_mean': motility_mean,
            'Active_PKA_final': out.observables['Active_PKA'][-1],
            'Cap_Sperm_final': out.observables['Cap_Sperm'][-1],
            'Ca_High_final': out.observables['Ca_High'][-1],
            'cAMP_High_final': out.observables['cAMP_High'][-1],
            'success': True
        }
    except Exception as e:
        logger.error(f"Simulation failed for parameters {param_tuple}: {e}")
        return {
            'k_bind': k_bind,
            'k_unbind': k_unbind,
            'k_df_block': k_df_block,
            'k_cAMP_boost': k_cAMP_boost,
            'Motility_Index_final': 0,
            'Motility_Index_max': 0,
            'Motility_Index_mean': 0,
            'Active_PKA_final': 0,
            'Cap_Sperm_final': 0,
            'Ca_High_final': 0,
            'cAMP_High_final': 0,
            'success': False
        }

def create_visualizations(results_df, output_dir='output'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    success_df = results_df[results_df['success'] == True].copy()
    
    if len(success_df) == 0:
        logger.error("No successful simulations to visualize")
        return
    
    plt.figure(figsize=(12, 8))
    for metric in ['Motility_Index_final', 'Motility_Index_max', 'Motility_Index_mean']:
        sns.lineplot(data=success_df, x="k_cAMP_boost", y=metric, 
                    label=metric.replace('_', ' ').title(), marker="o")
    
    plt.xscale("log")
    plt.title("Motility Metrics vs cAMP Boost Rate", fontsize=14, fontweight='bold')
    plt.xlabel("k_cAMP_boost (log scale)", fontsize=12)
    plt.ylabel("Motility Index", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "motility_vs_cAMP_boost_enhanced.png", dpi=300)
    plt.close()

    fig = px.scatter_3d(success_df, 
                       x='k_cAMP_boost', 
                       y='k_df_block', 
                       z='Motility_Index_final', 
                       color='k_bind',
                       size='Motility_Index_max',
                       hover_data=['k_unbind'],
                       title="3D Parameter Space Exploration")
    
    fig.update_layout(
        scene=dict(
            xaxis_type='log', 
            yaxis_type='log',
            xaxis_title="cAMP Boost Rate",
            yaxis_title="DF Block Rate",
            zaxis_title="Final Motility Index"
        ),
        coloraxis_colorbar=dict(title='k_bind')
    )
    fig.write_html(output_path / "3d_scatter_motility_enhanced.html")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['Motility_Index_final', 'Active_PKA_final', 'Cap_Sperm_final', 'Ca_High_final']
    titles = ['Final Motility Index', 'Final Active PKA', 'Final Capacitated Sperm', 'Final High Ca2+']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = idx // 2, idx % 2
        pivot_table = success_df.pivot_table(values=metric, index='k_bind', columns='k_df_block', aggfunc='mean')
        
        sns.heatmap(pivot_table, annot=True, fmt=".2e", cmap="viridis", ax=axes[row, col])
        axes[row, col].set_title(f"{title}: k_bind vs k_df_block")
        axes[row, col].set_xlabel("k_df_block")
        axes[row, col].set_ylabel("k_bind")
    
    plt.tight_layout()
    plt.savefig(output_path / "comprehensive_heatmaps.png", dpi=300)
    plt.close()

    numeric_cols = success_df.select_dtypes(include=[np.number]).columns
    correlations = success_df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlations, dtype=bool))
    sns.heatmap(correlations, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Parameter Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "correlation_matrix.png", dpi=300)
    plt.close()

    motility_corr = correlations['Motility_Index_final'].abs().sort_values(ascending=False)
    motility_corr = motility_corr[motility_corr.index != 'Motility_Index_final']
    
    plt.figure(figsize=(10, 6))
    motility_corr.plot(kind='bar')
    plt.title("Parameter Sensitivity for Final Motility Index", fontsize=14, fontweight='bold')
    plt.xlabel("Parameters", fontsize=12)
    plt.ylabel("Absolute Correlation", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "sensitivity_analysis.png", dpi=300)
    plt.close()

    logger.info(f"Visualizations saved to {output_path}")

def main():
    
    clean_cython_cache()
    
    logger.info("Starting sperm motility simulation parameter sweep")

    _ = create_simulator(global_model, verbose=True)
    
    k_bind_vals = np.logspace(-6, -2, 5)
    k_unbind_vals = np.logspace(-4, -1, 5)
    k_df_block_vals = np.logspace(-6, -2, 5)
    k_cAMP_boost_vals = np.logspace(-6, -2, 5)

    param_grid = list(product(k_bind_vals, k_unbind_vals, k_df_block_vals, k_cAMP_boost_vals))
    
    logger.info(f"Parameter grid size: {len(param_grid)} simulations")
    
    n_cores = min(2, cpu_count())
    logger.info(f"Using {n_cores} CPU cores (reduced to avoid  Cython compilation conflicts)")

    try:
        with Pool(n_cores) as pool:
            results = list(tqdm(pool.imap(simulate, param_grid, chunksize=8), 
                              total=len(param_grid), desc="Simulating"))
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return

    results_df = pd.DataFrame(results)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / "motility_results_enhanced.csv", index=False)
    
    success_rate = results_df['success'].mean() * 100
    logger.info(f"Simulation success rate: {success_rate:.1f}%")
    
    if success_rate > 0:
        successful_results = results_df[results_df['success'] == True]
        logger.info(f"Motility Index range: {successful_results['Motility_Index_final'].min():.2e} - {successful_results['Motility_Index_final'].max():.2e}")
        
        create_visualizations(results_df, output_dir)
        
        numeric_cols = successful_results.select_dtypes(include=[np.number]).columns
        correlations = successful_results[numeric_cols].corr()["Motility_Index_final"].abs().sort_values(ascending=False)
        correlations.to_csv(output_dir / "parameter_sensitivity_enhanced.csv")
        
        logger.info("Analysis complete. Check the 'output' directory for results.")
    else:
        logger.error("No successful simulations. Check model parameters.")

if __name__ == "__main__":
    freeze_support()
    main()