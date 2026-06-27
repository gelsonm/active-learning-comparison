"""
Active Learning: Uncertainty-Based Query Strategies for Label-Efficient Classification
Author: Moirangthem Gelson Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444466',
    'axes.labelcolor':  '#ccccdd',
    'text.color':       '#ccccdd',
    'xtick.color':      '#aaaacc',
    'ytick.color':      '#aaaacc',
    'grid.color':       '#2a2a4a',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titlepad':    12,
    'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#444466',
})

COLORS = {
    'random':     '#607d8b',
    'entropy':    '#7c4dff',
    'margin':     '#00e5ff',
    'least_conf': '#ff6d00',
}

# ─── Load Dataset (built-in, no download) ─────────────────────────────────────
print("Loading dataset...")
data = load_digits()
X, y = data.data, data.target

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

print(f"Dataset       : sklearn Digits (handwritten digit recognition)")
print(f"Classes       : {len(np.unique(y))} (digits 0-9)")
print(f"Features      : {X.shape[1]} (8x8 pixel values)")
print(f"Pool (unlabelled): {X_pool.shape[0]} samples")
print(f"Test           : {X_test.shape[0]} samples")


# ─── Query Strategy Functions ─────────────────────────────────────────────────
def entropy_query(model, X_pool, n_query):
    proba = np.clip(model.predict_proba(X_pool), 1e-10, 1.0)
    entropy = -np.sum(proba * np.log(proba), axis=1)
    return np.argsort(entropy)[-n_query:]

def margin_query(model, X_pool, n_query):
    proba = model.predict_proba(X_pool)
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return np.argsort(margin)[:n_query]

def least_confidence_query(model, X_pool, n_query):
    proba = model.predict_proba(X_pool)
    confidence = np.max(proba, axis=1)
    return np.argsort(confidence)[:n_query]

def random_query(n_pool, n_query, rng):
    return rng.choice(n_pool, n_query, replace=False)


# ─── Active Learning Loop ─────────────────────────────────────────────────────
def run_active_learning(X_pool, y_pool, X_test, y_test,
                         query_strategy, n_initial=10, n_query=10, n_rounds=40, seed=SEED):
    rng = np.random.default_rng(seed)
    labelled_idx   = rng.choice(len(X_pool), n_initial, replace=False)
    unlabelled_idx = np.setdiff1d(np.arange(len(X_pool)), labelled_idx)

    accuracies, n_labelled = [], []

    for _ in range(n_rounds):
        model = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
        model.fit(X_pool[labelled_idx], y_pool[labelled_idx])
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies.append(acc)
        n_labelled.append(len(labelled_idx))

        if len(unlabelled_idx) < n_query:
            break

        X_un = X_pool[unlabelled_idx]
        if   query_strategy == 'random':     chosen_local = random_query(len(X_un), n_query, rng)
        elif query_strategy == 'entropy':    chosen_local = entropy_query(model, X_un, n_query)
        elif query_strategy == 'margin':     chosen_local = margin_query(model, X_un, n_query)
        elif query_strategy == 'least_conf': chosen_local = least_confidence_query(model, X_un, n_query)

        new_idx        = unlabelled_idx[chosen_local]
        labelled_idx   = np.concatenate([labelled_idx, new_idx])
        unlabelled_idx = np.setdiff1d(unlabelled_idx, new_idx)

    return np.array(n_labelled), np.array(accuracies)


strategies = {
    'random':     'Random Sampling (Baseline)',
    'entropy':    'Max Entropy Sampling',
    'margin':     'Margin Sampling',
    'least_conf': 'Least Confidence Sampling',
}

print("\nRunning active learning simulations ...")
results = {}
for key in strategies:
    n_lab, acc = run_active_learning(X_pool, y_pool, X_test, y_test,
                                     query_strategy=key,
                                     n_initial=10, n_query=10, n_rounds=40)
    results[key] = (n_lab, acc)
    print(f"  [{key:12s}]  final acc: {acc[-1]:.3f}  labels used: {n_lab[-1]}")


# ─── Plot 1: Learning Curves ───────────────────────────────────────────────────
TARGET_ACC = 0.90
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Active Learning: Smarter Querying vs. Random Annotation\n'
             'Dataset: sklearn Digits (10-class handwritten digit recognition)',
             fontsize=13, fontweight='bold', color='#e0e0ff', y=1.02)

ax = axes[0]
for key, label in strategies.items():
    n_lab, acc = results[key]
    lw = 2.5 if key != 'random' else 1.5
    ls = '--' if key == 'random' else '-'
    alpha = 0.6 if key == 'random' else 1.0
    ax.plot(n_lab, acc, color=COLORS[key], lw=lw, ls=ls, alpha=alpha,
            label=label, marker='o', markersize=3)
ax.set_xlabel('Number of Labelled Samples', labelpad=8)
ax.set_ylabel('Test Accuracy', labelpad=8)
ax.set_title('Learning Curves by Query Strategy')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True)
ax.set_xlim(left=0)
ax.set_ylim(0.2, 1.0)

ax2 = axes[1]
labels_to_reach = {}
for key, label in strategies.items():
    n_lab, acc = results[key]
    idx = np.argmax(acc >= TARGET_ACC)
    labels_to_reach[label] = n_lab[idx] if acc[idx] >= TARGET_ACC else n_lab[-1]

colors_bar = [COLORS[k] for k in strategies]
bars = ax2.barh(list(labels_to_reach.keys()), list(labels_to_reach.values()),
                color=colors_bar, alpha=0.85, height=0.5, edgecolor='#aaaacc', linewidth=0.5)
for bar, val in zip(bars, labels_to_reach.values()):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
             f'{val} labels', va='center', color='#e0e0ff', fontsize=9)
ax2.set_xlabel(f'Labels Required to Reach {TARGET_ACC*100:.0f}% Accuracy', labelpad=8)
ax2.set_title(f'Label Efficiency at {TARGET_ACC*100:.0f}% Target Accuracy')
ax2.grid(True, axis='x')
ax2.set_xlim(0, max(labels_to_reach.values()) * 1.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("\nSaved: learning_curves.png")


# ─── Plot 2: Uncertainty Distributions ────────────────────────────────────────
rng2  = np.random.default_rng(SEED)
init  = rng2.choice(len(X_pool), 10, replace=False)
model_boot = LogisticRegression(max_iter=500, C=1.0, random_state=SEED)
model_boot.fit(X_pool[init], y_pool[init])

remaining = np.setdiff1d(np.arange(len(X_pool)), init)
X_rem  = X_pool[remaining]
proba  = np.clip(model_boot.predict_proba(X_rem), 1e-10, 1.0)
entropy_scores = -np.sum(proba * np.log(proba), axis=1)

top_ent   = np.argsort(entropy_scores)[-50:]
rand_50   = rng2.choice(len(entropy_scores), 50, replace=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Why Entropy Sampling Queries Different Samples Than Random',
             fontsize=13, fontweight='bold', color='#e0e0ff', y=1.02)

ax = axes[0]
ax.hist(entropy_scores,        bins=40, color='#444466', alpha=0.7, label='All unlabelled')
ax.hist(entropy_scores[top_ent], bins=20, color=COLORS['entropy'], alpha=0.9, label='Entropy-queried (top 50)')
ax.hist(entropy_scores[rand_50], bins=15, color=COLORS['random'],  alpha=0.7, label='Random-queried (50)',
        ls='--', histtype='step', lw=2)
ax.set_xlabel('Predictive Entropy (nats)', labelpad=8)
ax.set_ylabel('Count', labelpad=8)
ax.set_title('Entropy Score Distribution of Queried Samples')
ax.legend(fontsize=9)
ax.grid(True)

ax2 = axes[1]
conf_all = np.max(proba, axis=1)
ax2.hist(conf_all,              bins=40, color='#444466', alpha=0.7, label='All unlabelled')
ax2.hist(conf_all[top_ent],     bins=20, color=COLORS['entropy'], alpha=0.9, label='Entropy-queried (top 50)')
ax2.hist(conf_all[rand_50],     bins=15, color=COLORS['random'],  alpha=0.7, label='Random-queried (50)',
         ls='--', histtype='step', lw=2)
ax2.set_xlabel('Max Class Probability (Confidence)', labelpad=8)
ax2.set_ylabel('Count', labelpad=8)
ax2.set_title('Confidence of Queried Samples\nEntropy Targets Low-Confidence Regions')
ax2.legend(fontsize=9)
ax2.grid(True)

plt.tight_layout()
plt.savefig('uncertainty_distributions.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved: uncertainty_distributions.png")


# ─── Summary ──────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('  ACTIVE LEARNING EXPERIMENT - SUMMARY')
print('='*60)
rand_n, rand_acc = results['random']
rand_idx = np.argmax(rand_acc >= TARGET_ACC)
rand_need = rand_n[rand_idx] if rand_acc[rand_idx] >= TARGET_ACC else rand_n[-1]

for key, label in strategies.items():
    n_lab, acc = results[key]
    idx = np.argmax(acc >= TARGET_ACC)
    need = n_lab[idx] if acc[idx] >= TARGET_ACC else '>'
    saving = (f'{rand_need - n_lab[idx]} fewer labels ({(rand_need - n_lab[idx])/rand_need*100:.0f}%)'
              if acc[idx] >= TARGET_ACC and key != 'random' else 'baseline')
    print(f'  {label}')
    print(f'    Labels to {TARGET_ACC*100:.0f}%: {need}  |  {saving}')
    print()

print('='*60)
print('\nAll outputs saved. Experiment complete.')
