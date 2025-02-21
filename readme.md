# Applicazione dell'Apprendimento per Rinforzo nella Manipolazione Robotica

## Sommario
- [Introduzione](#introduzione)
- [Dettagli dell'Ambiente](#dettagli-dell-ambiente)
  - [Spazio degli Stati](#spazio-degli-stati)
  - [Spazio delle Azioni](#spazio-delle-azioni)
- [Algoritmi](#algoritmi)
  - [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Ambienti Personalizzati](#ambienti-personalizzati)
  - [Ostacoli Fissi](#ostacoli-fissi)
  - [Ostacoli Randomizzati](#ostacoli-randomizzati)
  - [Obiettivo Randomizzato](#obiettivo-randomizzato)
- [Analisi delle Performance](#analisi-delle-performance)
  - [Varianti di Densità](#varianti-di-densita)
- [Conclusioni](#conclusioni)
- [Riferimenti](#riferimenti)

## Introduzione
Questo progetto indaga l'applicazione degli algoritmi di apprendimento per rinforzo nei compiti di manipolazione robotica, concentrandosi sui trade-off tra politiche specializzate e adattabili. Lo studio esamina specificamente l'ambiente Pusher della suite Gym, confrontando le performance degli algoritmi Soft Actor-Critic (SAC) e Proximal Policy Optimization (PPO) nell'addestramento di un braccio robotico per spingere oggetti verso posizioni target. Vengono esplorate varie modifiche ambientali, inclusi ostacoli fissi e randomizzati, posizioni dinamiche degli obiettivi e variazioni di densità.

## Dettagli dell'Ambiente

### Spazio degli Stati
Lo spazio degli stati dell'ambiente Pusher è continuo e consiste di 20 variabili, che includono posizioni delle articolazioni, velocità, posizioni degli oggetti e posizioni degli obiettivi.

### Spazio delle Azioni
Lo spazio delle azioni è continuo e consiste di sette variabili, ciascuna rappresentante la coppia applicata a una specifica articolazione del braccio robotico.

## Algoritmi

### Soft Actor-Critic (SAC)
SAC è un algoritmo di apprendimento per rinforzo off-policy che ottimizza sia una rete di policy che una rete Q-function, incorporando la massimizzazione dell'entropia per incoraggiare l'esplorazione.

### Proximal Policy Optimization (PPO)
PPO è un metodo on-policy che migliora la stabilità degli aggiornamenti delle politiche utilizzando una funzione obiettivo con limitazione dei clip, restrigendo gli aggiornamenti delle politiche entro un certo intervallo per prevenire cambiamenti drastici.

## Ambienti Personalizzati

### Ostacoli Fissi
È stato creato un ambiente personalizzato con tre ostacoli fissi, che penalizza l'agente per le collisioni per incoraggiare l'apprendimento di una politica efficace.

### Ostacoli Randomizzati
Un ambiente con posizionamenti randomizzati degli ostacoli è stato progettato per migliorare l'adattabilità dell'agente, rendendolo più robusto in scenari dinamici.

### Obiettivo Randomizzato
Un setup con posizioni degli obiettivi randomizzate introduce un ulteriore livello di complessità, richiedendo all'agente di generalizzare il proprio comportamento per spostare con successo l'oggetto verso varie posizioni target.

## Analisi delle Performance

### Varianti di Densità
È stato analizzato l'impatto delle variazioni della densità degli oggetti e del braccio robotico. I risultati hanno mostrato che variazioni minori di densità non alterano significativamente le performance, mentre aumenti sostanziali influenzano la capacità dell'agente di completare i compiti.

## Conclusioni
Questa ricerca evidenzia l'importanza della variabilità ambientale nell'addestramento delle politiche di apprendimento per rinforzo, dimostrando che SAC supera PPO nell'ambiente Pusher. L'addestramento con ostacoli e obiettivi randomizzati porta a politiche più robuste e adattabili, utili per il trasferimento dalla simulazione alla realtà nella robotica.

## Riferimenti
1. R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction (Second Edition).
2. J. Kober, J. A. Bagnell, and J. Peters, “Reinforcement learning in robotics: A survey,” The International Journal of Robotics Research, 2013.
3. P. Kormushev, S. Calinon, and D. G. Caldwell, “Reinforcement learning in robotics: Applications and real-world challenges,” 2013.
4. S. H\" ofer, K. Bekris, A. Handa, J. C. Gamboa, F. Golemo, M. Mozian, ... and M. White, “Perspectives on sim2real transfer for robotics: A summary of the R: SS 2020 workshop,” 2020.
5. J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World,” arXiv, Mar. 20, 2017.
6. X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Sim-to-real transfer of robotic control with dynamics randomization,” 2018.
7. T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.”
8. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017.
9. Farama Foundation, “Pusher- Gymnasium Documentation,” Available: https://gymnasium.farama.org/environments/mujoco/pusher/.
10. Y. Liu, K. L. Man, T. R. Payne, and Y. Yue, “Evaluating and Selecting Deep Reinforcement Learning Models for Optimal Dynamic Pricing: A Systematic Comparison of PPO, DDPG, and SAC,” Jan. 2024, doi: 10.1145/3640824.3640871.
11. Stable-Baselines3 Docs- Reliable Reinforcement Learning Implementations https://stable-baselines3.readthedocs.io/en/master/
12. MuJoco Documentation https://mujoco.readthedocs.io/en/stable/overview.html
