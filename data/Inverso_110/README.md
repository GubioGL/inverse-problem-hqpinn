# Problema inverso de Burgers em 110 épocas: PINN clássica × PINN quântica

Estimativa da viscosidade ν com uma PINN treinada em apenas **110 épocas**, comparando dois modelos
treinados da mesma forma:
- **quântico:** pré-rede → camada quântica (RY + CNOT brickwork, ⟨Z⟩) medida no **QPU fake** → pós-rede.
  Código: [`Inverser_quantum_hardware_pinn_E110.py`](../../run/Inverser_quantum_hardware_pinn_E110.py);
- **clássico:** pré-rede → gargalo treinável de k neurônios → pós-rede.
  Código: [`Inverser_classical_pinn_E110.py`](../../run/Inverser_classical_pinn_E110.py).

**Casos:** u_t + u u_x = ν u_xx em [0, 1) periódico, u(x,0) = sin(2πx), **ν = 0,01**, tempo final
**T = 0,1** e **T = 1**. Em cada época, **N = 10** pontos de colocação novos. Os tamanhos são 3, 7 e 10
qubits no quântico e 3, 7 e 10 neurônios no clássico, com **seeds 0–10**: 33 runs por modelo e por caso.
Com a mesma seed, os dois modelos partem do mesmo ν inicial e usam os mesmos pontos de colocação.

**QPU fake:** FakeBerlin, o modelo de ruído do `ibm_berlin` simulado localmente com Aer, com 1024 shots.
Nenhum job foi enviado à IBM.

## Resultado

| Erro de ν, 33 runs | mediana / média / p90 / máx | por tamanho (3 / 7 / 10), mediana | L2 de u(x,T), mediana |
|---|---|---|---|
| T = 0,1, clássico | 7,3 / 9,8 / 20 / 30 % | 13,2 / 4,3 / 11,5 % | **0,014** |
| T = 0,1, quântico (fake) | **6,0 / 6,7 / 15 / 25 %** | 6,5 / 3,8 / 5,3 % | 0,021 |
| T = 1, clássico | **21,5 / 23,1 / 44 / 69 %** | 17,1 / 28,5 / 21,5 % | **0,073** |
| T = 1, quântico (fake) | 30,2 / 32,2 / 53 / 71 % | 36,9 / 15,9 / 39,6 % | 0,107 |

Figura principal: [`comparacao_classico_quantico_nu_0.01_E110.pdf`](../../paper/figures/comparacao_classico_quantico_nu_0.01_E110.pdf),
com uma linha por T: o erro de ν de cada run por tamanho, a mediana de ν por época com a faixa dos
quartis, e o erro de u(x,T) de cada run. Por caso também há `nu_summary_<modelo>_<caso>_E110.pdf` (ν por
época de todas as runs) e `comparacao_classico_quantico_<caso>_E110.pdf` (a melhor run de cada modelo).

Os modelos foram comparados run a run (mesmo tamanho e mesma seed), com o teste de Wilcoxon pareado:
- **T = 0,1: os dois empatam em ν.** O quântico tem a mediana menor (6,0% contra 7,3%) e é melhor em 61%
  das runs, mas a diferença não é significativa (p = 0,11). O quântico também tem menos runs ruins (p90
  de 15% contra 20%). Em u(x,T), **o clássico é melhor** (L2 0,014 contra 0,021, p < 0,001).
- **T = 1: o clássico é melhor nos dois.** Em ν, o clássico ganha em 61% das runs (21,5% contra 30,2%,
  p = 0,046). Em u(x,T), ganha em 97% das runs (L2 0,073 contra 0,107, p < 10⁻⁸). Nos dois modelos, ν é
  subestimado na maioria das runs (70% no clássico, 76% no quântico).
- **T = 1 é bem mais difícil que T = 0,1** para os dois modelos: com ν = 0,01, um choque se forma antes
  de T = 1, e 10 pontos por época dão pouca informação sobre a frente.
- **As melhores runs** chegam a ~1% de erro nos dois casos, mas isso é sorte do último lote:
  - T = 0,1: quântico 7 qubits, seed 6 (0,25%); clássico 7 neurônios, seed 2 (1,0%);
  - T = 1: clássico 3 neurônios, seed 2 (1,6%); quântico 7 qubits, seed 9 (1,7%).
- **Seeds 0–2 não representam bem as outras.** Em T = 1, o clássico tinha mediana de 10,1% nas seeds
  0–2 e tem 21,5% nas 11 seeds. O ν final oscila muito de uma época para a outra, então são necessárias
  muitas seeds para comparar.

## O treino

- **Dados nos pontos de colocação.** Em cada época, 10 pontos aleatórios novos são usados duas vezes:
  - no resíduo da EDP, r = A − ν·B, com A = u_t + u u_x e B = u_xx calculados por diferenças finitas
    no estêncil de 5 pontos;
  - como dados: a solução exata (Cole-Hopf) nesses mesmos pontos.

  No quântico, o QPU mede só esses 10 pontos por época, e o mesmo ⟨Z⟩ serve ao resíduo e aos dados.
- **Dois otimizadores:**
  - **rede:** AdamW, 16 passos por época nos pontos da época, com loss = rampa·MAE(resíduo) +
    w_d·MAE(dados). O peso dos dados w_d é fixo e bem maior que o do resíduo, e o peso do resíduo sobe
    de 0 a 1 nas primeiras épocas;
  - **ν ("varpro"), em log ν:** com a rede fixa, o ν que minimiza mean|A − ν·B| no lote é a mediana de
    A/B ponderada por |B|. A cada época, log ν é puxado na direção dele: `log ν ← log ν + β·(log ν* − log ν)`.
- Lr e β decaem em cosseno. O estado final é o da época 110 (sem L-BFGS e sem snapshot).

| Hiperparâmetro | Quântico | Clássico |
|---|---|---|
| lr da rede (AdamW), inicial → final | 3,07·10⁻² → 1,2·10⁻³ | 2,53·10⁻² → 3,2·10⁻⁴ |
| betas | (0,77; 0,99) | (0,73; 0,99) |
| weight decay | 6,4·10⁻⁶ | 9,3·10⁻⁴ |
| clip da norma do gradiente | 10 | 10 |
| peso dos dados w_d (resíduo = 1) | 85,8 | 37,8 |
| rampa do resíduo | 20 épocas | 10 épocas |
| β do varpro, inicial → final | 0,97 → 0,128 | 0,50 → 0,163 |
| passos de AdamW por época | 16 | 16 |

Esses valores foram buscados para cada modelo, com ν = 0,1 e T = 0,1, no mesmo protocolo:
- **busca:** seeds 100–105;
- **escolha:** seeds 200–219;
- **teste:** seeds 0–2.

## Ajustes em T = 1

Com ν = 0,01 e T = 1, o treino acima não representa bem u(x,T). A IC forte u = sin(2πx) + (t/T)·rede
obriga a rede a cancelar sin(2πx) linearmente em t enquanto forma o choque. Por isso esse caso, e
**só ele**, tem três ajustes (`ADJUSTMENTS` e `ADJUSTED_CASES` em
[`Inverser_classical_pinn_E110.py`](../../run/Inverser_classical_pinn_E110.py)). Os valores são os
mesmos nos dois modelos:
1. **IC fraca:** a saída da rede é u, e u(x,0) = sin(2πx) entra como uma loss MAE em 10 pontos
   aleatórios de t = 0 por época, com peso 10. No quântico, esses pontos são medidos no mesmo job dos
   pontos de colocação (20 pontos por job em vez de 10; continuam 110 jobs).
2. **Aquecimento:** nas 10 primeiras épocas, só a loss de dados treina a rede e ν fica congelado. A rampa
   do resíduo começa depois.
3. **lr por platô:** o lr da rede fica constante e é dividido por 1,5 quando a loss da época não cai 1%
   em 5 épocas, no lugar do cosseno.

`--treino-antigo` roda T = 1 sem os ajustes (na subpasta `treino_antigo/`).

### Como foram escolhidos (testes de 2026-10-22/23; os dados e os scripts dos testes foram apagados e estão no histórico do git)
- **Clássico:** 147 variantes de treino, cada uma mudando um ingrediente. Foram testados a condição
  inicial, 10 otimizadores, 5 técnicas de ajuste do lr ou dos pesos pela loss e o aquecimento sem a EDO.
  A escolha usou 60 runs (seeds 200–219).
  - O que mais ajudou foi tirar a IC forte: o L2 de u(x,T) caiu de 0,23 para 0,065.
  - Com a IC fraca, o lr por platô e o aquecimento com ν congelado, a mediana do erro de ν caiu de
    38,5% para 12,7%.
  - Não ajudaram: trocar o AdamW (L-BFGS é o pior), LRA, ReLoBRaLo e lr proporcional à loss, e um
    varpro com memória das épocas anteriores.
- **Quântico:** 52 variantes num proxy do fake, com 60 runs cada. O proxy usa o polinômio calibrado
  mais ruído com o RMS da calibração, leva ~3 s por run, e o ν dele tem correlação de 0,75 com o do
  fake.
  - Os ajustes do clássico também ajudam no quântico: a mediana cai de 32,7% para 20,7% (p = 0,002).
  - Aumentar o peso da IC (w_ic = 85,8) deu 16,3%, mas no limite (p = 0,06) e sem confirmação no fake,
    por isso ficou w_ic = 10.
  - Mais pontos por época ajudam muito: 20 pontos dão 10,5% e 40 pontos dão 4,2%. Mas isso muda o
    protocolo, e os dois modelos ficam com 10 pontos.
  - Não ajudaram: lr, β do varpro, passos internos, rampa, w_d, duração do aquecimento e média de ν nas
    últimas épocas.
- **Ruído do fake:** o simulador não fixa a seed. O mesmo treino rodado duas vezes nas seeds 0–2 deu
  27,2% e 19,4% de mediana. Diferenças pequenas só aparecem com muitas runs.

## O QPU no laço (quântico)
1. **Calibração:** 1 job por número de qubits, reaproveitado por todas as runs. A resposta medida em
   todas as cadeias do chip é ajustada por um polinômio trigonométrico de grau 1 em cada ângulo do cone
   causal.
2. **Treino:** 110 jobs, um por época. O forward usa os valores medidos no QPU, e o gradiente passa
   pelo polinômio (straight-through). A cada 10 épocas, o polinômio é reajustado com todas as medidas.
3. **Validação:** 1 job. u(x,T) em 100 pontos, com a rede treinada rodando no QPU.

No fake, com 12 runs em paralelo numa máquina de 16 núcleos, uma run leva ~7–8 min (mediana) e um caso
(33 runs) leva ~35–45 min. O clássico leva ~3 s por run.

## Limitações
- O fake não reproduz crosstalk, deriva no tempo nem a mitigação do runtime. O caminho para o QPU real
  (`--real`) está no script, mas **não foi executado**.
- O clássico roda sem ruído, e o quântico com o ruído do hardware simulado. O fake não fixa a seed do
  simulador: rodar de novo o quântico dá números um pouco diferentes.
- Com 10 pontos por época, o ν final depende muito do último lote: ele oscila ±30% entre épocas, mais no
  quântico, que tem quedas para ~10⁻³ em lotes ruins.
- Os ajustes de T = 1 foram escolhidos no clássico e testados no quântico só no proxy e em poucas runs
  no fake.

## Pastas
```
Inverso_110/
├── README.md
├── quantico/                       calibration_NQ{3,7,10}_fake_berlin.npz (independe de T)
│   ├── T_0.1_nu_0.01/              33 runs (3/7/10 qubits × seeds 0–10)
│   └── T_1.0_nu_0.01/              33 runs, treino com os ajustes de T = 1
└── classico/
    ├── T_0.1_nu_0.01/              33 runs (3/7/10 neurônios × seeds 0–10)
    └── T_1.0_nu_0.01/              33 runs, treino com os ajustes de T = 1
  cada run: model_*.pt, run_*.json, losshist_*.npz (ν e loss por época), validation_*.npz (u(x,T));
  e summary.json por caso
```
Os casos com ν = 0,1 (T_0.1 e T_1.0, das figuras atuais do paper) foram apagados em 2026-10-23. Eles estão
no git, no commit `137ca0a`.

## Como rodar
Os comandos e a preparação do ambiente para o quântico estão no [README principal](../../README.md),
na seção *Running the 110-epoch cases*. Em resumo, a partir da raiz do repositório:
```
python run/Inverser_classical_pinn_E110.py --tfin 0.1 --nu 0.01            # clássico, seeds 0–10
python run/Inverser_classical_pinn_E110.py --tfin 1 --nu 0.01              # idem, T = 1 (com os ajustes)
python run/Inverser_quantum_hardware_pinn_E110.py --tfin 0.1 --nu 0.01 --workers 12   # quântico no fake
python run/Inverser_quantum_hardware_pinn_E110.py --tfin 1 --nu 0.01 --workers 12
python run/plots/plot_inverse_E110.py                                      # todas as figuras -> paper/figures/
```
Os scripts de simulação também gravam `loss_*.png`, `nu_history_*.png` e `solution_*.png` por run. São
prévias, fora do git, que podem ser apagadas.
