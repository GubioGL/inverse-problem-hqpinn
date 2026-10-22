# Problema inverso de Burgers em 110 épocas: PINN clássica × PINN quântica

Estimativa da viscosidade ν com uma PINN treinada em apenas **110 épocas**, comparando dois modelos
treinados da mesma forma:
- **quântico:** pré-rede → camada quântica (RY + CNOT brickwork, ⟨Z⟩) medida no **QPU fake** → pós-rede.
  Código: [`Inverser_quantum_hardware_pinn_E110.py`](../../run/Inverser_quantum_hardware_pinn_E110.py);
- **clássico:** pré-rede → gargalo treinável de k neurônios → pós-rede.
  Código: [`Inverser_classical_pinn_E110.py`](../../run/Inverser_classical_pinn_E110.py).

**Caso:** u_t + u u_x = ν u_xx em [0, 1) periódico, u(x,0) = sin(2πx), **ν = 0,1**, **T = 0,1**,
**N = 10** pontos de colocação por época, seeds 0, 1, 2. Os tamanhos são 3, 7 e 10 qubits no quântico
e 3, 7 e 10 neurônios no clássico.

**QPU fake:** FakeBerlin, o modelo de ruído do `ibm_berlin` simulado localmente com Aer, com 1024 shots.
Nenhum job foi enviado à IBM.

## Resultado

ν aprendido (erro relativo) nas seeds 0, 1, 2:

| Quântico (QPU fake) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 3 qubits | 0,1052 (5,2%) | 0,0930 (7,0%) | **0,1000 (0,0%)** |
| 7 qubits | 0,0988 (1,2%) | 0,0932 (6,8%) | 0,1015 (1,5%) |
| 10 qubits | 0,1015 (1,5%) | 0,0934 (6,6%) | 0,0985 (1,5%) |

| Clássico | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 3 neurônios | 0,1043 (4,3%) | 0,0868 (13,2%) | 0,0917 (8,3%) |
| 7 neurônios | 0,0954 (4,6%) | 0,0862 (13,8%) | 0,0916 (8,4%) |
| 10 neurônios | **0,1020 (2,0%)** | 0,0889 (11,1%) | 0,0882 (11,8%) |

| Erro de ν | Quântico (QPU fake) | Clássico |
|---|---|---|
| Teste: 9 runs (seeds 0, 1, 2), mediana / média / máx | **1,5% / 3,5% / 7,0%** | 8,4% / 8,6% / 13,8% |
| Validação: 60 runs (seeds 200–219), mediana / média / máx | **2,2% / 2,7% / 7,6%** (no proxy do fake) | 4,0% / 5,2% / 18,4% |
| L2 relativo de u(x,T), mediana do teste | 3,4·10⁻² | 5,6·10⁻² |

**Nas mesmas 110 épocas, o modelo quântico estima ν melhor que o clássico**, mesmo com o ruído do
hardware: a mediana do erro é 1,5% contra 8,4% nas seeds de teste, e 2,2% contra 4,0% nas 60 runs de
validação. O clássico tende a **subestimar** ν (0,086–0,095 na maioria das runs). Nos dois modelos, a
seed 1 é a mais difícil.

![ν × época, quântico](nu_summary_quantico_T_0.1_E110.png)
![ν × época, clássico](nu_summary_classico_T_0.1_E110.png)

## Quântico com ν = 0,01 (T = 0,1)

Mesmo caso, com a viscosidade verdadeira 10× menor (`--tfin 0.1 --nu 0.01`). Pastas: `quantico/T_0.1_nu_0.01/`
(otimizador novo) e `quantico/T_0.1_nu_0.01/otimizador_antigo_inner_steps_1/` (o otimizador da tabela acima).

**Por que piora.** Até T = 0,1 o escoamento é quase invíscido: u_t e u u_x quase se cancelam e
A = u_t + u u_x vale em média 0,28, contra 2,14 com ν = 0,1. O sinal de ν é ~8× menor, e o mesmo erro
relativo nas derivadas da rede vira um erro ~8× maior em ν. O limite não é o ruído do QPU (no proxy sem
ruído de shots o erro é o mesmo) nem as diferenças finitas (na solução exata o estimador acerta ν com
0,02%), e sim a convergência da rede em 110 épocas: com 330 épocas o erro cai à metade. A busca só nos
hiperparâmetros (80 configurações) não melhorou nada na validação.

**O otimizador novo:** `inner_steps = 16` passos de AdamW por época, todos nos pontos da época e na
**mesma** medida do QPU (a correção straight-through é mantida). Continuam 110 jobs; os demais
hiperparâmetros são os de antes.

| Erro de ν, quântico | ν = 0,01, otimizador antigo | ν = 0,01, `inner_steps = 16` | ν = 0,1, `inner_steps = 16` |
|---|---|---|---|
| Validação no proxy, 60 runs (seeds 200–219), mediana / média / máx | 14,5% / 21,3% / 82% | **5,7% / 6,7% / 23%** | **1,0% / 1,1% / 3,9%** (antes 2,1% / 3,0% / 10,9%) |
| Teste no fake, 9 runs (seeds 0, 1, 2), mediana / média / máx | 10,7% / 14,7% / 27,8% | 8,1% / 11,1% / 32,8% | não rodado |
| L2 relativo de u(x,T) no fake, mediana | 3,9·10⁻² | 1,7·10⁻² | |

No teste no fake o ganho em ν é menor que no proxy (com 9 runs, a diferença está dentro do ruído) e o
caso 3 qubits, seed 2 piorou (32,8%); u(x,T) melhora mais que o dobro. A calibração foi refeita em
2026-10-20 (os arquivos antigos sumiram): mesmas cadeias do chip, ajuste estatisticamente equivalente
(RMS 0,0312 → 0,0312 em 3 qubits). Atenção: `inner_steps = 16` agora é o padrão do script, então rodar de
novo o caso ν = 0,1 **não** reproduz a tabela acima (feita com 1 passo por época).

## Comparação dos melhores casos

![comparação](comparacao_classico_quantico_T_0.1_E110.png)

| Melhor caso (menor erro de ν) | ν aprendido | Erro de ν | L2 relativo de u(x, T=0,1) |
|---|---|---|---|
| Quântico, 3 qubits, seed 2 (u calculado no QPU fake) | 0,10000 | 0,00% | 4,9·10⁻² |
| Clássico, 10 neurônios, seed 0 | 0,10198 | 1,98% | 4,1·10⁻² |

- **ν:** os dois chegam perto de 0,1 por volta das épocas 20–40. O quântico fica em torno do alvo e
  termina em cima dele. O clássico cai para ~0,07 entre as épocas ~55 e 65, só volta perto de 0,1
  depois da época ~80 e termina 2% acima.
- **u(x,T):** os dois melhores casos têm erro parecido, ~4–5%. O quântico foi escolhido pelo ν, não
  por u. As runs quânticas com 10 qubits têm o melhor u(x,T) (L2 2,3·10⁻²), com erro de ν de 1,5%.

## O treino (igual nos dois modelos)

- **Dados nos pontos de colocação.** Em cada época, 10 pontos aleatórios novos são usados duas vezes:
  - no resíduo da EDP, r = A − ν·B, com A = u_t + u u_x e B = u_xx calculados por diferenças finitas
    no estêncil de 5 pontos;
  - como dados: a solução exata (Cole-Hopf) nesses mesmos pontos.

  No quântico, o QPU mede só esses 10 pontos por época, e o mesmo ⟨Z⟩ serve ao resíduo e aos dados.
- **Dois otimizadores:**
  - **rede:** AdamW, com loss = rampa·MAE(resíduo) + w_d·MAE(dados). O peso dos dados w_d é fixo e
    bem maior que o do resíduo, e o peso do resíduo sobe de 0 a 1 nas primeiras épocas;
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
| passos de AdamW por época | 1 nas runs acima; 16 no script atual | 1 |

Os valores foram buscados para cada modelo com o mesmo protocolo:
1. **Busca:** seeds 100–105 × 3 tamanhos.
2. **Escolha:** seeds 200–219, 60 runs.
3. **Teste:** seeds 0, 1, 2, nunca usadas para escolher nada.

No quântico, a busca rodou na camada quântica exata e a escolha no proxy do fake (o polinômio calibrado
no FakeBerlin mais o ruído de 1024 shots). Com os hiperparâmetros do quântico, o clássico teria mediana
de 6,7% na validação, contra 4,0% com os seus próprios.

## O QPU no laço (quântico)
1. **Calibração:** 1 job por número de qubits, reaproveitado por todas as runs. A resposta medida em
   todas as cadeias do chip é ajustada por um polinômio trigonométrico de grau 1 em cada ângulo do cone
   causal.
2. **Treino:** 110 jobs, um por época. O forward usa os valores medidos no QPU, e o gradiente passa
   pelo polinômio (straight-through). A cada 10 épocas, o polinômio é reajustado com todas as medidas.
3. **Validação:** 1 job. u(x,T) em 100 pontos, com a rede treinada rodando no QPU.

No fake, uma run leva ~10 min (3 e 7 qubits) ou ~12,5 min (10 qubits). O clássico leva ~2 s.

## Limitações
- O fake não reproduz crosstalk, deriva no tempo nem a mitigação do runtime. O caminho para o QPU real
  (`--real`) está no script, mas **não foi executado**.
- O clássico roda sem ruído, e o quântico com o ruído do hardware simulado.
- Com 10 pontos ruidosos por época, um lote ruim pode puxar ν para longe. Exemplo: 3 qubits, seed 1,
  época ~54 em `nu_summary_quantico_T_0.1_E110.png`.

## Pastas
```
Inverso_110/
├── README.md
├── comparacao_classico_quantico_<caso>_E110.png  melhor caso de cada modelo: ν × época e u(x,T)
├── nu_summary_<quantico|classico>_<caso>_E110.png  ν × época de todas as runs de um modelo
├── quantico/                                calibration_NQ{3,7,10}_fake_berlin.npz (independe de T)
│   ├── T_0.1/                               as 9 runs com T = 0,1
│   ├── T_1.0/                               as runs com T = 1
│   └── T_0.1_nu_0.01/                       T = 0,1, ν = 0,01 (+ otimizador_antigo_inner_steps_1/)
└── classico/
    ├── T_0.1/                               as 9 runs com T = 0,1
    ├── T_1.0/                               as 9 runs com T = 1
    └── T_1.0_nu_0.01/                       as 9 runs com T = 1, ν = 0,01
      cada run: model_*.pt, run_*.json, losshist_*.npz (ν e loss por época), validation_*.npz (u(x,T)),
      loss_*.png, nu_history_*.png, solution_*.png; e summary.json
```

## Como rodar
No ambiente conda `qiskit`, a partir da raiz do repositório:
```
python Inverser_quantum_hardware_pinn_E110.py                 # quântico no fake: 3/7/10 qubits × seeds 0/1/2, T = 1
python Inverser_quantum_hardware_pinn_E110.py --tfin 0.1      # idem com T = 0,1 (o caso da tabela acima)
python Inverser_quantum_hardware_pinn_E110.py --qubits 3      # só 3 qubits
python Inverser_classical_pinn_E110.py                        # clássico: 3/7/10 neurônios × seeds 0/1/2, T = 1
python Inverser_classical_pinn_E110.py --tfin 0.1             # idem com T = 0,1
python plot_inverse_E110.py                                   # figuras em Inverso_110/: nu_summary + comparação, todos os casos
python plot_inverse_E110.py --tfin 0.1                        # só o caso T = 0,1 (ν = 0,1); --nu 0.01 para T_0.1_nu_0.01
```
Os scripts de simulação salvam só as runs e o summary.json; as figuras de resumo
(`nu_summary_<modelo>_<caso>_E110.png`, `comparacao_classico_quantico_<caso>_E110.png`, todas em `Inverso_110/`)
saem de `plot_inverse_E110.py`,
que lê os `losshist_*.npz` e `validation_*.npz` (esses arquivos não vão para o git).
