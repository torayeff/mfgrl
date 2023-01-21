# Problem formulation and environment description

## Problem formulation
The optimal configuration selection for the demand satisfaction problem is formulated as below:

**Given**:
- **Demand, $D$**: the number of products required to produce.
- **Demand time, $T_D$**: maximum allowed time to produce the demanded products.
- **Set of manufacturing configurations, $\mathbb{M}$**: a manufacturing configuration is a group of resources that have the capability to produce the demanded product. The total number of unique manufacturing configurations is $M = {|\mathbb{M}|}$. Each manufacturing configuration has the following attributes:
    - **Incurring cost**: the cost of purchasing the manufacturing configuration.
    - **Recurring cost**: the cost of running a manufacturing configuration for 1 unit of time.
    - **Production rate**: the number of products produced by the manufacturing configuration per 1 unit of time.
    - **Setup time**: the time required to set up the manufacturing configuration.
- **Buffer size, $B$**: the maximum number of allowed manufacturing configurations to purchase.

**Problem**: Find the multiset of manufacturing configurations that can meet the given demand in the given demand time with minimum cost, where the cost is the sum of incurred and recurred costs.


## Environment description
### State and action space
The **state** of the environment is an $(2 + 6B + 4M)$-dimensional vector consisting of the following information:
- Remaining demand: $D_r \in \mathbb{Z}^+$, at initialization $D_r = D$
- Remaining demand time: $T_r \in \mathbb{Z}^+$, at initialization $T_r = T_D$
- Incurred costs of purchased manufacturing configurations: $I \in \mathbb{R}_{>0}^B$
- Recurring costs of purchased manufacturing configurations: $R \in \mathbb{R}_{>0}^B$
- Production rates of purchased manufacturing configurations: $P \in \mathbb{R}_{>0}^B$
- Setup times, i.e., the time required to set up the newly purchased manufacturing configuration: $U \in \mathbb{R}_{>0}^B$
- Statuses of purchased manufacturing configurations, i.e., whether the configuration is producing or in the setup (or maintenance) phase: $S \in \mathbb{R}_{\geq0, \leq1}^B$
- Produced products, i.e., the number of products produced by each of the purchased manufacturing configurations: $O \in \mathbb{R}_{\geq0}^B$
- Market incurring costs, i.e., stochastically changing the purchase prices of manufacturing configurations in the market: $\mathcal{I} \in \mathbb{R}_{>0}^M$
- Market recurring costs, i.e., stochastically changing the recurring costs of manufacturing configurations in the market: $\mathcal{R} \in \mathbb{R}_{>0}^M$
- Market production rates, i.e., stochastically changing the production rates of manufacturing configurations in the market: $\mathcal{P} \in \mathbb{R}_{>0}^M$
- Market setup times, i.e., stochastically changing the setup times of manufacturing configurations in the market: $\mathcal{U} \in \mathbb{R}_{>0}^M$

The valid **action** in the environment is an integer between $0$ and $M$ inclusive, i.e., $a \in [0, M]$:
$$
    \textit{Step}(a) = \begin{cases} 
      ``\text{buy configuration } a" \text{, if } 0\leq a < M\\
      \text{``continue production", otherwise}
   \end{cases}
$$
where ***Step($a$)*** makes one episode step in the environment.

### Environment dynamics
- The selected action by an agent affects the dynamics of the environment as follows:
    - Action "buy configuration $a$" adds a configuration $a$ into the production buffer. It also pauses the remaining demand time, $T_r$, in the environment. Counter-intuitively, stopping the remaining demand time resembles a decision-making process in the real world where purchasing decisions can be made while production is still running.
    - Continuing production decreases the remaining demand time.
    - An agent can make purchase decisions until the buffer is full. As soon as the buffer is full, an agent exceeds all its action choices, and the environment advances independently until the termination criteria are reached.
    - The environment terminates when the condition "$D_r\le0 \text{ OR } T_r\le0$" is met.

### Learning principles
- Two main **learning principles** are defined for an agent:
    - P1: Agent must meet the demand at any cost
    - P2: Total cost must be minimized

- The **first learning principle** is enforced by giving a high penalty if the demand is missed within the given demand time. The penalty is defined as follows as a function of the remaining demand:

$$
    J(D_r) = \begin{cases} 
      -D_rK, & \text{if } D_r > 0\\
      0, & \text{otherwise}
   \end{cases}
$$

where $K = R_{max} + 1$ is a penalty coefficient, and $R_{max} = B(\max{\mathcal{I}} + T_D\max{\mathcal{R}})$. The above equation is derived from inequality $DK > R_{\text{max}} + (D - 1)K$.

- The **second learning principle** is enforced by giving a negative cost of purchased manufacturing configuration if an action is a "buy configuration" or by giving a sum of negative recurring costs of running manufacturing configurations if an action is a "continue production" action.
