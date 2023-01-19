Philosophy
1) Meet the demand at any cost
2) Minimise the cost

Edge cases
1) Max. rec. cost R = R_max, Remaining D = 0
2) Min. rec. cost R = 0, Remaining D = 2000


$0 + D * k > R_{max} + (D - 1) * k$

$k > R_{max}$

Penalty for missing the demand:
$-1.0 * D_r * k$ where $D_r$ is a remaining demand
